import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.image as mpimg
import os

# ----------------------------------------------------------------------
# 0. ページ設定
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="⚾ MLB Analyzer Pro",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# 1. ライブラリ読み込み
# ----------------------------------------------------------------------
try:
    import pybaseball
    from pybaseball import statcast_pitcher, statcast_batter, playerid_lookup, statcast
    # キャッシュ設定は環境によってエラーの元になるので今回はオフ
except ImportError as e:
    st.error(f"ライブラリ不足: {e}")
    st.stop()
except Exception as e:
    st.error(f"初期化エラー: {e}")
    st.stop()

# ----------------------------------------------------------------------
# 2. 関数定義 (キャッシュ付き)
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600)
def search_player(name_str):
    """選手名検索 (キャッシュ有効)"""
    if not name_str: return pd.DataFrame()
    try:
        # playerid_lookupはLast Name検索
        return playerid_lookup(name_str.lower().strip())
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_statcast_data(start_dt, end_dt, p_id, b_id, game_types_list):
    """データ取得 (キャッシュ有効)"""
    try:
        s_dt = pd.to_datetime(start_dt).strftime('%Y-%m-%d')
        e_dt = pd.to_datetime(end_dt).strftime('%Y-%m-%d')
        df = pd.DataFrame()

        if p_id and b_id:
            raw = statcast_pitcher(start_dt=s_dt, end_dt=e_dt, player_id=p_id)
            if not raw.empty and 'batter' in raw.columns:
                df = raw[raw['batter'] == b_id].copy()
        elif p_id:
            df = statcast_pitcher(start_dt=s_dt, end_dt=e_dt, player_id=p_id)
        elif b_id:
            df = statcast_batter(start_dt=s_dt, end_dt=e_dt, player_id=b_id)
        else:
            # 選手指定なしは重いので注意
            df = statcast(start_dt=s_dt, end_dt=e_dt)
        
        # 試合タイプ絞り込み
        if not df.empty and game_types_list:
            if 'game_type' in df.columns:
                targets = []
                if 'P' in game_types_list: targets.extend(['F', 'D', 'L', 'W'])
                targets.extend(game_types_list)
                targets = list(set(targets))
                df = df[df['game_type'].isin(targets)]
        return df
    except:
        return pd.DataFrame()

def process_data(df_input):
    """データ加工"""
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'game_date' in df.columns: df = df.sort_values('game_date').reset_index(drop=True)

    # 必要なカラムがなければ埋める
    cols = ['balls', 'strikes', 'outs_when_up', 'launch_speed', 'launch_angle', 'woba_value', 'plate_x', 'plate_z', 'stand']
    for c in cols:
        if c not in df.columns: df[c] = np.nan

    # イベント判定
    if 'events' in df.columns:
        events = df['events'].fillna('').str.lower()
        hits = ['single', 'double', 'triple', 'home_run']
        df['is_hit'] = events.isin(hits).astype(int)
        ab_evts = hits + ['field_out', 'strikeout', 'grounded_into_double_play', 'double_play', 'fielders_choice', 'force_out']
        df['is_at_bat'] = events.isin(ab_evts).astype(int)
        pa_evts = ab_evts + ['walk', 'hit_by_pitch', 'sac_fly']
        df['is_pa_event'] = events.isin(pa_evts).astype(int)
        
        # 出塁率などの分母・分子
        df['is_obp_denom'] = (df['is_at_bat'] | events.isin(['walk', 'hit_by_pitch', 'sac_fly'])).astype(int)
        df['is_on_base'] = (df['is_hit'] | events.isin(['walk', 'hit_by_pitch'])).astype(int)
        
        # 長打率計算用の塁打数
        tb_map = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
        df['slugging_base'] = events.map(tb_map).fillna(0).astype(int)
    else:
        df['is_hit'] = 0; df['is_at_bat'] = 0; df['is_pa_event'] = 0

    # Hard Hit / Barrel
    df['is_hard_hit'] = (df['launch_speed'].fillna(0) >= 95.0).astype(int)
    ls = df['launch_speed'].fillna(0); la = df['launch_angle'].fillna(0)
    cond = (ls >= 98) & (la >= 26) & (la <= 30)
    df['is_barrel'] = np.where(cond, 1, 0)

    return df

# ----------------------------------------------------------------------
# 3. メイン処理 (If文を使ったシンプルな構成)
# ----------------------------------------------------------------------
def main():
    st.sidebar.title("⚾ MLB Analyzer Pro")

    # --- 共通設定 ---
    st.sidebar.header("1. 期間 & 試合タイプ")
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("開始", datetime.date(2025, 3, 27))
    end_date = c2.date_input("終了", datetime.date(2025, 11, 2))
    
    game_types = st.sidebar.multiselect("対象試合", ['Regular Season', 'Postseason', 'Spring Training', 'All-Star'], default=['Regular Season', 'Postseason'])
    game_type_codes = [{'Regular Season':'R', 'Postseason':'P', 'Spring Training':'S', 'All-Star':'A'}.get(x) for x in game_types]

    # --- 選手選択 (If文で段階的に表示) ---
    st.sidebar.header("2. 選手選択")
    
    selected_p_id = None
    selected_p_name = None
    selected_b_id = None
    selected_b_name = None

    # 投手検索
    p_input = st.sidebar.text_input("投手 姓 (Last Name)", key="p_in")
    if p_input:
        # 入力があったら検索実行
        p_found = search_player(p_input)
        if not p_found.empty:
            p_found['label'] = p_found['name_first'] + " " + p_found['name_last'] + " (" + p_found['mlb_played_first'].astype(str) + "-" + p_found['mlb_played_last'].astype(str) + ")"
            p_choice = st.sidebar.selectbox("投手を選択", ["指定なし"] + p_found['label'].tolist(), key="p_sel")
            
            if p_choice != "指定なし":
                row = p_found[p_found['label'] == p_choice].iloc[0]
                selected_p_id = int(row['key_mlbam'])
                selected_p_name = f"{row['name_first']} {row['name_last']}"
        else:
            st.sidebar.warning("投手が見つかりません")

    # 打者検索
    b_input = st.sidebar.text_input("打者 姓 (Last Name)", key="b_in")
    if b_input:
        # 入力があったら検索実行
        b_found = search_player(b_input)
        if not b_found.empty:
            b_found['label'] = b_found['name_first'] + " " + b_found['name_last'] + " (" + b_found['mlb_played_first'].astype(str) + "-" + b_found['mlb_played_last'].astype(str) + ")"
            b_choice = st.sidebar.selectbox("打者を選択", ["指定なし"] + b_found['label'].tolist(), key="b_sel")
            
            if b_choice != "指定なし":
                row = b_found[b_found['label'] == b_choice].iloc[0]
                selected_b_id = int(row['key_mlbam'])
                selected_b_name = f"{row['name_first']} {row['name_last']}"
        else:
            st.sidebar.warning("打者が見つかりません")

    # --- データ取得ボタン ---
    st.sidebar.markdown("---")
    if st.sidebar.button("データを取得・分析開始", type="primary"):
        
        # 警告: 両方なしの場合
        if not selected_p_id and not selected_b_id:
            st.warning("選手が指定されていません。リーグ全体のデータを取得します（時間がかかります）。")

        # ローディング表示
        with st.spinner("データを取得中..."):
            df = get_statcast_data(str(start_date), str(end_date), selected_p_id, selected_b_id, game_type_codes)
        
        if df.empty:
            st.error("該当するデータが見つかりませんでした。")
        else:
            # --- 分析画面の表示 (データがある場合のみここに入る) ---
            df = process_data(df)
            show_analysis_page(df, selected_p_name, selected_b_name, selected_p_id, selected_b_id)


# ----------------------------------------------------------------------
# 4. 分析・描画ページ (メインコンテンツ)
# ----------------------------------------------------------------------
def show_analysis_page(df, p_name, b_name, p_id, b_id):
    
    # タイトル作成
    title = "League Wide Analysis"
    if p_name and b_name: title = f"{p_name} vs {b_name}"
    elif p_name: title = f"{p_name} Pitching Analysis"
    elif b_name: title = f"{b_name} Batting Analysis"
    
    st.title(f"⚾ {title}")
    st.write(f"Total Pitches: {len(df)}")
    
    # --- フィルター設定 (Expanderで隠す) ---
    with st.expander("詳細フィルター設定", expanded=True):
        col1, col2, col3 = st.columns(3)
        pitch_types = sorted(df['pitch_type'].dropna().unique())
        pitch_sel = col1.multiselect("球種", pitch_types, default=pitch_types)
        
        stands = df['stand'].unique()
        stand_sel = col2.multiselect("打席 (L/R)", stands, default=stands)
        
        results = df['events'].dropna().unique()
        result_sel = col3.multiselect("結果", results, default=results)

    # フィルタリング実行
    df_filt = df.copy()
    if pitch_sel: df_filt = df_filt[df_filt['pitch_type'].isin(pitch_sel)]
    if stand_sel: df_filt = df_filt[df_filt['stand'].isin(stand_sel)]
    if result_sel: df_filt = df_filt[df_filt['events'].isin(result_sel)]

    if df_filt.empty:
        st.warning("フィルター後のデータがありません。")
        return

    # --- 指標サマリー ---
    st.markdown("### Key Metrics")
    # 打率 / 被打率 の切り替え
    ba_label = "Batting Avg"
    if p_id and not b_id: ba_label = "BA Against (被打率)"
    
    pa = df_filt['is_pa_event'].sum()
    ab = df_filt['is_at_bat'].sum()
    hits = df_filt['is_hit'].sum()
    ba = hits / ab if ab > 0 else 0
    
    # OPS計算
    obp = df_filt['is_on_base'].sum() / df_filt['is_obp_denom'].sum() if df_filt['is_obp_denom'].sum() > 0 else 0
    slg = df_filt['slugging_base'].sum() / ab if ab > 0 else 0
    ops = obp + slg
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PA / AB", f"{pa} / {ab}")
    c2.metric(ba_label, f"{ba:.3f}")
    c3.metric("OPS", f"{ops:.3f}")
    c4.metric("Hard Hit %", f"{df_filt['is_hard_hit'].mean():.1%}")

    # --- グラフ描画 ---
    st.markdown("### Pitch Chart (Pitcher's View)")
    
    plot_type = st.radio("グラフタイプ", ["投球分布 (Density)", "打率マップ (BA)", "OPSマップ", "球速ヒートマップ"], horizontal=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 5x5グリッド & ゾーン描画
    draw_zone(ax)
    
    # 打者画像の配置 (投手視点: 右打者は左、左打者は右)
    # フィルタリング結果にL/Rが混ざっている場合は両方出す、あるいは代表値を出す
    current_stand = 'R' # デフォルト
    if len(stand_sel) == 1: current_stand = stand_sel[0]
    draw_batter_placeholder(ax, current_stand)

    # データプロット (欠損値除去)
    df_plot = df_filt.dropna(subset=['plate_x', 'plate_z'])
    
    if plot_type == "投球分布 (Density)":
        if len(df_plot) > 5:
            sns.kdeplot(data=df_plot, x='plate_x', y='plate_z', fill=True, cmap='Reds', alpha=0.5, ax=ax, thresh=0.05)
        ax.scatter(df_plot['plate_x'], df_plot['plate_z'], s=20, color='black', alpha=0.3, edgecolors='white')
    
    elif plot_type in ["打率マップ (BA)", "OPSマップ", "球速ヒートマップ"]:
        # グリッド計算
        draw_heatmap(ax, df_plot, plot_type)

    # 投手視点設定
    ax.set_xlim(-2.5, 2.5) # 左がマイナス(右打者側)、右がプラス(左打者側)
    ax.set_ylim(0, 5.0)
    ax.set_aspect('equal')
    ax.set_xlabel("Pitcher's View (Left: RHB, Right: LHB)")
    
    st.pyplot(fig)
    
    # データテーブル
    with st.expander("データ一覧を表示"):
        st.dataframe(df_filt[['game_date', 'batter', 'pitcher', 'events', 'description', 'pitch_type', 'release_speed', 'plate_x', 'plate_z']])


def draw_zone(ax):
    """ストライクゾーンと5x5グリッドを描画"""
    # ゾーン定義 (ft)
    sz_left, sz_right = -0.708, 0.708
    sz_bot, sz_top = 1.5, 3.5
    
    # 3x3 の分割線
    w = (sz_right - sz_left) /
