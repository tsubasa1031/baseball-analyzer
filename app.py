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
import traceback
import os
import time 

# ----------------------------------------------------------------------
# 0. ページ設定 (何よりも先に実行)
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
    # クラウド環境でのクラッシュを防ぐため、明示的なキャッシュ有効化は行わない
except ImportError as e:
    st.error(f"ライブラリの読み込みに失敗しました。requirements.txtを確認してください: {e}")
    st.stop()
except Exception as e:
    st.error(f"予期せぬエラーが発生しました: {e}")
    st.stop()

# ----------------------------------------------------------------------
# 2. 定数・設定
# ----------------------------------------------------------------------
GAME_TYPE_MAP = {
    'Regular Season': 'R',
    'Postseason': 'P',
    'Spring Training': 'S',
    'All-Star': 'A',
    'Exhibition': 'E'
}

# ----------------------------------------------------------------------
# 3. セッションステート初期化
# ----------------------------------------------------------------------
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = pd.DataFrame()

# 古いデータ形式(5要素)が残っていたら、強制的にリセットして7要素にする
if 'data_params' not in st.session_state or \
   st.session_state.data_params is None or \
   len(st.session_state.data_params) != 7:
    # (p_name, b_name, s_date, e_date, g_types, is_p_focus, is_b_focus)
    st.session_state.data_params = (None, None, None, None, None, False, False)

# ----------------------------------------------------------------------
# 4. データ取得・検索関数
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600)
def search_player_cached(name_str):
    """選手名検索をキャッシュして高速化・安定化"""
    if not name_str:
        return pd.DataFrame()
    try:
        # last_name検索
        return playerid_lookup(name_str.lower().strip())
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_statcast_data_safe(start_dt, end_dt, p_id, b_id, game_types_list):
    """Statcastデータの取得"""
    try:
        s_dt = pd.to_datetime(start_dt).strftime('%Y-%m-%d')
        e_dt = pd.to_datetime(end_dt).strftime('%Y-%m-%d')
        df = pd.DataFrame()

        if p_id and b_id:
            raw = statcast_pitcher(start_dt=s_dt, end_dt=e_dt, player_id=p_id)
            if not raw.empty and 'batter' in raw.columns: df = raw[raw['batter'] == b_id].copy()
        elif p_id:
            df = statcast_pitcher(start_dt=s_dt, end_dt=e_dt, player_id=p_id)
        elif b_id:
            df = statcast_batter(start_dt=s_dt, end_dt=e_dt, player_id=b_id)
        else:
            # リーグ全体
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
    except Exception:
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 5. データ加工 & 描画補助関数
# ----------------------------------------------------------------------
def process_statcast_data(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'game_date' in df.columns: df = df.sort_values('game_date').reset_index(drop=True)

    cols_to_init = ['balls', 'strikes', 'outs_when_up', 'launch_speed', 'launch_angle', 'woba_value', 'plate_x', 'plate_z', 'stand', 'p_throws', 'bb_type']
    for c in cols_to_init:
        if c not in df.columns: df[c] = 0 if c not in ['woba_value', 'stand', 'p_throws', 'bb_type'] else np.nan

    if 'events' in df.columns:
        events = df['events'].fillna('nan').str.lower()
        hits = ['single', 'double', 'triple', 'home_run']
        df['is_hit'] = events.isin(hits).astype(int)
        ab_events = hits + ['field_out', 'strikeout', 'grounded_into_double_play', 'double_play', 'fielders_choice', 'force_out']
        df['is_at_bat'] = events.isin(ab_events).astype(int)
        pa_events = ab_events + ['walk', 'hit_by_pitch', 'sac_fly']
        df['is_pa_event'] = events.isin(pa_events).astype(int)
        
        # K, BB, HR, SF カウント追加
        df['is_strikeout'] = events.isin(['strikeout', 'strikeout_double_play']).astype(int)
        df['is_walk'] = events.isin(['walk', 'intent_walk']).astype(int)
        df['is_hr'] = events.isin(['home_run']).astype(int)
        df['is_sac_fly'] = events.isin(['sac_fly']).astype(int)

        tb_map = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
        df['slugging_base'] = events.map(tb_map).fillna(0).astype(int)
        df['is_obp_denom'] = (df['is_at_bat'] | events.isin(['walk', 'hit_by_pitch', 'sac_fly'])).astype(int)
        df['is_on_base'] = (df['is_hit'] | events.isin(['walk', 'hit_by_pitch'])).astype(int)
        
        # BIPの分母となる打球 (HBP, K, BB, SF を除いた打席)
        df['is_bip_ab'] = df['is_at_bat'] - df['is_strikeout'] 
    else:
        df['is_hit'] = 0; df['is_at_bat'] = 0; df['is_pa_event'] = 0; df['slugging_base'] = 0; df['is_strikeout'] = 0; df['is_walk'] = 0; df['is_hr'] = 0
        df['is_bip_ab'] = 0


    df['is_hard_hit'] = (df['launch_speed'].fillna(0) >= 95.0).astype(int)
    ls = df['launch_speed'].fillna(0); la = df['launch_angle'].fillna(0)
    cond = (ls >= 98) & (la >= 26) & (la <= 30)
    df['is_barrel'] = np.where(cond, 1, 0)
    
    df['on_1b_bool'] = df['on_1b'].notna(); df['on_2b_bool'] = df['on_2b'].notna(); df['on_3b_bool'] = df['on_3b'].notna()
    df['is_empty'] = (~df['on_1b_bool']) & (~df['on_2b_bool']) & (~df['on_3b_bool'])
    df['is_risp'] = (df['on_2b_bool']) | (df['on_3b_bool'])
    df['is_on_base_no_risp'] = (df['on_1b_bool']) & (~df['on_2b_bool']) & (~df['on_3b_bool'])

    return df

def get_metrics_summary(df, is_batter_focus, is_pitcher_focus):
    if df.empty: return "#### データがありません"
    pa = df['is_pa_event'].sum()
    ab = df['is_at_bat'].sum()
    h = df['is_hit'].sum()
    
    k_count = df['is_strikeout'].sum()
    bb_count = df['is_walk'].sum()
    hr_count = df['is_hr'].sum()
    
    # TBF (対戦打者数) は PA (打席) を使用
    denom_pa = pa if pa > 0 else 1 
    bip_denom = ab - k_count - hr_count # 打数からKとHRを引いたもの

    k_percent = (k_count / denom_pa) * 100
    bb_percent = (bb_count / denom_pa) * 100
    k_to_bb = k_count / bb_count if bb_count > 0 else k_count if k_count > 0 else 0
    
    # BA, SLG, OPS
    ba = h / ab if ab > 0 else 0.0
    slg = df['slugging_base'].sum() / ab if ab > 0 else 0.0
    obp = df['is_on_base'].sum() / df['is_obp_denom'].sum() if df['is_obp_denom'].sum() > 0 else 0.0
    ops = obp + slg
    
    # セイバーメトリクス指標
    iso = slg - ba
    babip = (h - hr_count) / bip_denom if bip_denom > 0 else 0.0
    woba_avg = df['woba_value'].mean() if 'woba_value' in df.columns and pa > 0 else np.nan

    hard_hit_rate = df['is_hard_hit'].mean()
    
    # ----------------------------------------------------
    # サマリー表示構築
    # ----------------------------------------------------
    
    summary_parts = []
    
    # 1. 基本/伝統指標
    ba_label = "BA" if is_batter_focus and not is_pitcher_focus else "BA Against" if is_pitcher_focus and not is_batter_focus else "BA/BA Against"
    summary_parts.append(f"PA: {pa} | {ba_label}: {ba:.3f} | OPS: {ops:.3f}")
    
    # 2. セイバーメトリクス指標
    woba_str = f"wOBA: {woba_avg:.3f} (Statcast)" if not np.isnan(woba_avg) else "wOBA: N/A"
    summary_parts.append(f"{woba_str} | ISO: {iso:.3f} | BABIP: {babip:.3f}")
    
    # 3. K/BB指標
    kbb_summary = f"K%: {k_percent:.1f}% | BB%: {bb_percent:.1f}%"
    if is_pitcher_focus and not is_batter_focus:
        kbb_summary += f" | K/BB: {k_to_bb:.2f}"
    summary_parts.append(kbb_summary)
    
    # 4. ハードヒット指標
    summary_parts.append(f"HardHit%: {hard_hit_rate:.1%} | Barrel%: {df['is_barrel'].mean():.1%}")

    # 5. 割合情報 (投打別)
    if is_pitcher_focus and not is_batter_focus and 'bb_type' in df.columns:
        batted_balls = df[df['bb_type'].notna()].copy()
        if not batted_balls.empty and batted_balls['bb_type'].value_counts().sum() > 0:
            bb_ratios = (batted_balls['bb_type'].value_counts(normalize=True) * 100).round(1).apply(lambda x: f"{x}%")
            bb_summary = "打球タイプ: "
            bb_summary += f"GB: {bb_ratios.get('ground_ball', '0.0%')} / FB: {bb_ratios.get('fly_ball', '0.0%')} / LD: {bb_ratios.get('line_drive', '0.0%')} / PU: {bb_ratios.get('popup', '0.0%')}"
            summary_parts.append(bb_summary)

    elif is_batter_focus and not is_pitcher_focus and 'pitch_type' in df.columns:
        pitch_mix = df[df['pitch_type'].notna()].copy()
        if not pitch_mix.empty and pitch_mix['pitch_type'].value_counts().sum() > 0:
            pt_ratios = (pitch_mix['pitch_type'].value_counts(normalize=True) * 100).round(1).sort_values(ascending=False)
            top_3 = pt_ratios.head(3)
            pt_summary = "球種割合 (Top 3): "
            pt_summary += " / ".join([f"{name}: {ratio:.1f}%" for name, ratio in top_3.items()])
            summary_parts.append(pt_summary)

    final_title = "投球分析" if is_pitcher_focus and not is_batter_focus else "打撃分析" if is_batter_focus and not is_pitcher_focus else "集計分析"
    
    return f"#### {final_title}\n" + "\n\n".join(summary_parts)

# --- 描画 ---
def draw_5x5_grid(ax):
    sz_left, sz_right = -0.708, 0.708
    sz_bot, sz_top = 1.5, 3.5
    w = (sz_right - sz_left) / 3; h = (sz_top - sz_bot) / 3
    x_lines = [sz_left - w, sz_left, sz_left + w, sz_right - w, sz_right, sz_right + w]
    z_lines = [sz_bot - h, sz_bot, sz_bot + h, sz_top - h, sz_top, sz_top + h]
    
    line_props = {'color': 'black', 'linestyle': '-', 'alpha': 0.3, 'linewidth': 1}
    zone_props = {'color': 'blue', 'linestyle': '-', 'alpha': 0.8, 'linewidth': 2} 

    for i, x in enumerate(x_lines):
        props = zone_props if i in [1, 4] else line_props
        ax.plot([x, x], [z_lines[0], z_lines[5]], **props)

    for i, z in enumerate(z_lines):
        props = zone_props if i in [1, 4] else line_props
        ax.plot([x_lines[0], x_lines[5]], [z, z], **props)

    return x_lines, z_lines

def draw_batter(ax, stand):
    # 投手視点: 右打者(R)は左側(-)、左打者(L)は右側(+)
    if stand == 'R':
        base_x = -2.5
        extent = [-4.0, -1.0, 0, 6.0]
    else:
        base_x = 2.5
        extent = [1.0, 4.0, 0, 6.0]

    loaded = False
    img_file = 'batterR.png' if stand == 'R' else 'batterL.png'
    if os.path.exists(img_file):
        try:
            img = mpimg.imread(img_file)
            ax.imshow(img, extent=extent, aspect='auto', zorder=0)
            loaded = True
        except: pass
    
    if not loaded:
        ax.add_patch(patches.Ellipse((base_x, 3.0), 1.5, 5.5, color='gray', alpha=0.5, zorder=0))
        ax.add_patch(patches.Circle((base_x, 5.5), 0.4, color='gray', alpha=0.5, zorder=0))

# ----------------------------------------------------------------------
# 6. メインアプリケーション
# ----------------------------------------------------------------------
def main():
    st.sidebar.title("⚾ MLB Analyzer Pro")

    # A. 期間
    st.sidebar.markdown("### STEP 1: データ取得")
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1: start_date = st.date_input("開始", datetime.date(2025, 3, 27))
    with col_d2: end_date = st.date_input("終了", datetime.date(2025, 11, 2))

    # A2. 試合タイプ
    selected_game_types_label = st.sidebar.multiselect(
        "対象試合", options=list(GAME_TYPE_MAP.keys()), default=['Regular Season', 'Postseason']
    )
    selected_game_types_code = [GAME_TYPE_MAP[l] for l in selected_game_types_label]

    # B. 選手選択 (名前検索)
    st.sidebar.subheader("👤 選手選択 (名前検索)")
    st.sidebar.caption("姓(Last Name)を入力し、Enterまたはカーソルを外して確定してください。")
    
    selected_p_id, selected_p_name = None, ""
    selected_b_id, selected_b_name = None, ""

    # --- 投手検索 ---
    p_search = st.sidebar.text_input("投手 姓 (例: darvish)", key="p_search_input")
    p_choice_label = "指定なし"
    if p_search:
        p_found = search_player_cached(p_search)
        if not p_found.empty:
            
            p_found['label'] = p_found['name_first'] + " " + p_found['name_last'] + " (" + p_found['mlb_played_first'].astype(str) + "-" + p_found['mlb_played_last'].astype(str) + ")"
            p_options = ["指定なし"] + p_found['label'].tolist()
            default_index = 0
            
            if len(p_found) == 1:
                default_index = 1
                st.sidebar.info(f"投手: **{p_found.iloc[0]['label']}** を自動選択しました。")
            
            p_choice_label = st.sidebar.selectbox("候補 (P)", p_options, index=default_index, key="p_select_box")
            
            if p_choice_label != "指定なし":
                row = p_found[p_found['label'] == p_choice_label].iloc[0]
                selected_p_id, selected_p_name = int(row['key_mlbam']), f"{row['name_first']} {row['name_last']}"
        else:
            st.sidebar.warning("投手がヒットしません")

    # --- 打者検索 ---
    b_search = st.sidebar.text_input("打者 姓 (例: ohtani)", key="b_search_input")
    b_choice_label = "指定なし"
    if b_search:
        b_found = search_player_cached(b_search)
        if not b_found.empty:
            
            b_found['label'] = b_found['name_first'] + " " + b_found['name_last'] + " (" + b_found['mlb_played_first'].astype(str) + "-" + b_found['mlb_played_last'].astype(str) + ")"
            b_options = ["指定なし"] + b_found['label'].tolist()
            default_index = 0
            
            if len(b_found) == 1:
                default_index = 1
                st.sidebar.info(f"打者: **{b_found.iloc[0]['label']}** を自動選択しました。")

            b_choice_label = st.sidebar.selectbox("候補 (B)", b_options, index=default_index, key="b_select_box")

            if b_choice_label != "指定なし":
                row = b_found[b_found['label'] == b_choice_label].iloc[0]
                selected_b_id, selected_b_name = int(row['key_mlbam']), f"{row['name_first']} {row['name_last']}"
        else:
            st.sidebar.warning("打者がヒットしません")

    # データ取得ボタン
    if st.sidebar.button("データ取得 (Get Data) 📥", type="primary", key="btn_get_data"):
        
        if not selected_p_id and not selected_b_id:
            st.warning("選手が指定されていません。リーグ全体のデータを取得します。")
            
        if not selected_p_id and not selected_b_id and (end_date - start_date).days > 14:
             st.warning("期間が長すぎるため、リーグ全体の取得はタイムアウトする可能性があります。")

        with st.spinner('データ取得中...'):
            df_raw = get_statcast_data_safe(str(start_date), str(end_date), selected_p_id, selected_b_id, selected_game_types_code)
            
            if df_raw.empty:
                st.session_state.raw_data = pd.DataFrame()
                st.session_state.data_params = (None, None, None, None, None, False, False)
                st.error("データが見つかりませんでした。条件を変更してください。")
            else:
                is_p_focus = selected_p_id is not None
                is_b_focus = selected_b_id is not None
                st.session_state.raw_data = df_raw
                st.session_state.data_params = (selected_p_name, selected_b_name, str(start_date), str(end_date), ", ".join(selected_game_types_label), is_p_focus, is_b_focus)
                st.success(f"データ取得完了: {len(df_raw)} 球")

    # ==========================================
    # STEP 2: フィルター & 分析
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("### STEP 2: フィルター & 分析")

    if st.session_state.raw_data.empty:
        st.info("データがありません。STEP 1を実行してください。")
    else:
        # 要素展開
        p_name, b_name, s_date, e_date, g_types, is_p_focus, is_b_focus = st.session_state.data_params
        
        title_str = "League Wide"
        if p_name: title_str = f"P: {p_name}"
        if b_name: title_str += f" vs B: {b_name}"
        st.subheader(f"⚾ {title_str}")
        st.caption(f"{s_date} ~ {e_date} | {g_types} | {len(st.session_state.raw_data)} Pitches")

        with st.sidebar.expander("⚙️ 詳細フィルター", expanded=True):
            pitch_code = st.selectbox("球種", ['', 'FF', 'SL', 'CU', 'CH', 'FS', 'SI', 'FC', 'ST'], format_func=lambda x: "All" if x == "" else x)
            batter_stand = st.radio("打席", ["All", "R", "L"], horizontal=True, index=0)
            c1, c2 = st.columns(2)
            with c1:
                target_balls = st.selectbox("ボール", ['', '0', '1', '2', '3'])
                target_outs = st.selectbox("アウト", ['', '0', '1', '2'])
            with c2:
                target_strikes = st.selectbox("ストライク", ['', '0', '1', '2'])
                target_runners = st.selectbox("走者", ['', 'Empty', 'RISP', 'On Base (Not RISP)'])

            target_bb_type = st.selectbox("打球タイプ", ['', 'ground_ball', 'fly_ball', 'line_drive', 'popup'])
            target_result = st.selectbox("結果", ['', 'strikeout', 'walk', 'single', 'double', 'triple', 'home_run', 'hit_into_play', 'woba_zero'])

        ANALYSIS_OPTIONS = {
            'Density (投球分布)': 'density', 'OPS Map': 'ops', 'Batting Avg / Against': 'ba',
            'wOBA': 'woba', 'Hard Hit%': 'hard_hit', 'Barrel%': 'barrel'
        }
        analysis_label = st.sidebar.selectbox("📊 分析タイプ", list(ANALYSIS_OPTIONS.keys()))
        analysis_type = ANALYSIS_OPTIONS[analysis_label]

        if st.sidebar.button("グラフ描画 📊", type="secondary"):
            df = process_statcast_data(st.session_state.raw_data)
            df_filtered = df.copy()
            
            if pitch_code:
                col = 'pitch_type' if 'pitch_type' in df.columns else 'pitch_name'
                if col in df.columns: df_filtered = df_filtered[df_filtered[col] == pitch_code]
            if batter_stand != "All": df_filtered = df_filtered[df_filtered['stand'] == batter_stand]
            if target_balls != '': df_filtered = df_filtered[df_filtered['balls'] == int(target_balls)]
            if target_strikes != '': df_filtered = df_filtered[df_filtered['strikes'] == int(target_strikes)]
            if target_outs != '': df_filtered = df_filtered[df_filtered['outs_when_up'] == int(target_outs)]
            if target_runners == 'Empty': df_filtered = df_filtered[df_filtered['is_empty']]
            elif target_runners == 'RISP': df_filtered = df_filtered[df_filtered['is_risp']]
            elif target_runners == 'On Base (Not RISP)': df_filtered = df_filtered[df_filtered['is_on_base_no_risp']]
            if target_bb_type: df_filtered = df_filtered[df_filtered['bb_type'] == target_bb_type]
            if target_result:
                if target_result == 'hit_into_play': df_filtered = df_filtered[df_filtered['description'] == 'hit_into_play']
                elif target_result == 'woba_zero': df_filtered = df_filtered[df_filtered['woba_value'] == 0]
                else: df_filtered = df_filtered[df_filtered['events'] == target_result]

            col_res1, col_res2 = st.columns([3, 1])
            with col_res1:
                fig, ax = plt.subplots(figsize=(8, 8))
                x_grid, z_grid = draw_5x5_grid(ax)
                stand_draw = batter_stand if batter_stand != "All" else 'R' 
                draw_batter(ax, stand_draw)

                df_plot = df_filtered.dropna(subset=['plate_x', 'plate_z'])
                
                if df_plot.empty:
                    st.info("条件に該当するデータがありません")
                elif analysis_type == 'density':
                    try: sns.kdeplot(data=df_plot, x='plate_x', y='plate_z', fill=True, cmap='Reds', alpha=0.6, ax=ax, thresh=0.05)
                    except: pass 
                    ax.scatter(df_plot['plate_x'], df_plot['plate_z'], s=15, color='black', alpha=0.2)
                    ax.set_title(f"Pitch Density (n={len(df_plot)})")
                else:
                    if analysis_type == 'ops': metric = 'OPS'; vmin, vmax = 0.4, 1.2; cmap = 'coolwarm'
                    elif analysis_type == 'ba': metric = 'AVG'; vmin, vmax = 0.1, 0.4; cmap = 'coolwarm'
                    elif analysis_type == 'woba': metric = 'wOBA'; vmin, vmax = 0.2, 0.5; cmap = 'coolwarm'
                    elif analysis_type == 'hard_hit': metric = 'HardHit%'; vmin, vmax = 0.2, 0.6; cmap = 'Reds'
                    elif analysis_type == 'barrel': metric = 'Barrel%'; vmin, vmax = 0.0, 0.2; cmap = 'Reds'
                    
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                    m = cm.ScalarMappable(norm=norm, cmap=cmap)

                    for i in range(5): 
                        for j in range(5):
                            x1, x2 = x_grid[j], x_grid[j+1]
                            z1, z2 = z_grid[i], z_grid[i+1]
                            sub = df_plot[(df_plot['plate_x'] >= x1) & (df_plot['plate_x'] < x2) & (df_plot['plate_z'] >= z1) & (df_plot['plate_z'] < z2)]
                            if len(sub) > 0:
                                val = np.nan
                                if analysis_type == 'ops':
                                    denom = sub['is_at_bat'].sum()
                                    if denom > 0: val = (sub['is_on_base'].sum()/sub['is_obp_denom'].sum()) + (sub['slugging_base'].sum()/denom)
                                elif analysis_type == 'ba':
                                    denom = sub['is_at_bat'].sum()
                                    if denom > 0: val = sub['is_hit'].sum() / denom
                                elif analysis_type == 'woba': val = sub['woba_value'].mean()
                                elif analysis_type == 'hard_hit': val = sub['is_hard_hit'].mean()
                                elif analysis_type == 'barrel': val = sub['is_barrel'].mean()
                                
                                if not np.isnan(val):
                                    ax.add_patch(patches.Rectangle((x1, z1), x2-x1, z2-z1, color=m.to_rgba(val), alpha=0.8))
                                    col = 'white' if norm(val) > 0.6 or norm(val) < 0.4 else 'black'
                                    fmt = ".3f" if metric in ['OPS', 'AVG', 'wOBA'] else ".0%"
                                    ax.text((x1+x2)/2, (z1+z2)/2, f"{val:{fmt}}\n({len(sub)})", ha='center', va='center', fontsize=7, color=col)
                    plt.colorbar(m, ax=ax, label=metric)

                ax.set_xlim(-2.5, 2.5) # 投手視点
                ax.set_ylim(0, 6.0)
                ax.set_aspect('equal')
                ax.set_xlabel("Pitcher's View")
                st.pyplot(fig)

            with col_res2:
                st.markdown("### Summary")
                st.info(get_metrics_summary(df_filtered, is_b_focus, is_p_focus))
                st.dataframe(df_filtered[['game_date', 'events', 'description', 'pitch_type', 'launch_speed']].head(20))

if __name__ == "__main__":
    try: main()
    except Exception as e:
        st.error("エラーが発生しました")
        st.code(traceback.format_exc())
