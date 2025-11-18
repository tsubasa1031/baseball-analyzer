import streamlit as st
import pybaseball
import pandas as pd
from pybaseball import statcast_pitcher, statcast_batter, playerid_lookup, batting_stats, pitching_stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import datetime
import numpy as np

# ----------------------------------------------------------------------
# ページ設定
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="⚾ MLB Analyzer Pro",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# 1. データ取得・キャッシュ関数 (ロースター & Statcast)
# ----------------------------------------------------------------------

# チーム名と略称の対応辞書
MLB_TEAMS = {
    'AL East': {
        'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS', 'New York Yankees': 'NYY', 
        'Tampa Bay Rays': 'TB', 'Toronto Blue Jays': 'TOR'
    },
    'AL Central': {
        'Chicago White Sox': 'CWS', 'Cleveland Guardians': 'CLE', 'Detroit Tigers': 'DET', 
        'Kansas City Royals': 'KC', 'Minnesota Twins': 'MIN'
    },
    'AL West': {
        'Houston Astros': 'HOU', 'Los Angeles Angels': 'LAA', 'Oakland Athletics': 'OAK', 
        'Seattle Mariners': 'SEA', 'Texas Rangers': 'TEX'
    },
    'NL East': {
        'Atlanta Braves': 'ATL', 'Miami Marlins': 'MIA', 'New York Mets': 'NYM', 
        'Philadelphia Phillies': 'PHI', 'Washington Nationals': 'WSH'
    },
    'NL Central': {
        'Chicago Cubs': 'CHC', 'Cincinnati Reds': 'CIN', 'Milwaukee Brewers': 'MIL', 
        'Pittsburgh Pirates': 'PIT', 'St. Louis Cardinals': 'STL'
    },
    'NL West': {
        'Arizona Diamondbacks': 'AZ', 'Colorado Rockies': 'COL', 'Los Angeles Dodgers': 'LAD', 
        'San Diego Padres': 'SD', 'San Francisco Giants': 'SF'
    }
}

@st.cache_data(ttl=86400) # 1日キャッシュ
def load_active_rosters(year=2024):
    """指定年の打撃・投球成績を取得し、チームごとの選手リストを作成する"""
    try:
        # 打者データの取得
        batters = batting_stats(year, qual=1) # qual=1 で少なくとも1打席以上
        if not batters.empty:
            batters = batters[['Name', 'Team', 'IDfg', 'mlbID']].copy()
            batters['Role'] = 'Batter'
        
        # 投手データの取得
        pitchers = pitching_stats(year, qual=1)
        if not pitchers.empty:
            pitchers = pitchers[['Name', 'Team', 'IDfg', 'mlbID']].copy()
            pitchers['Role'] = 'Pitcher'
        
        # 結合
        if batters.empty and pitchers.empty:
            return pd.DataFrame()
            
        roster = pd.concat([batters, pitchers], ignore_index=True)
        
        # 名前の重複削除（大谷などは両方にいる可能性があるため）
        roster = roster.drop_duplicates(subset=['mlbID'], keep='first')
        
        return roster
    except Exception as e:
        # エラー時は空のDFを返す（画面上にエラーを出さない）
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_statcast_data(start_dt, end_dt, p_id, b_id, game_types):
    """Statcastデータの取得"""
    try:
        df = pd.DataFrame()
        # 1. 投手 vs 打者
        if p_id and b_id:
            p_data = statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=p_id)
            if not p_data.empty and 'batter' in p_data.columns:
                df = p_data[p_data['batter'] == b_id].copy()
        # 2. 投手のみ
        elif p_id:
            df = statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=p_id)
        # 3. 打者のみ
        elif b_id:
            df = statcast_batter(start_dt=start_dt, end_dt=end_dt, player_id=b_id)
        
        # 試合タイプ絞り込み
        if not df.empty and game_types:
            if 'game_type' in df.columns:
                # P (Postseason) の展開
                targets = []
                if 'P' in game_types:
                    targets.extend([t for t in game_types if t != 'P'])
                    targets.extend(['F', 'D', 'L', 'W'])
                else:
                    targets = game_types
                df = df[df['game_type'].isin(targets)]
        
        return df
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 2. データ加工・計算関数
# ----------------------------------------------------------------------
def process_statcast_data(df_input):
    if df_input.empty: return df_input
    
    # ★ここ重要: キャッシュされたデータを変更しないようにコピーを作成
    df = df_input.copy()
    
    # ソート
    if 'game_date' in df.columns:
        df = df.sort_values('game_date').reset_index(drop=True)

    # カウント計算 (簡易版)
    if 'balls' not in df.columns: df['balls'] = 0
    if 'strikes' not in df.columns: df['strikes'] = 0
    
    # 打撃結果フラグ
    if 'events' in df.columns:
        events = df['events'].fillna('nan').str.lower()
        df['is_hit'] = events.isin(['single', 'double', 'triple', 'home_run']).astype(int)
        df['is_at_bat'] = ((df['is_hit'] == 1) | events.isin(['field_out', 'strikeout', 'grounded_into_double_play', 'double_play', 'fielders_choice'])).astype(int)
        df['is_pa_event'] = (df['is_at_bat'] == 1) | events.isin(['walk', 'hit_by_pitch', 'sac_fly']).astype(int)
        
        # 塁打
        tb_map = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
        df['slugging_base'] = events.map(tb_map).fillna(0).astype(int)
    else:
        df['is_hit'] = 0; df['is_at_bat'] = 0; df['is_pa_event'] = 0; df['slugging_base'] = 0

    # Hard Hit & Barrel
    if 'launch_speed' in df.columns:
        df['is_hard_hit'] = (df['launch_speed'].fillna(0) >= 95.0).astype(int)
    else: df['is_hard_hit'] = 0
    
    if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
        ls = df['launch_speed'].fillna(0); la = df['launch_angle'].fillna(0)
        # Barrel簡易定義
        cond = (ls >= 98) & (la >= 26) & (la <= 30) # 簡易条件
        df['is_barrel'] = np.where(cond, 1, 0)
    else: df['is_barrel'] = 0

    # 走者状況
    if 'on_1b' in df.columns:
        df['is_empty'] = (df['on_1b'].isna()) & (df['on_2b'].isna()) & (df['on_3b'].isna())
        df['is_risp'] = (df['on_2b'].notna()) | (df['on_3b'].notna())
    else:
        df['is_empty'] = True; df['is_risp'] = False
    
    return df

def get_metrics_summary(df):
    if df.empty: return "No Data"
    pa = df['is_pa_event'].sum()
    ab = df['is_at_bat'].sum()
    h = df['is_hit'].sum()
    bb = df[df['events'].isin(['walk'])].shape[0] if 'events' in df.columns else 0
    
    ba = h / ab if ab > 0 else 0.0
    obp = (h + bb) / pa if pa > 0 else 0.0
    slg = df['slugging_base'].sum() / ab if ab > 0 else 0.0
    ops = obp + slg
    
    return f"PA: {pa} | BA: {ba:.3f} | OPS: {ops:.3f}"

# ----------------------------------------------------------------------
# 3. UI - サイドバー (選手選択 & フィルター)
# ----------------------------------------------------------------------
st.sidebar.title("⚾ MLB Analyzer Pro")

# --- A. 期間選択 (カレンダー) ---
st.sidebar.subheader("📅 期間 (Date Range)")
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    start_date = st.date_input("開始", datetime.date(2024, 3, 20))
with col_d2:
    end_date = st.date_input("終了", datetime.date(2024, 11, 2))

# --- B. 選手選択 (タブ切り替え) ---
st.sidebar.subheader("👤 選手選択 (Player Select)")
search_mode = st.sidebar.radio("検索方法", ["チームから探す (現役)", "名前検索 (引退/全選手)"], index=0)

selected_p_id = None
selected_p_name = ""
selected_b_id = None
selected_b_name = ""

# B-1. チームから探す (現役選手)
if search_mode == "チームから探す (現役)":
    # ロースター読み込み (2024年基準)
    roster_df = load_active_rosters(2024)
    
    if not roster_df.empty:
        # --- 投手選択 ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔽 投手 (Pitcher)**")
        p_league = st.sidebar.selectbox("リーグ (P)", list(MLB_TEAMS.keys()), key="pl")
        p_team_name = st.sidebar.selectbox("チーム (P)", list(MLB_TEAMS[p_league].keys()), key="pt")
        p_team_abbr = MLB_TEAMS[p_league][p_team_name]
        
        team_pitchers = roster_df[(roster_df['Team'] == p_team_abbr) & (roster_df['Role'] == 'Pitcher')].sort_values('Name')
        p_options = ["指定なし"] + team_pitchers['Name'].tolist()
        p_select = st.sidebar.selectbox("選手名 (Pitcher)", p_options)
        
        if p_select != "指定なし":
            player_row = team_pitchers[team_pitchers['Name'] == p_select].iloc[0]
            selected_p_id = int(player_row['mlbID'])
            selected_p_name = p_select

        # --- 打者選択 ---
        st.sidebar.markdown("**🔽 打者 (Batter)**")
        b_league = st.sidebar.selectbox("リーグ (B)", list(MLB_TEAMS.keys()), key="bl")
        b_team_name = st.sidebar.selectbox("チーム (B)", list(MLB_TEAMS[b_league].keys()), key="bt")
        b_team_abbr = MLB_TEAMS[b_league][b_team_name]
        
        team_batters = roster_df[(roster_df['Team'] == b_team_abbr)].sort_values('Name')
        b_options = ["指定なし"] + team_batters['Name'].tolist()
        b_select = st.sidebar.selectbox("選手名 (Batter)", b_options, index=0)
        
        if b_select != "指定なし":
            player_row = team_batters[team_batters['Name'] == b_select].iloc[0]
            selected_b_id = int(player_row['mlbID'])
            selected_b_name = b_select
    else:
        st.sidebar.error("選手リストの読み込みに失敗しました。")

# B-2. 名前検索 (引退選手含む)
else:
    st.sidebar.info("💡 英語の姓(Last Name)を入力して検索します。引退選手も検索可能です。")
    
    # 投手検索
    st.sidebar.markdown("**🔍 投手検索**")
    p_search_str = st.sidebar.text_input("投手 姓 (例: darvish)", "")
    if p_search_str:
        try:
            found_p = playerid_lookup(p_search_str)
            if not found_p.empty:
                found_p['label'] = found_p['name_first'] + " " + found_p['name_last'] + " (" + found_p['mlb_played_first'].astype(str) + "-" + found_p['mlb_played_last'].astype(str) + ")"
                p_choice = st.sidebar.selectbox("候補を選択 (P)", found_p['label'].tolist())
                p_row = found_p[found_p['label'] == p_choice].iloc[0]
                selected_p_id = int(p_row['key_mlbam'])
                selected_p_name = f"{p_row['name_first']} {p_row['name_last']}"
            else:
                st.sidebar.warning("見つかりませんでした")
        except: pass

    # 打者検索
    st.sidebar.markdown("**🔍 打者検索**")
    b_search_str = st.sidebar.text_input("打者 姓 (例: jeter)", "")
    if b_search_str:
        try:
            found_b = playerid_lookup(b_search_str)
            if not found_b.empty:
                found_b['label'] = found_b['name_first'] + " " + found_b['name_last'] + " (" + found_b['mlb_played_first'].astype(str) + "-" + found_b['mlb_played_last'].astype(str) + ")"
                b_choice = st.sidebar.selectbox("候補を選択 (B)", found_b['label'].tolist())
                b_row = found_b[found_b['label'] == b_choice].iloc[0]
                selected_b_id = int(b_row['key_mlbam'])
                selected_b_name = f"{b_row['name_first']} {b_row['name_last']}"
            else:
                st.sidebar.warning("見つかりませんでした")
        except: pass

# --- C. 詳細フィルター ---
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ 詳細フィルター (Filters)"):
    pitch_code = st.selectbox("球種", ['', 'FF', 'SL', 'CU', 'CH', 'FS', 'SI', 'FC', 'ST'], format_func=lambda x: "All" if x == "" else x)
    batter_stand = st.radio("打席", ["All", "R", "L"], horizontal=True)
    if batter_stand == "All": batter_stand = ""
    
    target_result = st.selectbox("結果", ['', 'strikeout', 'walk', 'single', 'home_run', 'hit_into_play'], format_func=lambda x: "All" if x == "" else x)

analysis_type = st.sidebar.selectbox("📊 分析タイプ", ['ops', 'woba', 'ba', 'density'], index=0)

# ----------------------------------------------------------------------
# 4. メイン処理
# ----------------------------------------------------------------------

# 実行ボタン
if st.sidebar.button("分析実行 (Analyze) 🚀", type="primary"):
    
    # 選手が少なくとも1人選ばれているか確認
    if not selected_p_id and not selected_b_id:
        st.error("投手または打者を選択してください。")
    else:
        title_str = ""
        if selected_p_name: title_str += f"Pitcher: {selected_p_name} "
        if selected_p_name and selected_b_name: title_str += "vs "
        if selected_b_name: title_str += f"Batter: {selected_b_name}"
        
        st.subheader(f"⚾ {title_str}")
        st.caption(f"Period: {start_date} ~ {end_date}")

        with st.spinner('データ取得中... (Statcast)'):
            df_raw = get_statcast_data(
                str(start_date), str(end_date), 
                selected_p_id, selected_b_id, 
                ['R', 'P']
            )
            
        if df_raw.empty:
            st.warning("データが見つかりませんでした。期間や条件を変更してください。")
        else:
            # データ加工 (ここでコピーされるので安全)
            df = process_statcast_data(df_raw)
            
            # フィルター適用
            df_filtered = df.copy()
            if pitch_code:
                col = 'pitch_type' if 'pitch_type' in df.columns else 'pitch_name'
                if col in df.columns: df_filtered = df_filtered[df_filtered[col] == pitch_code]
            if batter_stand:
                df_filtered = df_filtered[df_filtered['stand'] == batter_stand]
            if target_result:
                if target_result == 'hit_into_play':
                    df_filtered = df_filtered[df_filtered['description'] == 'hit_into_play']
                elif 'events' in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered['events'] == target_result]

            # 結果表示
            col_res1, col_res2 = st.columns([3, 1])
            
            with col_res1:
                # グラフ描画
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # ストライクゾーン
                plate_width = 17/12
                sz_top, sz_bottom = 3.5, 1.5
                ax.add_patch(patches.Rectangle((-plate_width/2, sz_bottom), plate_width, sz_top-sz_bottom, fill=False, edgecolor='blue', lw=2, ls='--'))
                ax.add_patch(patches.Polygon([(-plate_width/2, 0), (plate_width/2, 0), (plate_width/2, 0.2), (0, 0.4), (-plate_width/2, 0.2)], color='gray', alpha=0.3))
                
                # 打者シルエット (簡易)
                stand_draw = batter_stand if batter_stand else 'L'
                base_x = -2.5 if stand_draw == 'R' else 2.5
                ax.add_patch(patches.Ellipse((base_x, 3.0), 2.0, 6.0, color='gray', alpha=0.3))

                # プロット
                if 'plate_x' in df_filtered.columns and 'plate_z' in df_filtered.columns:
                    df_plot = df_filtered.dropna(subset=['plate_x', 'plate_z'])
                    
                    if analysis_type == 'density':
                        if not df_plot.empty:
                            sns.kdeplot(data=df_plot, x='plate_x', y='plate_z', fill=True, cmap='Reds', alpha=0.6, ax=ax, thresh=0.05)
                            ax.scatter(df_plot['plate_x'], df_plot['plate_z'], s=10, color='black', alpha=0.3)
                    else:
                        # OPSなどのグリッドマップ (簡易版: 散布図で代用しつつ色分け)
                        if not df_plot.empty:
                            colors = df_plot['is_hit'].apply(lambda x: 'red' if x == 1 else 'blue')
                            ax.scatter(df_plot['plate_x'], df_plot['plate_z'], c=colors, s=30, alpha=0.6, edgecolors='white')
                            ax.scatter([], [], c='red', label='Hit')
                            ax.scatter([], [], c='blue', label='Out/Other')
                            ax.legend(loc='upper right')
                else:
                    st.info("投球位置データがありません")

                ax.set_xlim(2.5, -2.5)
                ax.set_ylim(0, 5.0)
                ax.set_aspect('equal')
                ax.set_xlabel("Catcher's View")
                ax.set_title(f"{analysis_type.upper()} Analysis")
                
                st.pyplot(fig)

            with col_res2:
                st.markdown("### Stats")
                st.info(get_metrics_summary(df_filtered))
                
                st.markdown("### Data")
                cols_to_show = ['game_date', 'batter', 'pitcher', 'events', 'pitch_type', 'launch_speed']
                # 存在するカラムだけ表示
                valid_cols = [c for c in cols_to_show if c in df_filtered.columns]
                st.dataframe(df_filtered[valid_cols].head(10), height=300)