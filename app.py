import streamlit as st
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import traceback

# ライブラリのインポート自体で失敗する場合に備える
try:
    import pybaseball
    from pybaseball import statcast_pitcher, statcast_batter, playerid_lookup, batting_stats, pitching_stats, statcast
except ImportError as e:
    st.error(f"ライブラリの読み込みに失敗しました: {e}")
    st.stop()

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
# 定数・設定
# ----------------------------------------------------------------------
GAME_TYPE_MAP = {
    'Regular Season': 'R',
    'Postseason': 'P',
    'Spring Training': 'S',
    'All-Star': 'A',
    'Exhibition': 'E'
}

MLB_TEAMS_DICT = {
    'AL East': ['BAL', 'BOS', 'NYY', 'TB', 'TOR'],
    'AL Central': ['CWS', 'CLE', 'DET', 'KC', 'MIN'],
    'AL West': ['HOU', 'LAA', 'OAK', 'SEA', 'TEX'],
    'NL East': ['ATL', 'MIA', 'NYM', 'PHI', 'WSH'],
    'NL Central': ['CHC', 'CIN', 'MIL', 'PIT', 'STL'],
    'NL West': ['AZ', 'COL', 'LAD', 'SD', 'SF']
}

# ----------------------------------------------------------------------
# 1. データ取得・キャッシュ関数
# ----------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_active_rosters_safe(year):
    """
    ロースター取得関数 (エラーハンドリング強化版)
    batting_statsなどは外部サイトへのアクセス失敗で落ちやすいため防御する
    """
    def fetch_year(y):
        try:
            # qual=1 で軽量化
            b = batting_stats(y, qual=1)
            p = pitching_stats(y, qual=1)
            
            df_b = pd.DataFrame()
            if not b.empty:
                df_b = b[['Name', 'Team', 'IDfg', 'mlbID']].copy()
                df_b['Role'] = 'Batter'
            
            df_p = pd.DataFrame()
            if not p.empty:
                df_p = p[['Name', 'Team', 'IDfg', 'mlbID']].copy()
                df_p['Role'] = 'Pitcher'
            
            # どちらも空なら失敗とみなす
            if df_b.empty and df_p.empty:
                return pd.DataFrame()
                
            return pd.concat([df_b, df_p], ignore_index=True)
        except Exception:
            return pd.DataFrame()

    # まず指定年
    roster = fetch_year(year)
    
    # 失敗したら前年
    if roster.empty:
        roster = fetch_year(year - 1)
    
    # それでも失敗したら空を返す (アプリを落とさない)
    if roster.empty:
        return pd.DataFrame()
    
    # 重複削除とソート
    try:
        roster = roster.drop_duplicates(subset=['mlbID'], keep='first')
        roster = roster.sort_values('Name')
    except:
        pass
        
    return roster

@st.cache_data(ttl=3600)
def get_statcast_data_safe(start_dt, end_dt, p_id, b_id, game_types_list):
    """Statcastデータの取得 (エラーハンドリング強化版)"""
    try:
        # 日付変換
        s_dt = pd.to_datetime(start_dt).strftime('%Y-%m-%d')
        e_dt = pd.to_datetime(end_dt).strftime('%Y-%m-%d')
        
        df = pd.DataFrame()

        # 1. 投手 vs 打者
        if p_id and b_id:
            raw = statcast_pitcher(start_dt=s_dt, end_dt=e_dt, player_id=p_id)
            if not raw.empty and 'batter' in raw.columns:
                df = raw[raw['batter'] == b_id].copy()
        # 2. 投手のみ
        elif p_id:
            df = statcast_pitcher(start_dt=s_dt, end_dt=e_dt, player_id=p_id)
        # 3. 打者のみ
        elif b_id:
            df = statcast_batter(start_dt=s_dt, end_dt=e_dt, player_id=b_id)
        # 4. 両方なし（リーグ全体）
        else:
            # データ量過多のエラーを防ぐため try-except
            try:
                df = statcast(start_dt=s_dt, end_dt=e_dt)
            except Exception as e:
                st.error(f"リーグ全体のデータ取得中にエラーが発生しました。期間を短くしてください。\n詳細: {e}")
                return pd.DataFrame()
        
        # 試合タイプ絞り込み
        if not df.empty and game_types_list:
            if 'game_type' in df.columns:
                targets = []
                if 'P' in game_types_list:
                    targets.extend(['F', 'D', 'L', 'W'])
                targets.extend(game_types_list)
                targets = list(set(targets))
                df = df[df['game_type'].isin(targets)]
        
        return df
    except Exception as e:
        # ここでキャッチしてNoneやエラーメッセージを出せるようにする
        print(f"Statcast fetch error: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 2. データ加工
# ----------------------------------------------------------------------
def process_statcast_data(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    
    if 'game_date' in df.columns:
        df = df.sort_values('game_date').reset_index(drop=True)

    cols_to_init = ['balls', 'strikes', 'outs_when_up', 'launch_speed', 'launch_angle', 'woba_value', 'plate_x', 'plate_z']
    for c in cols_to_init:
        if c not in df.columns: df[c] = 0 if c != 'woba_value' else np.nan

    if 'events' in df.columns:
        events = df['events'].fillna('nan').str.lower()
        hits = ['single', 'double', 'triple', 'home_run']
        df['is_hit'] = events.isin(hits).astype(int)
        ab_events = hits + ['field_out', 'strikeout', 'grounded_into_double_play', 'double_play', 'fielders_choice', 'force_out']
        df['is_at_bat'] = events.isin(ab_events).astype(int)
        pa_events = ab_events + ['walk', 'hit_by_pitch', 'sac_fly']
        df['is_pa_event'] = events.isin(pa_events).astype(int)
        tb_map = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
        df['slugging_base'] = events.map(tb_map).fillna(0).astype(int)
        df['is_obp_denom'] = (df['is_at_bat'] | events.isin(['walk', 'hit_by_pitch', 'sac_fly'])).astype(int)
        df['is_on_base'] = (df['is_hit'] | events.isin(['walk', 'hit_by_pitch'])).astype(int)
        df['is_batted_ball'] = df['type'] == 'X'
    else:
        df['is_hit'] = 0; df['is_at_bat'] = 0; df['is_pa_event'] = 0; df['slugging_base'] = 0; df['is_batted_ball'] = 0

    df['is_hard_hit'] = (df['launch_speed'].fillna(0) >= 95.0).astype(int)
    ls = df['launch_speed'].fillna(0); la = df['launch_angle'].fillna(0)
    cond = (ls >= 98) & (la >= 26) & (la <= 30)
    df['is_barrel'] = np.where(cond, 1, 0)

    df['on_1b_bool'] = df['on_1b'].notna()
    df['on_2b_bool'] = df['on_2b'].notna()
    df['on_3b_bool'] = df['on_3b'].notna()
    df['is_empty'] = (~df['on_1b_bool']) & (~df['on_2b_bool']) & (~df['on_3b_bool'])
    df['is_risp'] = (df['on_2b_bool']) | (df['on_3b_bool'])
    df['is_on_base_no_risp'] = (df['on_1b_bool']) & (~df['on_2b_bool']) & (~df['on_3b_bool'])

    return df

def get_metrics_summary(df):
    if df.empty: return "No Data"
    pa = df['is_pa_event'].sum()
    ba = df['is_hit'].sum() / df['is_at_bat'].sum() if df['is_at_bat'].sum() > 0 else 0.0
    obp = df['is_on_base'].sum() / df['is_obp_denom'].sum() if df['is_obp_denom'].sum() > 0 else 0.0
    slg = df['slugging_base'].sum() / df['is_at_bat'].sum() if df['is_at_bat'].sum() > 0 else 0.0
    ops = obp + slg
    return f"PA: {pa} | BA: {ba:.3f} | OPS: {ops:.3f} | HardHit%: {df['is_hard_hit'].mean():.1%}"

# ----------------------------------------------------------------------
# 3. アプリケーションのメインロジック (try-exceptでラップ)
# ----------------------------------------------------------------------
def main():
    st.sidebar.title("⚾ MLB Analyzer Pro")

    # --- A. 期間 ---
    st.sidebar.subheader("📅 期間 (Date Range)")
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1: start_date = st.date_input("開始", datetime.date(2025, 3, 27))
    with col_d2: end_date = st.date_input("終了", datetime.date(2025, 11, 2))

    # --- A2. 試合タイプ ---
    st.sidebar.subheader("🏟️ 試合タイプ")
    selected_game_types_label = st.sidebar.multiselect(
        "対象試合 (複数選択可)",
        options=list(GAME_TYPE_MAP.keys()),
        default=['Regular Season', 'Postseason']
    )
    selected_game_types_code = [GAME_TYPE_MAP[l] for l in selected_game_types_label]

    # --- B. 選手選択 ---
    st.sidebar.subheader("👤 選手選択")
    st.sidebar.caption("※両方空欄なら「リーグ全体」のデータを分析します")
    search_mode = st.sidebar.radio("検索方法", ["チームから探す (現役)", "名前検索 (引退/全選手)"])

    selected_p_id, selected_p_name = None, ""
    selected_b_id, selected_b_name = None, ""

    # ==========================================
    # B-1. チームから探す (ロースター読み込み)
    # ==========================================
    if search_mode == "チームから探す (現役)":
        # ロースター読み込み (失敗しても空DFが返る)
        roster_df = load_active_rosters_safe(2025)
        
        if not roster_df.empty:
            # 存在するチーム一覧
            available_teams = sorted([t for t in roster_df['Team'].unique() if pd.notna(t)])
            
            # 投手
            st.sidebar.markdown("**🔽 投手 (Pitcher)**")
            p_team = st.sidebar.selectbox("チーム (P)", ["指定なし"] + available_teams, key="p_team_select")
            if p_team != "指定なし":
                team_pitchers = roster_df[(roster_df['Team'] == p_team) & (roster_df['Role'] == 'Pitcher')]
                p_select = st.sidebar.selectbox("選手名 (P)", ["指定なし"] + team_pitchers['Name'].tolist())
                if p_select != "指定なし":
                    row = team_pitchers[team_pitchers['Name'] == p_select].iloc[0]
                    selected_p_id, selected_p_name = int(row['mlbID']), p_select
            
            # 打者
            st.sidebar.markdown("**🔽 打者 (Batter)**")
            b_team = st.sidebar.selectbox("チーム (B)", ["指定なし"] + available_teams, key="b_team_select")
            if b_team != "指定なし":
                team_batters = roster_df[(roster_df['Team'] == b_team)]
                b_select = st.sidebar.selectbox("選手名 (B)", ["指定なし"] + team_batters['Name'].tolist())
                if b_select != "指定なし":
                    row = team_batters[team_batters['Name'] == b_select].iloc[0]
                    selected_b_id, selected_b_name = int(row['mlbID']), b_select
        else:
            st.sidebar.warning("⚠️ 選手リストの取得に失敗しました。データ提供元の接続不良などが考えられます。\n「名前検索」タブに切り替えて直接検索してください。")

    # ==========================================
    # B-2. 名前検索
    # ==========================================
    else:
        st.sidebar.info("姓(Last Name)を英語入力 (例: judge)")
        p_search = st.sidebar.text_input("投手 姓 (P)")
        if p_search:
            try:
                found = playerid_lookup(p_search)
                if not found.empty:
                    found['label'] = found['name_first'] + " " + found['name_last'] + " (" + found['mlb_played_first'].astype(str) + "-" + found['mlb_played_last'].astype(str) + ")"
                    p_choice = st.sidebar.selectbox("候補 (P)", ["指定なし"] + found['label'].tolist())
                    if p_choice != "指定なし":
                        row = found[found['label'] == p_choice].iloc[0]
                        selected_p_id, selected_p_name = int(row['key_mlbam']), f"{row['name_first']} {row['name_last']}"
            except: pass
        
        b_search = st.sidebar.text_input("打者 姓 (B)")
        if b_search:
            try:
                found = playerid_lookup(b_search)
                if not found.empty:
                    found['label'] = found['name_first'] + " " + found['name_last'] + " (" + found['mlb_played_first'].astype(str) + "-" + found['mlb_played_last'].astype(str) + ")"
                    b_choice = st.sidebar.selectbox("候補 (B)", ["指定なし"] + found['label'].tolist())
                    if b_choice != "指定なし":
                        row = found[found['label'] == b_choice].iloc[0]
                        selected_b_id, selected_b_name = int(row['key_mlbam']), f"{row['name_first']} {row['name_last']}"
            except: pass

    # --- C. 詳細フィルター ---
    st.sidebar.markdown("---")
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

    # --- D. 分析タイプ ---
    st.sidebar.markdown("---")
    ANALYSIS_OPTIONS = {
        'Density (投球分布)': 'density',
        'OPS Map (OPS)': 'ops',
        'Batting Avg Map (打率)': 'ba',
        'wOBA Map (wOBA)': 'woba',
        'Hard Hit% Map (強打率)': 'hard_hit',
        'Barrel% Map (バレル率)': 'barrel'
    }
    analysis_label = st.sidebar.selectbox("📊 分析タイプ", list(ANALYSIS_OPTIONS.keys()))
    analysis_type = ANALYSIS_OPTIONS[analysis_label]

    # ==========================================
    # メイン処理実行ボタン
    # ==========================================
    if st.sidebar.button("分析実行 (Analyze) 🚀", type="primary"):
        
        title_str = "League Wide Analysis"
        if selected_p_name and selected_b_name: title_str = f"Pitcher: {selected_p_name} vs Batter: {selected_b_name}"
        elif selected_p_name: title_str = f"Pitcher: {selected_p_name}"
        elif selected_b_name: title_str = f"Batter: {selected_b_name}"
        
        st.subheader(f"⚾ {title_str}")
        st.caption(f"Period: {start_date} ~ {end_date} | Game Types: {', '.join(selected_game_types_label)}")

        with st.spinner('データ取得・処理中...'):
            df_raw = get_statcast_data_safe(
                str(start_date), str(end_date), 
                selected_p_id, selected_b_id, 
                selected_game_types_code
            )
            
        if df_raw.empty:
            st.warning("データが見つかりませんでした。条件を変更してください。")
        else:
            df = process_statcast_data(df_raw)
            df_filtered = df.copy()
            
            # --- フィルター適用 ---
            if pitch_code:
                col = 'pitch_type' if 'pitch_type' in df.columns else 'pitch_name'
                if col in df.columns: df_filtered = df_filtered[df_filtered[col] == pitch_code]
            if batter_stand != "All":
                df_filtered = df_filtered[df_filtered['stand'] == batter_stand]
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

            # --- 描画 ---
            col_res1, col_res2 = st.columns([3, 1])
            with col_res1:
                fig, ax = plt.subplots(figsize=(8, 8))
                
                sz_top, sz_bottom, plate_width = 3.5, 1.5, 17/12
                ax.add_patch(patches.Rectangle((-plate_width/2, sz_bottom), plate_width, sz_top-sz_bottom, fill=False, edgecolor='black', lw=2, ls='--'))
                ax.add_patch(patches.Polygon([(-plate_width/2, 0), (plate_width/2, 0), (plate_width/2, 0.2), (0, 0.4), (-plate_width/2, 0.2)], color='gray', alpha=0.3))
                
                stand_draw = batter_stand if batter_stand != "All" else 'L'
                base_x = -2.5 if stand_draw == 'R' else 2.5
                ax.add_patch(patches.Ellipse((base_x, 3.0), 2.0, 6.0, color='gray', alpha=0.3))

                df_plot = df_filtered.dropna(subset=['plate_x', 'plate_z'])
                
                if df_plot.empty:
                    st.info(f"条件に該当するデータがありません (元のデータ数: {len(df_filtered)})")
                
                # A. Density
                elif analysis_type == 'density':
                    try:
                        sns.kdeplot(data=df_plot, x='plate_x', y='plate_z', fill=True, cmap='Reds', alpha=0.6, ax=ax, thresh=0.05)
                    except: pass 
                    ax.scatter(df_plot['plate_x'], df_plot['plate_z'], s=15, color='black', alpha=0.2, label='Pitch')
                    ax.set_title(f"Pitch Density (n={len(df_plot)})")
                
                # B. Grid Maps
                else:
                    grid_size = 5
                    x_edges = np.linspace(-2.0, 2.0, grid_size + 1)
                    z_edges = np.linspace(0.5, 4.5, grid_size + 1)
                    
                    if analysis_type == 'ops':
                        metric_name = 'OPS'; vmin, vmax = 0.4, 1.2; cmap = 'coolwarm'
                    elif analysis_type == 'ba':
                        metric_name = 'AVG'; vmin, vmax = 0.100, 0.400; cmap = 'coolwarm'
                    elif analysis_type == 'woba':
                        metric_name = 'wOBA'; vmin, vmax = 0.200, 0.500; cmap = 'coolwarm'
                    elif analysis_type == 'hard_hit':
                        metric_name = 'HardHit%'; vmin, vmax = 0.2, 0.6; cmap = 'Reds'
                    elif analysis_type == 'barrel':
                        metric_name = 'Barrel%'; vmin, vmax = 0.0, 0.2; cmap = 'Reds'
                    
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                    m = cm.ScalarMappable(norm=norm, cmap=cmap)

                    for i in range(grid_size):
                        for j in range(grid_size):
                            x_min, x_max = x_edges[j], x_edges[j+1]
                            z_min, z_max = z_edges[i], z_edges[i+1]
                            
                            in_zone = df_plot[
                                (df_plot['plate_x'] >= x_min) & (df_plot['plate_x'] < x_max) &
                                (df_plot['plate_z'] >= z_min) & (df_plot['plate_z'] < z_max)
                            ]
                            
                            if len(in_zone) > 0:
                                val = np.nan
                                count_label = ""
                                
                                if analysis_type == 'ops':
                                    denom = in_zone['is_at_bat'].sum()
                                    if denom > 0:
                                        obp_d = in_zone['is_obp_denom'].sum()
                                        obp = in_zone['is_on_base'].sum() / obp_d if obp_d > 0 else 0
                                        slg = in_zone['slugging_base'].sum() / denom
                                        val = obp + slg
                                        count_label = f"PA:{len(in_zone)}"
                                elif analysis_type == 'ba':
                                    denom = in_zone['is_at_bat'].sum()
                                    if denom > 0:
                                        val = in_zone['is_hit'].sum() / denom
                                        count_label = f"AB:{denom}"
                                elif analysis_type == 'woba':
                                    val = in_zone['woba_value'].mean()
                                    count_label = f"n:{len(in_zone)}"
                                elif analysis_type == 'hard_hit':
                                    val = in_zone['is_hard_hit'].mean()
                                    count_label = f"n:{len(in_zone)}"
                                elif analysis_type == 'barrel':
                                    val = in_zone['is_barrel'].mean()
                                    count_label = f"n:{len(in_zone)}"
                                
                                if not np.isnan(val):
                                    color = m.to_rgba(val)
                                    rect = patches.Rectangle((x_min, z_min), x_max-x_min, z_max-z_min, linewidth=0.5, edgecolor='gray', facecolor=color, alpha=0.8)
                                    ax.add_patch(rect)
                                    txt_color = 'white' if (norm(val) > 0.7 or norm(val) < 0.3) else 'black'
                                    fmt = ".3f" if analysis_type in ['ops', 'ba', 'woba'] else ".1%"
                                    ax.text((x_min+x_max)/2, (z_min+z_max)/2, f"{val:{fmt}}\n({count_label})", 
                                            ha='center', va='center', fontsize=7, color=txt_color)

                    ax.set_title(f"{metric_name} Map")
                    plt.colorbar(m, ax=ax, label=metric_name)

                ax.set_xlim(2.5, -2.5); ax.set_ylim(0, 5.0); ax.set_aspect('equal')
                ax.set_xlabel("Catcher's View (ft)")
                st.pyplot(fig)

            with col_res2:
                st.markdown("### Summary")
                st.info(get_metrics_summary(df_filtered))
                st.write(f"Total: {len(df_filtered)}")
                
                st.markdown("### Data")
                cols = ['game_date', 'events', 'description', 'pitch_type', 'launch_speed', 'launch_angle']
                valid_cols = [c for c in cols if c in df_filtered.columns]
                st.dataframe(df_filtered[valid_cols].head(20), height=400)

# 実行ブロック
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("予期せぬエラーが発生しました。")
        st.error(traceback.format_exc()) # 詳細なエラーログを表示
