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
# ライブラリ読み込み
# ----------------------------------------------------------------------
try:
    import pybaseball
    from pybaseball import statcast_pitcher, statcast_batter, playerid_lookup, statcast
    pybaseball.cache.enable()
except ImportError as e:
    st.error(f"ライブラリの読み込みに失敗しました: {e}")
    st.stop()

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

# ----------------------------------------------------------------------
# セッションステート初期化
# ----------------------------------------------------------------------
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = pd.DataFrame()
    st.session_state.data_params = None # 取得時のパラメータ (表示用)

# ----------------------------------------------------------------------
# 1. データ取得関数 (シンプル版)
# ----------------------------------------------------------------------
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
    except Exception as e:
        raise e 

# ----------------------------------------------------------------------
# 2. データ加工 & 描画補助関数
# ----------------------------------------------------------------------
def process_statcast_data(df_input):
    if df_input.empty: return df_input
    df = df_input.copy()
    if 'game_date' in df.columns: df = df.sort_values('game_date').reset_index(drop=True)

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
    
    df['on_1b_bool'] = df['on_1b'].notna(); df['on_2b_bool'] = df['on_2b'].notna(); df['on_3b_bool'] = df['on_3b'].notna()
    df['is_empty'] = (~df['on_1b_bool']) & (~df['on_2b_bool']) & (~df['on_3b_bool'])
    df['is_risp'] = (df['on_2b_bool']) | (df['on_3b_bool'])
    df['is_on_base_no_risp'] = (df['on_1b_bool']) & (~df['on_2b_bool']) & (~df['on_3b_bool'])

    return df

def get_metrics_summary(df):
    if df.empty: return "No Data"
    pa = df['is_pa_event'].sum(); ba = df['is_hit'].sum() / df['is_at_bat'].sum() if df['is_at_bat'].sum() > 0 else 0.0
    obp = df['is_on_base'].sum() / df['is_obp_denom'].sum() if df['is_obp_denom'].sum() > 0 else 0.0
    slg = df['slugging_base'].sum() / df['is_at_bat'].sum() if df['is_at_bat'].sum() > 0 else 0.0
    ops = obp + slg
    return f"PA: {pa} | BA: {ba:.3f} | OPS: {ops:.3f} | HardHit%: {df['is_hard_hit'].mean():.1%}"

# --- 描画用関数 ---
def draw_5x5_grid(ax):
    """5x5グリッド（ストライクゾーン3x3 + 外枠）を描画"""
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

    rect = patches.Rectangle((sz_left, sz_bot), sz_right-sz_left, sz_top-sz_bot, fill=False, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)
    
    plate_width = 17/12
    ax.add_patch(patches.Polygon([(-plate_width/2, 0), (plate_width/2, 0), (plate_width/2, 0.2), (0, 0.4), (-plate_width/2, 0.2)], color='gray', alpha=0.5))
    return x_lines, z_lines

def draw_batter(ax, stand):
    """打者画像またはシルエットを描画 (投手視点)"""
    img_file = 'batterR.png' if stand == 'R' else 'batterL.png'
    
    if stand == 'R':
        extent = [-4.0, -1.0, 0, 6.0] 
        base_x = -2.5
    else:
        extent = [1.0, 4.0, 0, 6.0]
        base_x = 2.5

    loaded = False
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
# 3. メインアプリケーション
# ----------------------------------------------------------------------
def main():
    st.sidebar.title("⚾ MLB Analyzer Pro")

    # ==========================================
    # STEP 1: データ取得
    # ==========================================
    st.sidebar.markdown("### STEP 1: データ取得 (重い処理)")
    
    # A. 期間
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1: start_date = st.date_input("開始", datetime.date(2025, 3, 27), key="start_date_input")
    with col_d2: end_date = st.date_input("終了", datetime.date(2025, 11, 2), key="end_date_input")

    # A2. 試合タイプ
    selected_game_types_label = st.sidebar.multiselect(
        "対象試合", options=list(GAME_TYPE_MAP.keys()), default=['Regular Season', 'Postseason'], key="step1_game_type"
    )
    selected_game_types_code = [GAME_TYPE_MAP[l] for l in selected_game_types_label]

    # B. 選手選択 (名前検索のみ)
    st.sidebar.subheader("👤 選手選択 (名前検索)")
    st.sidebar.caption("Last Name (姓) をローマ字入力してください")
    
    selected_p_id, selected_p_name = None, ""
    selected_b_id, selected_b_name = None, ""

    # 投手
    p_search = st.sidebar.text_input("投手 姓 (例: darvish)", key="p_search")
    p_search_lower = p_search.lower().strip()
    if p_search_lower:
        try:
            found = playerid_lookup(p_search_lower)
            if not found.empty:
                found['label'] = found['name_first'] + " " + found['name_last'] + " (" + found['mlb_played_first'].astype(str) + "-" + found['mlb_played_last'].astype(str) + ")"
                p_choice = st.sidebar.selectbox("候補 (P)", ["指定なし"] + found['label'].tolist(), key="p_choice")
                if p_choice != "指定なし":
                    row = found[found['label'] == p_choice].iloc[0]
                    selected_p_id, selected_p_name = int(row['key_mlbam']), f"{row['name_first']} {row['name_last']}"
            else: st.sidebar.warning(f"投手 '{p_search}' の候補が見つかりませんでした。")
        except Exception as e: st.sidebar.error(f"検索エラー: {e}")
    
    # 打者
    b_search = st.sidebar.text_input("打者 姓 (例: ohtani)", key="b_search")
    b_search_lower = b_search.lower().strip()
    if b_search_lower:
        try:
            found = playerid_lookup(b_search_lower)
            if not found.empty:
                found['label'] = found['name_first'] + " " + found['name_last'] + " (" + found['mlb_played_first'].astype(str) + "-" + found['mlb_played_last'].astype(str) + ")"
                b_choice = st.sidebar.selectbox("候補 (B)", ["指定なし"] + found['label'].tolist(), key="b_choice")
                if b_choice != "指定なし":
                    row = found[found['label'] == b_choice].iloc[0]
                    selected_b_id, selected_b_name = int(row['key_mlbam']), f"{row['name_first']} {row['name_last']}"
            else: st.sidebar.warning(f"打者 '{b_search}' の候補が見つかりませんでした。")
        except Exception as e: st.sidebar.error(f"検索エラー: {e}")
            
    # データ取得実行ボタン
    if st.sidebar.button("データ取得 (Get Data) 📥", type="primary", key="get_data_button"):
        
        if not selected_p_id and not selected_b_id and (end_date - start_date).days > 14:
             st.warning(f"選手が指定されていません。期間({(end_date - start_date).days}日)が長すぎるため、タイムアウトする可能性が高いです。続行します。")

        with st.spinner('データ取得中... (時間がかかります)'):
            try:
                df_raw = get_statcast_data_safe(str(start_date), str(end_date), selected_p_id, selected_b_id, selected_game_types_code)
                
                if df_raw.empty:
                    st.session_state.raw_data = pd.DataFrame()
                    st.session_state.data_params = None
                    st.error("データが見つかりませんでした。条件を変更してください。")
                else:
                    # 取得したデータをセッションステートに保存
                    st.session_state.raw_data = df_raw
                    st.session_state.data_params = (selected_p_name, selected_b_name, str(start_date), str(end_date), ", ".join(selected_game_types_label))
                    st.success(f"データ取得完了: {len(df_raw)} 球 (セッションに保存済)")
            except Exception as e:
                st.session_state.raw_data = pd.DataFrame()
                st.session_state.data_params = None
                st.error(f"データ取得中にエラーが発生しました。期間を短くしてください。: {e}")


    # ==========================================
    # STEP 2: フィルター & 分析
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("### STEP 2: フィルター & 分析 (高速処理)")

    if st.session_state.raw_data.empty:
        st.info("データがありません。STEP 1でデータを取得してください。")
    else:
        # 取得パラメータの表示
        p_name, b_name, s_date, e_date, g_types = st.session_state.data_params
        
        # タイトル
        title_str = "League Wide"
        if p_name: title_str = f"P: {p_name}"
        if b_name: title_str += f" vs B: {b_name}"
        st.subheader(f"⚾ {title_str}")
        st.caption(f"Period: {s_date} ~ {e_date} | Game Types: {g_types} | Total Pitches: {len(st.session_state.raw_data)}")
        st.markdown("---")

        # C. 詳細フィルター
        with st.sidebar.expander("⚙️ 詳細フィルター", expanded=True):
            pitch_code = st.selectbox("球種", ['', 'FF', 'SL', 'CU', 'CH', 'FS', 'SI', 'FC', 'ST'], format_func=lambda x: "All" if x == "" else x, key="filter_pitch_code")
            batter_stand = st.radio("打席", ["All", "R", "L"], horizontal=True, index=0, key="filter_batter_stand")
            c1, c2 = st.columns(2)
            with c1:
                target_balls = st.selectbox("ボール", ['', '0', '1', '2', '3'], key="filter_balls")
                target_outs = st.selectbox("アウト", ['', '0', '1', '2'], key="filter_outs")
            with c2:
                target_strikes = st.selectbox("ストライク", ['', '0', '1', '2'], key="filter_strikes")
                target_runners = st.selectbox("走者", ['', 'Empty', 'RISP', 'On Base (Not RISP)'], key="filter_runners")

            target_bb_type = st.selectbox("打球タイプ", ['', 'ground_ball', 'fly_ball', 'line_drive', 'popup'], key="filter_bb_type")
            target_result = st.selectbox("結果", ['', 'strikeout', 'walk', 'single', 'double', 'triple', 'home_run', 'hit_into_play', 'woba_zero'], key="filter_result")

        # D. 分析タイプ
        ANALYSIS_OPTIONS = {
            'Density (投球分布)': 'density',
            'OPS Map (OPS)': 'ops',
            'Batting Avg Map (打率)': 'ba',
            'wOBA Map (wOBA)': 'woba',
            'Hard Hit% Map (強打率)': 'hard_hit',
            'Barrel% Map (バレル率)': 'barrel'
        }
        analysis_label = st.sidebar.selectbox("📊 分析タイプ", list(ANALYSIS_OPTIONS.keys()), key="filter_analysis_type")
        analysis_type = ANALYSIS_OPTIONS[analysis_label]

        # グラフ描画実行ボタン
        if st.sidebar.button("グラフ描画 (Analyze & Plot) 📊", type="secondary", key="plot_button"):
            
            df = process_statcast_data(st.session_state.raw_data)
            df_filtered = df.copy()
            
            # --- フィルター適用 ---
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

            # 描画
            col_res1, col_res2 = st.columns([3, 1])
            with col_res1:
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # 5x5グリッド & 打者画像
                x_grid, z_grid = draw_5x5_grid(ax)
                stand_draw = batter_stand if batter_stand != "All" else 'R' 
                draw_batter(ax, stand_draw)

                df_plot = df_filtered.dropna(subset=['plate_x', 'plate_z'])
                
                if df_plot.empty:
                    st.info(f"条件に該当するデータがありません (元のデータ数: {len(df_filtered)})")
                
                # A. Density
                elif analysis_type == 'density':
                    try: sns.kdeplot(data=df_plot, x='plate_x', y='plate_z', fill=True, cmap='Reds', alpha=0.6, ax=ax, thresh=0.05)
                    except: pass 
                    ax.scatter(df_plot['plate_x'], df_plot['plate_z'], s=15, color='black', alpha=0.2, label='Pitch')
                    ax.set_title(f"Pitch Density (n={len(df_plot)})")
                
                # B. Grid Maps
                else:
                    if analysis_type == 'ops': metric = 'OPS'; vmin, vmax = 0.4, 1.2; cmap = 'coolwarm'
                    elif analysis_type == 'ba': metric = 'AVG'; vmin, vmax = 0.1, 0.4; cmap = 'coolwarm'
                    elif analysis_type == 'woba': metric = 'wOBA'; vmin, vmax = 0.2, 0.5; cmap = 'coolwarm'
                    elif analysis_type == 'hard_hit': metric = 'HardHit%'; vmin, vmax = 0.2, 0.6; cmap = 'Reds'
                    elif analysis_type == 'barrel': metric = 'Barrel%'; vmin, vmax = 0.0, 0.2; cmap = 'Reds'
                    
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                    m = cm.ScalarMappable(norm=norm, cmap=cmap)

                    for i in range(5): # z
                        for j in range(5): # x
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

                # 投手視点の設定 (左がマイナス、右がプラス)
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(0, 6.0)
                ax.set_aspect('equal')
                ax.set_xlabel("Pitcher's View (Left=RHB, Right=LHB)")
                ax.set_title(f"{analysis_type.upper()} Map (Pitcher's View)")
                st.pyplot(fig)

            with col_res2:
                st.markdown("### Summary")
                st.info(get_metrics_summary(df_filtered))
                st.dataframe(df_filtered[['game_date', 'events', 'description', 'pitch_type', 'launch_speed']].head(20))

if __name__ == "__main__":
    try: main()
    except Exception as e:
        st.error("エラーが発生しました")
        st.code(traceback.format_exc())
