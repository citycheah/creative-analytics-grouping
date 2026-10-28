import json
import os
import sys
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure root directory is on sys.path for config & export imports
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import config as cfg
from export import generate_pdf_report

brand_name = getattr(cfg, "ADVERTISER_NAME", "Maxis")
country_name = getattr(cfg, "COUNTRY", "Malaysia")

# Set page config for a wider layout
st.set_page_config(
    page_title=f"{brand_name} ({country_name}) — Creative Theme & Category Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a sleek modern Telco Intelligence aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #065f46;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        border: 1px solid #d1fae5;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(6, 95, 70, 0.05);
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(6, 95, 70, 0.09);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #065f46;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    .insight-banner {
        background: linear-gradient(90deg, #ecfdf5 0%, #f8fafc 100%);
        border: 1px solid #a7f3d0;
        border-left: 5px solid #059669;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 14px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header Row
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown(f'<div class="main-title">📡 {brand_name} ({country_name}) — Creative Theme & Category Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Multimodal AI clustering pipeline consolidating multi-format video & display creatives into unified strategic themes, cross-brand Malaysia Telco Share-of-Voice (SOV), and FinOps telemetry.</div>',
        unsafe_allow_html=True,
    )

with head_col2:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    try:
        pdf_data = generate_pdf_report()
        st.download_button(
            label="📄 Export Executive PDF Report",
            data=pdf_data,
            file_name=f"{brand_name.lower().replace(' ', '_')}_{country_name.lower()}_creative_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"PDF export unavailable: {e}")


def render_asset_media(source_url: str, media_type: str = "video"):
    """Renders local visual creative card or remote CDN media with clean fallback."""
    if not isinstance(source_url, str) or not source_url.strip():
        st.warning("No media URL available for this asset.")
        return

    # Check if source_url points to a local file in repo_root
    local_candidate = source_url if os.path.isabs(source_url) else os.path.join(repo_root, source_url)
    if os.path.exists(local_candidate):
        if local_candidate.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            st.image(local_candidate, use_container_width=True)
            return
        elif local_candidate.lower().endswith((".mp4", ".mov", ".webm")):
            st.video(local_candidate)
            return

    url_lower = source_url.lower()
    is_video = url_lower.endswith((".mp4", ".mov", ".webm")) or "video" in str(media_type).lower()
    try:
        if url_lower.startswith(("http://", "https://")):
            if is_video:
                st.video(source_url)
            else:
                st.image(source_url, use_container_width=True)
        else:
            st.info(f"Preview card: `{os.path.basename(source_url)}`")
    except Exception as e:
        st.warning(f"Could not load media directly: {e}")


@st.cache_data
def load_data():
    data_dir = os.path.join(current_dir, "data")
    eval_dir = os.path.join(repo_root, "evaluation")

    try:
        assets_raw = pd.read_csv(os.path.join(data_dir, "assets_raw.csv"))
        assets_clustered = pd.read_csv(os.path.join(data_dir, "assets_clustered.csv"))
        themes_summary = pd.read_csv(os.path.join(data_dir, "themes_summary.csv"))

        assets_raw["asset_id"] = assets_raw["asset_id"].astype(str)
        assets_clustered["asset_id"] = assets_clustered["asset_id"].astype(str)

        themes_summary["theme_label"] = themes_summary["theme_label"].fillna(
            themes_summary["cluster_id"].apply(lambda c: "Unclustered / unrelated" if c == -1 else f"Theme {c}")
        )
        assets_clustered["cluster_label"] = assets_clustered["cluster_label"].fillna(
            assets_clustered["cluster_id"].apply(lambda c: "Unclustered / unrelated" if c == -1 else f"Theme {c}")
        )

        if "first_seen" in assets_raw.columns:
            assets_raw["first_seen_dt"] = pd.to_datetime(assets_raw["first_seen"], errors="coerce")
        if "last_seen" in assets_raw.columns:
            assets_raw["last_seen_dt"] = pd.to_datetime(assets_raw["last_seen"], errors="coerce")

        # Load optional Competitive & Ablation artifacts
        comp_themes_path = os.path.join(data_dir, "competitive_themes.csv")
        comp_assets_path = os.path.join(data_dir, "competitive_assets.csv")
        comp_metrics_path = os.path.join(data_dir, "competitive_brand_metrics.csv")
        ablation_path = os.path.join(eval_dir, "ablation_report.json")

        comp_themes = pd.read_csv(comp_themes_path) if os.path.exists(comp_themes_path) else pd.DataFrame()
        comp_assets = pd.read_csv(comp_assets_path) if os.path.exists(comp_assets_path) else pd.DataFrame()
        comp_metrics = pd.read_csv(comp_metrics_path) if os.path.exists(comp_metrics_path) else pd.DataFrame()

        ablation_data = {}
        if os.path.exists(ablation_path):
            with open(ablation_path, "r", encoding="utf-8") as fh:
                ablation_data = json.load(fh)

        return assets_raw, assets_clustered, themes_summary, comp_themes, comp_assets, comp_metrics, ablation_data
    except FileNotFoundError as e:
        st.error(f"Data file error: {e}. Run `python scripts/generate_synthetic_portfolio.py` and `python run_pipeline.py` first.")
        st.stop()


assets_raw, assets_clustered, themes_summary, comp_themes, comp_assets, comp_metrics, ablation_data = load_data()

# Merge raw data with cluster data
raw_cols = [
    "asset_id", "source_url", "creative_campaign_name", "social_campaign_text",
    "advertiser_name", "channel_name", "social_ad_placement"
]
if "first_seen_dt" in assets_raw.columns:
    raw_cols.extend(["first_seen_dt", "last_seen_dt"])

merged_df = pd.merge(
    assets_clustered,
    assets_raw[[c for c in raw_cols if c in assets_raw.columns]],
    on="asset_id",
    how="left",
)

# Separate valid themes and noise
valid_themes = themes_summary[themes_summary["cluster_id"] != -1].sort_values("total_spend", ascending=False)
noise_theme = themes_summary[themes_summary["cluster_id"] == -1]

# Executive Metrics
total_spend = themes_summary["total_spend"].sum()
num_raw_creatives = len(assets_clustered)
num_distinct_renders = themes_summary["distinct_renders"].sum() if "distinct_renders" in themes_summary else len(assets_clustered)
num_themes = len(valid_themes)

top_theme = valid_themes.iloc[0] if len(valid_themes) > 0 else None
top_theme_name = top_theme["theme_label"] if top_theme is not None else "Top Theme"
top_theme_pct = top_theme["pct_of_total_spend"] if top_theme is not None else 0

# Compute 5G Convergence & Home Fibre family spend concentration
convergence_mask = valid_themes["theme_label"].astype(str).str.contains("5G|Fibre", case=False, na=False)
convergence_spend = valid_themes[convergence_mask]["total_spend"].sum()
convergence_pct = (100.0 * convergence_spend / total_spend) if total_spend > 0 else 0.0

st.markdown("### Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Spend (USD)</div>
        <div class="metric-value">${total_spend:,.0f}</div>
        <div class="metric-sub">Across all {country_name} ad occurrences</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Raw Creatives</div>
        <div class="metric-value">{num_raw_creatives}</div>
        <div class="metric-sub">Ad placement occurrences</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Distinct Renders</div>
        <div class="metric-value">{num_distinct_renders}</div>
        <div class="metric-sub">Unique media files (pHash deduped)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Identified Themes</div>
        <div class="metric-value">{num_themes}</div>
        <div class="metric-sub">Top: {top_theme_name[:22]} ({top_theme_pct:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-banner">
    <b>💡 Key Strategic Finding:</b> 5G & Home Fibre Convergence campaigns account for <b>${convergence_spend:,.0f} ({convergence_pct:.1f}%)</b> of total portfolio spend, led by <b>{top_theme_name}</b> ({top_theme_pct:.1f}%). Perceptual hashing compressed {num_raw_creatives} raw occurrences into {num_distinct_renders} distinct renders (-{100.0*(1 - num_distinct_renders/max(num_raw_creatives, 1)):.1f}% redundant AI compute).
</div>
""", unsafe_allow_html=True)

# Executive Briefing Expander
summary_file = os.path.join(current_dir, "data", "executive_summary.md")
if os.path.exists(summary_file):
    with open(summary_file, "r", encoding="utf-8") as fh:
        exec_md = fh.read()
    with st.expander("🤖 Strategic AI Executive Briefing (Click to expand / collapse)", expanded=False):
        st.markdown(exec_md)

# Main Dashboard Tabs (6 Tabs: Core 4 + Direction B Competitive + Direction C Ablation/FinOps)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Cluster Explorer (2D UMAP)",
    "📋 Ranked Themes Performance",
    "🔍 Theme Inspector & Assets",
    "📈 Temporal Flighting & Channels",
    "🏆 Malaysia Telco Competitive Intelligence",
    "🔬 Methodology Ablation & FinOps Lab",
])

# --- TAB 1: UMAP ---
with tab1:
    st.subheader("2D Semantic Space Projection (UMAP)")
    st.markdown(
        "Each point represents a creative render. Distance reflects combined **visual + text semantic similarity** "
        "(weighted `0.3` visual / `0.7` text, pre-reduced to `30-D` UMAP + centroid merged at `>= 0.94`). Point size corresponds to advertising spend."
    )

    try:
        chart_df = merged_df[merged_df["cluster_id"] != -1].copy()
        chart_df["spend_fmt"] = chart_df["spend"].apply(lambda x: f"${x:,.0f}")
        chart_df["impressions_fmt"] = chart_df["impressions"].apply(lambda x: f"{x:,.0f}")
        chart_df["concept_snippet"] = chart_df["core_concept_summary"].astype(str).str.slice(0, 110) + "..."

        scatter = alt.Chart(chart_df).mark_circle(opacity=0.85, stroke="#1e293b", strokeWidth=0.5).encode(
            x=alt.X("umap_x:Q", title="UMAP Dimension 1", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("umap_y:Q", title="UMAP Dimension 2", axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("cluster_label:N", legend=alt.Legend(title="Creative Theme", orient="right", columns=1)),
            size=alt.Size("spend:Q", title="Spend ($)", scale=alt.Scale(range=[45, 650])),
            tooltip=[
                alt.Tooltip("asset_id:N", title="Asset ID"),
                alt.Tooltip("cluster_label:N", title="Theme"),
                alt.Tooltip("platform:N", title="Platform"),
                alt.Tooltip("aspect_ratio:N", title="Aspect Ratio"),
                alt.Tooltip("spend_fmt:N", title="Spend"),
                alt.Tooltip("impressions_fmt:N", title="Impressions"),
                alt.Tooltip("concept_snippet:N", title="Concept"),
            ],
        ).properties(height=540).interactive()

        st.altair_chart(scatter, use_container_width=True)

        # --- Cluster Asset Media Gallery ---
        st.markdown("---")
        st.subheader("🎬 Cluster Asset Media Gallery")
        st.markdown("Select a creative cluster/theme to inspect its member multi-format renders side-by-side.")

        p_col1, p_col2, p_col3 = st.columns([2, 1.2, 1.5])
        with p_col1:
            theme_list = valid_themes["theme_label"].tolist() + ["All Themes"]
            selected_filter_theme = st.selectbox("Select Cluster / Theme to View:", theme_list, key="umap_theme_filter")

        filtered_preview_df = chart_df.copy()
        if selected_filter_theme != "All Themes":
            filtered_preview_df = filtered_preview_df[filtered_preview_df["cluster_label"] == selected_filter_theme]

        with p_col2:
            top_limit_str = st.selectbox("Show Amount:", ["Top 5", "Top 10", "All Creatives"], index=0, key="umap_top_limit")

        with p_col3:
            rep_only = st.checkbox("Show Key Representatives Only (⭐)", value=True, key="umap_rep_only")
            if rep_only:
                rep_df = filtered_preview_df[filtered_preview_df["is_representative"] == 1]
                if not rep_df.empty:
                    filtered_preview_df = rep_df

        filtered_preview_df = filtered_preview_df.sort_values("spend", ascending=False)
        if top_limit_str == "Top 5":
            preview_items = filtered_preview_df.head(5)
        elif top_limit_str == "Top 10":
            preview_items = filtered_preview_df.head(10)
        else:
            preview_items = filtered_preview_df

        st.caption(f"Displaying **{len(preview_items)}** assets for **{selected_filter_theme}**")

        if preview_items.empty:
            st.info("No assets match current filters.")
        else:
            grid_cols = st.columns(min(len(preview_items), 5))
            for i, (_, row) in enumerate(preview_items.iterrows()):
                col_target = grid_cols[i % len(grid_cols)]
                with col_target:
                    is_rep_badge = " ⭐ Key Rep" if row["is_representative"] == 1 else ""
                    st.markdown(f"**#{i+1}: `{row['asset_id']}`**{is_rep_badge}")
                    st.caption(f"Spend: **${row['spend']:,.0f}** | {row['aspect_ratio']}")
                    render_asset_media(row["source_url"], row["media_type"])
                    if row.get("core_concept_summary") and str(row["core_concept_summary"]).strip():
                        with st.expander("Concept Details"):
                            st.write(row["core_concept_summary"])

    except Exception as err:
        st.error(f"Error rendering Cluster Explorer: {err}")

# --- TAB 2: RANKED THEMES ---
with tab2:
    st.subheader("Creative Themes Ranked by Investment")
    st.markdown("Aggregated performance metrics across all identified creative clusters, ordered by total advertising spend.")

    display_df = valid_themes.copy()
    display_df["cpm"] = (display_df["total_spend"] / display_df["total_impressions"] * 1000).fillna(0.0)
    display_df["total_spend_fmt"] = display_df["total_spend"].apply(lambda x: f"${x:,.0f}")
    display_df["total_impressions_fmt"] = display_df["total_impressions"].apply(lambda x: f"{x:,.0f}")
    display_df["pct_fmt"] = display_df["pct_of_total_spend"].apply(lambda x: f"{x:.1f}%")
    display_df["cpm_fmt"] = display_df["cpm"].apply(lambda x: f"${x:.2f}")

    display_table = display_df[[
        "theme_label", "total_spend_fmt", "pct_fmt", "total_impressions_fmt", "cpm_fmt", "creative_count", "distinct_renders"
    ]]
    display_table.columns = [
        "Theme Label", "Total Spend ($)", "% of Spend", "Impressions", "Effective CPM", "Raw Occurrences", "Distinct Renders"
    ]
    st.dataframe(display_table, use_container_width=True, hide_index=True)

# --- TAB 3: THEME INSPECTOR ---
with tab3:
    st.subheader("Theme Detail & Creative Inspector")
    st.markdown("Select a theme to inspect member creatives, aspect ratios, channels, and Gemini concept summaries.")

    theme_options = valid_themes["theme_label"].tolist() + ["Unclustered Outliers / Noise (Cluster -1)"]
    selected_theme_label = st.selectbox("Select Creative Theme / Group:", theme_options)

    if selected_theme_label == "Unclustered Outliers / Noise (Cluster -1)":
        selected_cid = -1
        if not noise_theme.empty:
            noise_spend = noise_theme.iloc[0]["total_spend"]
            noise_raw = noise_theme.iloc[0]["creative_count"]
            noise_renders = noise_theme.iloc[0]["distinct_renders"]
            st.warning(
                f"Unclustered noise represents **${noise_spend:,.2f}** ({noise_theme.iloc[0]['pct_of_total_spend']:.2f}% of spend), "
                f"covering {noise_raw} raw occurrences across {noise_renders} distinct outlier renders."
            )
    else:
        selected_theme_row = valid_themes[valid_themes["theme_label"] == selected_theme_label].iloc[0]
        selected_cid = selected_theme_row["cluster_id"]

        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            st.metric("Theme Spend", f"${selected_theme_row['total_spend']:,.0f}")
        with t_col2:
            st.metric("% of Total Spend", f"{selected_theme_row['pct_of_total_spend']:.1f}%")
        with t_col3:
            st.metric("Raw Occurrences", selected_theme_row["creative_count"])
        with t_col4:
            st.metric("Distinct Renders", selected_theme_row["distinct_renders"])

    st.markdown("---")
    theme_assets = merged_df[merged_df["cluster_id"] == selected_cid].sort_values("spend", ascending=False)

    if not theme_assets.empty:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("##### Platform Breakdown")
            plat_counts = theme_assets.groupby("platform")["spend"].sum().reset_index()
            plat_chart = alt.Chart(plat_counts).mark_bar(color="#059669").encode(
                x=alt.X("spend:Q", title="Spend ($)"),
                y=alt.Y("platform:N", title="Platform", sort="-x"),
                tooltip=["platform", alt.Tooltip("spend:Q", format="$,.0f")],
            ).properties(height=190)
            st.altair_chart(plat_chart, use_container_width=True)

        with f_col2:
            st.markdown("##### Aspect Ratio Breakdown (Cross-Format Consolidation)")
            aspect_counts = theme_assets.groupby("aspect_ratio")["spend"].sum().reset_index()
            aspect_chart = alt.Chart(aspect_counts).mark_bar(color="#0284c7").encode(
                x=alt.X("spend:Q", title="Spend ($)"),
                y=alt.Y("aspect_ratio:N", title="Ratio", sort="-x"),
                tooltip=["aspect_ratio", alt.Tooltip("spend:Q", format="$,.0f")],
            ).properties(height=190)
            st.altair_chart(aspect_chart, use_container_width=True)

    st.markdown(f"**Member Creatives ({len(theme_assets)} total occurrences)**:")
    assets_table = theme_assets[[
        "asset_id", "is_representative", "aspect_ratio", "platform", "media_type",
        "spend", "impressions", "core_concept_summary", "social_campaign_text"
    ]].copy()
    assets_table["spend"] = assets_table["spend"].apply(lambda x: f"${x:,.2f}")
    assets_table["impressions"] = assets_table["impressions"].apply(lambda x: f"{x:,.0f}")
    assets_table["is_representative"] = assets_table["is_representative"].apply(lambda x: "⭐ Key Rep" if x == 1 else "Variant")
    assets_table.columns = [
        "Asset ID", "Role", "Aspect Ratio", "Platform", "Media", "Spend", "Impressions", "Gemini Concept Summary", "Ad Copy"
    ]
    st.dataframe(assets_table, use_container_width=True, hide_index=True)

# --- TAB 4: TEMPORAL & CROSS-PLATFORM ---
with tab4:
    st.subheader("Temporal Trends & Cross-Platform Dynamics")
    st.markdown("Analyze campaign flighting velocity, cadence lifespan, and cross-channel spend allocations per theme.")

    if "first_seen_dt" in merged_df.columns and merged_df["first_seen_dt"].notna().any():
        temp_df = merged_df[merged_df["cluster_id"] != -1].dropna(subset=["first_seen_dt"]).copy()
        temp_df["week"] = temp_df["first_seen_dt"].dt.to_period("W").dt.start_time
        weekly_theme_spend = temp_df.groupby(["week", "cluster_label"])["spend"].sum().reset_index()
        weekly_theme_spend["spend_fmt"] = weekly_theme_spend["spend"].apply(lambda x: f"${x:,.0f}")

        top_6_labels = valid_themes.head(6)["theme_label"].tolist()
        trend_chart_data = weekly_theme_spend[weekly_theme_spend["cluster_label"].isin(top_6_labels)]

        st.markdown("##### Weekly Spend Velocity (Top 6 Hero Themes)")
        trend_line = alt.Chart(trend_chart_data).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("week:T", title="Flight Week", axis=alt.Axis(format="%b %d")),
            y=alt.Y("spend:Q", title="Weekly Spend ($)"),
            color=alt.Color("cluster_label:N", legend=alt.Legend(title="Theme", orient="top")),
            tooltip=["cluster_label", alt.Tooltip("week:T", format="%Y-%m-%d"), alt.Tooltip("spend_fmt:N", title="Spend")],
        ).properties(height=340).interactive()
        st.altair_chart(trend_line, use_container_width=True)

        st.markdown("---")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("##### Cross-Channel Spend Share by Theme")
            channel_col = "channel_name" if "channel_name" in merged_df.columns else "platform"
            channel_data = merged_df[merged_df["cluster_id"] != -1].groupby(["cluster_label", channel_col])["spend"].sum().reset_index()
            channel_data = channel_data[channel_data["cluster_label"].isin(top_6_labels)]

            stacked_bar = alt.Chart(channel_data).mark_bar().encode(
                x=alt.X("spend:Q", title="Share of Theme Spend", stack="normalize"),
                y=alt.Y("cluster_label:N", title="Theme", sort="-x"),
                color=alt.Color(f"{channel_col}:N", legend=alt.Legend(title="Channel", orient="bottom")),
                tooltip=["cluster_label", f"{channel_col}:N", alt.Tooltip("spend:Q", format="$,.0f")],
            ).properties(height=300)
            st.altair_chart(stacked_bar, use_container_width=True)

        with t_col2:
            st.markdown("##### Theme Lifespan & Flight Cadence")
            lifespan_df = temp_df.groupby("cluster_label").agg(
                first_flight=("first_seen_dt", "min"),
                last_flight=("last_seen_dt", "max"),
                total_spend=("spend", "sum"),
                total_impressions=("impressions", "sum"),
            ).reset_index()
            lifespan_df["duration_days"] = (lifespan_df["last_flight"] - lifespan_df["first_flight"]).dt.days + 1
            lifespan_df["first_flight_str"] = lifespan_df["first_flight"].dt.strftime("%Y-%m-%d")
            lifespan_df["last_flight_str"] = lifespan_df["last_flight"].dt.strftime("%Y-%m-%d")
            lifespan_df = lifespan_df.sort_values("total_spend", ascending=False).head(8)

            display_life = lifespan_df[["cluster_label", "first_flight_str", "last_flight_str", "duration_days", "total_spend"]].copy()
            display_life["total_spend"] = display_life["total_spend"].apply(lambda x: f"${x:,.0f}")
            display_life.columns = ["Theme", "First Flight", "Last Flight", "Duration (Days)", "Spend ($)"]
            st.dataframe(display_life, use_container_width=True, hide_index=True)
    else:
        st.info("Temporal date fields ('first_seen') are not populated for these assets.")

# --- TAB 5: MALAYSIA TELCO COMPETITIVE INTELLIGENCE (DIRECTION B) ---
with tab5:
    st.subheader("🏆 Malaysia Telco Category Intelligence (Maxis vs CelcomDigi vs U Mobile vs Unifi)")
    st.markdown(
        "Cross-advertiser projection in a **Shared Joint Multimodal Embedding Space**, quantifying **Share of Voice (SOV)** "
        "on contested category battlegrounds, **Herfindahl-Hirschman Index ($HHI$) spend concentration**, and **Brand-Exclusive Whitespace**."
    )

    if not comp_metrics.empty and not comp_themes.empty:
        # 1. Brand HHI & Market SOS KPI Row
        b_cols = st.columns(len(comp_metrics))
        for idx, b_row in comp_metrics.iterrows():
            with b_cols[idx % len(b_cols)]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{b_row['brand']} (SOS: {b_row['market_sos_pct']:.1f}%)</div>
                    <div class="metric-value">${b_row['total_spend']:,.0f}</div>
                    <div class="metric-sub"><b>HHI: {b_row['hhi_index']:.3f}</b> &middot; {b_row['active_themes']} Themes ({b_row['exclusive_whitespace_themes']} Exclusive)</div>
                    <div class="metric-sub" style="color:#059669;">{b_row['strategy_posture']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_col1, c_col2 = st.columns([1.35, 1])

        with c_col1:
            st.markdown("##### Shared Category Battlegrounds: Spend & Share of Voice (SOV %)")
            shared_df = comp_themes[comp_themes["classification"] == "Shared Battleground"].copy()
            sov_bar = alt.Chart(shared_df).mark_bar().encode(
                x=alt.X("brand_spend:Q", title="Category Theme Spend ($)"),
                y=alt.Y("category_theme:N", title="Contested Telco Battleground", sort="-x"),
                color=alt.Color(
                    "brand:N",
                    scale=alt.Scale(
                        domain=["Maxis", "CelcomDigi", "U Mobile", "Unifi"],
                        range=["#059669", "#2563eb", "#f97316", "#7c3aed"],
                    ),
                    legend=alt.Legend(title="Telco Brand", orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("category_theme:N", title="Battleground Theme"),
                    alt.Tooltip("brand:N", title="Brand"),
                    alt.Tooltip("brand_spend:Q", title="Spend ($)", format="$,.0f"),
                    alt.Tooltip("theme_sov_pct:Q", title="Theme SOV (%)", format=".1f"),
                    alt.Tooltip("brand_hook:N", title="Creative Hook"),
                ],
            ).properties(height=340)
            st.altair_chart(sov_bar, use_container_width=True)

        with c_col2:
            st.markdown("##### Joint Category 2D UMAP Manifold (4 Telcos)")
            if not comp_assets.empty:
                cat_scatter = alt.Chart(comp_assets).mark_circle(opacity=0.8).encode(
                    x=alt.X("umap_x:Q", title="Joint Category UMAP 1", axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("umap_y:Q", title="Joint Category UMAP 2", axis=alt.Axis(labels=False, ticks=False)),
                    color=alt.Color(
                        "brand:N",
                        scale=alt.Scale(
                            domain=["Maxis", "CelcomDigi", "U Mobile", "Unifi"],
                            range=["#059669", "#2563eb", "#f97316", "#7c3aed"],
                        ),
                    ),
                    size=alt.Size("spend:Q", scale=alt.Scale(range=[30, 320]), legend=None),
                    tooltip=["brand:N", "category_theme:N", "classification:N", "channel:N", "brand_hook:N"],
                ).properties(height=340).interactive()
                st.altair_chart(cat_scatter, use_container_width=True)

        st.markdown("---")
        w_col1, w_col2 = st.columns([1.2, 1])
        with w_col1:
            st.markdown("##### Brand-Exclusive Creative Whitespace (Zero Competitor Overlap)")
            excl_df = comp_themes[comp_themes["classification"] == "Brand-Exclusive Whitespace"][[
                "brand", "category_theme", "proposition_pillar", "brand_spend", "distinct_renders", "brand_hook"
            ]].copy()
            excl_df["brand_spend"] = excl_df["brand_spend"].apply(lambda x: f"${x:,.0f}")
            excl_df.columns = ["Telco Brand", "Exclusive Whitespace Theme", "Pillar", "Spend ($)", "Renders", "Proprietary Hook"]
            st.dataframe(excl_df, use_container_width=True, hide_index=True)

        with w_col2:
            st.markdown("##### Strategic Portfolio Concentration Table ($HHI = \\sum s_i^2$)")
            bm_disp = comp_metrics[[
                "brand", "total_spend", "market_sos_pct", "hhi_index", "strategy_posture", "top_theme", "top_theme_share_pct"
            ]].copy()
            bm_disp["total_spend"] = bm_disp["total_spend"].apply(lambda x: f"${x:,.0f}")
            bm_disp["market_sos_pct"] = bm_disp["market_sos_pct"].apply(lambda x: f"{x:.1f}%")
            bm_disp["top_theme_share_pct"] = bm_disp["top_theme_share_pct"].apply(lambda x: f"{x:.1f}%")
            bm_disp.columns = ["Brand", "Total Spend", "Market SOS", "HHI Index", "Posture", "Hero Theme", "Hero Share"]
            st.dataframe(bm_disp, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python embedding-clustering/competitive.py` to populate competitive category intelligence.")

# --- TAB 6: METHODOLOGY ABLATION & FINOPS LAB (DIRECTION C) ---
with tab6:
    st.subheader("🔬 Methodology Ablation Study & Cloud FinOps Telemetry")
    st.markdown(
        "Empirical proof of our 4 core architectural decisions evaluated against the **blind human-labeled evaluation benchmark**, "
        "plus live cost/latency accounting from `costs_ledger.csv`."
    )

    if ablation_data:
        m_sweep = pd.DataFrame(ablation_data.get("modality_sweep", []))
        arch_ladder = pd.DataFrame(ablation_data.get("architecture_ladder", []))
        finops_df = pd.DataFrame(ablation_data.get("finops_architectures", []))
        geom = ablation_data.get("centroid_geometry", {})

        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown("##### 1. Modality Weight Sweep ($w_{\\text{vis}}$ Curve: Frame vs Video)")
            st.caption("Single-brand telco creatives share brand colors/logos; `0.3 Visual / 0.7 Text` maximizes ARI & AMI.")
            if not m_sweep.empty:
                mod_chart = alt.Chart(m_sweep).mark_line(point=True, strokeWidth=2.5).encode(
                    x=alt.X("visual_weight:Q", title="Visual Embedding Weight (w_vis)"),
                    y=alt.Y("ari:Q", title="Adjusted Rand Index (ARI)", scale=alt.Scale(zero=False)),
                    color=alt.Color("track:N", title="Embedding Track", scale=alt.Scale(domain=["frame", "video"], range=["#059669", "#6366f1"])),
                    tooltip=["config", "track", "visual_weight", "ari", "ami", "clusters", "noise_spend_pct"],
                ).properties(height=290)
                st.altair_chart(mod_chart, use_container_width=True)

        with a_col2:
            st.markdown("##### 2. 4-Stage Pipeline Architecture Ladder (ARI vs Noise Spend %)")
            st.caption("UMAP 30-D eliminates high-dimensional noise (`29.8% -> 0.06%`), while 1792-D Centroid Merge fixes format over-splitting.")
            if not arch_ladder.empty:
                arch_disp = arch_ladder[["stage", "clusters", "noise_spend_pct", "ari", "ami"]].copy()
                arch_disp["noise_spend_pct"] = arch_disp["noise_spend_pct"].apply(lambda x: f"{x:.2f}%")
                arch_disp.columns = ["Pipeline Stage", "Clusters", "Spend in Noise (%)", "ARI", "AMI"]
                st.dataframe(arch_disp, use_container_width=True, hide_index=True)

                st.info(
                    f"**Centroid Geometry Proof:** Pre-merge clusters: **{geom.get('pre_merge_clusters', 15)}** $\\rightarrow$ "
                    f"Post-merge (`cos >= {geom.get('merge_threshold', 0.94)}`): **{geom.get('post_merge_clusters', 12)}** themes. "
                    f"Median off-diagonal cosine in `1792-D` is `{geom.get('median_off_diagonal_cosine_1792d', 0.35):.3f}`, "
                    f"whereas uncentered UMAP cosine is artificially inflated (`{geom.get('median_off_diagonal_cosine_umap30d', 0.91):.3f}`) — "
                    f"proving why merging MUST occur in original $L_2$ space."
                )

        st.markdown("---")
        st.markdown("##### 3. Cloud FinOps Pareto Frontier: Video vs Keyframe + Two-Tier pHash Dedup")
        if not finops_df.empty:
            f_disp = finops_df[[
                "architecture", "dedup_strategy", "visual_mode", "gemini_config",
                "total_cost_usd", "pipeline_latency_min", "ari"
            ]].copy()
            f_disp["total_cost_usd"] = f_disp["total_cost_usd"].apply(lambda x: f"${x:.2f}")
            f_disp["pipeline_latency_min"] = f_disp["pipeline_latency_min"].apply(lambda x: f"{x:.1f} min")
            f_disp.columns = [
                "Pipeline Architecture", "Deduplication", "Visual Encoder", "Gemini Setup",
                "Total Cost (USD)", "End-to-End Latency", "ARI Accuracy"
            ]
            st.dataframe(f_disp, use_container_width=True, hide_index=True)

        ledger_rows = ablation_data.get("ledger_breakdown", [])
        if ledger_rows:
            with st.expander("🧾 View Granular SKU Billing Ledger (`costs_ledger.csv`)", expanded=False):
                l_df = pd.DataFrame(ledger_rows)
                l_df["cost_usd"] = l_df["cost_usd"].apply(lambda x: f"${x:.4f}")
                l_df.columns = ["Stage", "Cloud Service", "Verified SKU", "API Calls", "Billable Units", "Cost (USD)", "Mean Latency (s)"]
                st.dataframe(l_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python embedding-clustering/benchmark_ablation.py` to generate ablation telemetry.")
