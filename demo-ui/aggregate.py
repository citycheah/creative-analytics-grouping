import pandas as pd
import numpy as np

def compute_themes_summary(assets_raw: pd.DataFrame, assets_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the themes_summary rollup from the raw and clustered data.
    Equivalent to BigQuery: 
    SELECT cluster_id, ANY_VALUE(cluster_label) as theme_label, 
    SUM(spend) as total_spend, SUM(impressions) as total_impressions,
    COUNT(*) as creative_count
    FROM assets_raw JOIN assets_clustered USING(asset_id)
    GROUP BY cluster_id
    """
    # Merge datasets avoiding duplicate column collision
    overlap = [c for c in assets_clustered.columns if c in assets_raw.columns and c != "asset_id"]
    cl_clean = assets_clustered.drop(columns=overlap)
    merged = pd.merge(assets_raw, cl_clean, on="asset_id")
    
    # Calculate totals for percentage
    total_system_spend = assets_raw['spend'].sum()
    
    # Group by cluster
    summary = merged.groupby('cluster_id').agg(
        theme_label=('cluster_label', 'first'),
        total_spend=('spend', 'sum'),
        total_impressions=('impressions', 'sum'),
        creative_count=('asset_id', 'count')
    ).reset_index()
    
    # Calculate percentage
    summary['pct_of_total_spend'] = (summary['total_spend'] / total_system_spend) * 100
    
    # Placeholder for recommendation (would be a Gemini call in real life)
    summary['recommendation'] = summary['cluster_id'].apply(
        lambda x: "Exclude these from analysis." if x == -1 else "Consider reallocating spend to this theme."
    )
    
    # Sort by spend descending
    summary = summary.sort_values('total_spend', ascending=False)
    
    return summary

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    try:
        raw = pd.read_csv(os.path.join(data_dir, "assets_raw.csv"))
        clustered = pd.read_csv(os.path.join(data_dir, "assets_clustered.csv"))
        
        summary = compute_themes_summary(raw, clustered)
        
        # Save output
        summary.to_csv(os.path.join(data_dir, "themes_summary_computed.csv"), index=False)
        print("Successfully computed themes_summary_computed.csv")
    except FileNotFoundError:
        print("Mock data not found. Run generate_mock_data.py first.")
