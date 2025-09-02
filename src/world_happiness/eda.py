import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

def perform_eda(df: pd.DataFrame):
    """
    Perform advanced exploratory data analysis with nice plots.
    """
    # Summary statistics
    print(df.describe())
    
    # Time-series analysis: Average happiness over years
    avg_happiness = df.groupby('Year')['Happiness Score'].mean()
    plt.figure(figsize=(10, 6))
    avg_happiness.plot(kind='line', marker='o')
    plt.title('Average World Happiness Score Over Years')
    plt.xlabel('Year')
    plt.ylabel('Average Happiness Score')
    plt.grid(True)
    plt.savefig('avg_happiness.png')  # Save for docs
    plt.show()
    
    # Interactive heatmap with Plotly
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
    fig.update_layout(title='Correlation Heatmap of Features')
    fig.write_html('correlation_heatmap.html')  # Save for docs
    fig.show()
    
    # Anomaly detection: Z-score for happiness outliers
    df['Happiness Z-Score'] = np.abs((df['Happiness Score'] - df['Happiness Score'].mean()) / df['Happiness Score'].std())
    anomalies = df[df['Happiness Z-Score'] > 3]
    print(f"Detected {len(anomalies)} anomalies in happiness scores.")
    
    # World map plot: Choropleth with Plotly (nice plot)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df_latest = df[df['Year'] == df['Year'].max()]
    merged = world.merge(df_latest, left_on='name', right_on='Country', how='left')
    fig = px.choropleth(merged,
                        locations='iso_a3',
                        color='Happiness Score',
                        hover_name='Country',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='World Happiness Scores (Latest Year)')
    fig.write_html('world_happiness_map.html')
    fig.show()
    
    # Another nice plot: Scatter with trendline
    fig_scatter = px.scatter(df, x='GDP per capita', y='Happiness Score', color='Year',
                             trendline='ols', title='Happiness vs GDP with Trendline')
    fig_scatter.write_html('happiness_vs_gdp.html')
    fig_scatter.show()