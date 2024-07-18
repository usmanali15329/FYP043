import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import logging
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_incidents_over_time(data):
    fig = px.line(data, x='date', y=data.index, title='Number of Terrorism Incidents Over Time in Pakistan')
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Incidents')
    fig.update_layout(clickmode='event+select')
    st.plotly_chart(fig, use_container_width=True)
    logging.info("Incidents over time plot generated successfully")

def plot_geographical_distribution(data):
    try:
        st.header('Geographical Distribution of Incidents')

        # Create a Folium map centered around Pakistan with OpenStreetMap tiles
        m = folium.Map(location=[30.3753, 69.3451], zoom_start=5, tiles='OpenStreetMap')

        # Add a marker cluster to the map
        marker_cluster = MarkerCluster().add_to(m)

        # Filter out rows with NaN values in 'latitude' and 'longitude'
        filtered_data = data.dropna(subset=['latitude', 'longitude'])

        # Add markers to the map
        for idx, row in filtered_data.iterrows():
            folium.Marker(location=[row['latitude'], row['longitude']],
                          popup=f"City: {row['city']}<br>Date: {row['date'].strftime('%Y-%m-%d')}<br>Attack Type: {row['attacktype1_txt']}<br>Target Type: {row['targtype1_txt']}<br>Number of Kills: {row['nkill']}",
                          tooltip=row['city']).add_to(marker_cluster)

        # Display the map
        folium_static(m, width=700, height=500)
        logging.info("Geographical distribution plot generated successfully")
    except Exception as e:
        logging.error(f"Error generating geographical distribution plot: {e}")
        raise

def plot_heatmap(data):
    data = data.dropna(subset=['latitude', 'longitude'])  # Drop rows with NaN values in lat/lon
    fig = px.density_mapbox(data, lat='latitude', lon='longitude', z='nkill', radius=10,
                            center=dict(lat=30.3753, lon=69.3451), zoom=5,
                            mapbox_style="satellite", title='Heatmap of Terrorism Incidents in Pakistan')
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
    logging.info("Heatmap plot generated successfully")

def plot_pie_chart(data, column):
    fig = px.pie(data, names=column, title=f'Distribution of {column}')
    fig.update_layout(legend_title_text='Categories', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)
    logging.info("Pie chart generated successfully")

def plot_bar_chart(data, column):
    fig = px.bar(data, x=column, title=f'Number of Incidents per {column}')
    fig.update_layout(xaxis_title=column.capitalize(), yaxis_title='Number of Incidents')
    fig.update_layout(clickmode='event+select')
    st.plotly_chart(fig, use_container_width=True)
    logging.info("Bar chart generated successfully")

def plot_feature_importance(model_name, X_test, y_test):
    try:
        model, features = joblib.load(f'terrorism_model_{model_name}.pkl')
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()

        fig = go.Figure(go.Bar(
            x=result.importances_mean[sorted_idx],
            y=[features[i] for i in sorted_idx],
            orientation='h'
        ))

        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature"
        )

        st.plotly_chart(fig, use_container_width=True)
        logging.info("Feature importance plot generated successfully")
    except Exception as e:
        logging.error(f"Error generating feature importance plot: {e}")
        raise
