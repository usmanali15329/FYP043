import pandas as pd
import streamlit as st
from data_processing import load_data, preprocess_data, omit_empty_columns, filter_data_by_date
from visualization import (
    plot_incidents_over_time,
    plot_geographical_distribution,
    plot_heatmap,
    plot_pie_chart,
    plot_bar_chart,
    plot_feature_importance
)
from predict import train_model, predict_future_incidents
import datetime
import os

# Load and preprocess data
data_path = 'globalterrorismdb.xlsx'
data = load_data(data_path)
data = preprocess_data(data, num_threads=128)  # Use the maximum number of threads
data = omit_empty_columns(data)

# Ensure 'date' column is in datetime format
data['date'] = pd.to_datetime(data['date'])

# Streamlit interface
st.title('Terrorism Incidents in Pakistan')
st.sidebar.title('Visualization Options')

# Interactive Time Slider
st.sidebar.subheader('Time Range Selection')
min_date = data['date'].min().to_pydatetime()
max_date = data['date'].max().to_pydatetime()

start_date, end_date = st.sidebar.slider(
    'Select a date range',
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY"
)
filtered_data = filter_data_by_date(data, start_date, end_date)

# Summary Statistics
st.sidebar.subheader('Summary Statistics')
st.sidebar.write(f"Total Incidents: {len(filtered_data)}")
st.sidebar.write(f"Total Fatalities: {filtered_data['nkill'].sum()}")
st.sidebar.write(f"Total Injured: {filtered_data['nwound'].sum()}")

# Plot incidents over time
if st.sidebar.checkbox('Plot Incidents Over Time'):
    plot_incidents_over_time(filtered_data)

# Plot geographical distribution
if st.sidebar.checkbox('Plot Geographical Distribution'):
    plot_geographical_distribution(filtered_data)

# Plot heatmap
if st.sidebar.checkbox('Plot Heatmap'):
    plot_heatmap(filtered_data)

# Plot attack type distribution (Pie Chart)
if st.sidebar.checkbox('Plot Attack Type Distribution'):
    plot_pie_chart(filtered_data, 'attacktype1_txt')

# Plot target type distribution (Pie Chart)
if st.sidebar.checkbox('Plot Target Type Distribution'):
    plot_pie_chart(filtered_data, 'targtype1_txt')

# Plot bar chart for number of incidents per city
if st.sidebar.checkbox('Plot Number of Incidents per City'):
    plot_bar_chart(filtered_data, 'city')

# Machine Learning Model Training and Prediction
st.sidebar.subheader('Machine Learning Model')
model_name = st.sidebar.selectbox(
    'Select Model', 
    ['Linear Regression', 'Decision Tree', 'Random Forest', 'Support Vector Regression', 'Gradient Boosting']
)

model = None  # Initialize the model variable

if st.sidebar.button('Train Model'):
    model, X_test, y_test = train_model(filtered_data, model_name)
    st.sidebar.write(f"{model_name} model trained successfully")

    # Feature importance
    st.sidebar.subheader('Feature Importance')
    plot_feature_importance(model_name, X_test, y_test)

# Predict future incidents
st.sidebar.subheader('Predict Future Incidents')
prediction_months = st.sidebar.slider('Select number of months for prediction', 1, 12, 3)

if st.sidebar.button('Predict'):
    if model is None:
        # Train the model if not already trained
        model, X_test, y_test = train_model(filtered_data, model_name)
    future_predictions = predict_future_incidents(filtered_data, model_name)
    st.write(f"Predicted incidents for the next {prediction_months} months:")
    st.dataframe(future_predictions)
