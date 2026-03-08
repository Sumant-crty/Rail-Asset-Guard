import os

streamlit_app_content = '''
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(layout="wide")

# --- Paths --- #
PROJECT_ROOT = 'Rail-Asset-Guard'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Ensure paths exist (for local development/Colab execution)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Load Model and Data ---
@st.cache_resource # Cache the model loading for performance
def load_model():
    model_path = os.path.join(MODELS_DIR, 'predictive_maintenance_model.joblib')
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please ensure Step 3 (ML Training) was executed and the model was saved.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data # Cache the data loading for performance
def load_data():
    # In a real scenario, this would load fresh data or a processed dataset
    processed_data_path = os.path.join(DATA_DIR, 'processed_sensor_data.parquet')
    if os.path.exists(processed_data_path):
        df = pd.read_parquet(processed_data_path)
        return df
    else:
        st.warning(f"Processed data not found at {processed_data_path}. Using dummy data for demonstration.")
        # Fallback to generating some dummy data if processed_data.parquet isn't found
        dummy_data = {
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='h')),
            'sensor_1_vibration': np.random.rand(10) * 10 + 50,
            'sensor_2_temperature': np.random.rand(10) * 5 + 80,
            'operational_hours': np.arange(1, 11),
            'load_factor': np.random.rand(10) * 0.5 + 0.5
        }
        dummy_df = pd.DataFrame(dummy_data)
        # Recreate engineered features for dummy data consistency
        dummy_df['vibration_rolling_mean_3h'] = dummy_df['sensor_1_vibration'].rolling(window=3).mean().fillna(0)
        dummy_df['temperature_diff'] = dummy_df['sensor_2_temperature'].diff().fillna(0)
        dummy_df['cumulative_mileage'] = dummy_df['operational_hours'] * 10
        return dummy_df

model = load_model()
raw_data_for_display = load_data()

# --- Prepare Data for Prediction (using the same columns as training) ---
def prepare_input_features(df_input):
    # This should match the features used in model training (X)
    feature_cols = ['operational_hours', 'load_factor', 'vibration_rolling_mean_3h', 'temperature_diff', 'cumulative_mileage']
    return df_input[feature_cols]

# --- Make Predictions ---
def make_prediction(model, features_df):
    predictions = model.predict(features_df)
    probabilities = model.predict_proba(features_df)[:, 1] # Probability of failure
    return predictions, probabilities


# --- Streamlit UI ---
st.title("⚙️ Rail-Asset-Guard: AI-Powered Predictive Maintenance")
st.markdown("### Real-time Asset Health Monitoring and Failure Prediction")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Fleet Overview", "Interactive Predictor", "ROI & Savings"]) # Added tab3

with tab1:
    st.header("Current Fleet Overview")
    st.write("Displays the health status of assets based on recent data.")

    if not raw_data_for_display.empty:
        # Ensure 'raw_data_for_display' has the necessary columns for prediction
        display_df = raw_data_for_display.copy()

        # Add a synthetic 'failure_status' for the display if it's not already there
        # This ensures consistency with the ML training step's target creation logic
        if 'failure_status' not in display_df.columns:
             display_df['failure_status'] = ((display_df['sensor_1_vibration'] > display_df['sensor_1_vibration'].mean() + display_df['sensor_1_vibration'].std()) &
                                            (display_df['sensor_2_temperature'] > display_df['sensor_2_temperature'].mean() + display_df['sensor_2_temperature'].std())) | \
                                            (display_df['temperature_diff'].abs() > display_df['temperature_diff'].std() * 1.5) | \
                                            (display_df['cumulative_mileage'] > display_df['cumulative_mileage'].mean() + display_df['cumulative_mileage'].std())
             display_df['failure_status'] = display_df['failure_status'].astype(int)


        # Make predictions on the loaded data for display
        features_for_display = prepare_input_features(display_df)
        preds_display, probs_display = make_prediction(model, features_for_display)

        display_df['Predicted_Failure'] = preds_display
        display_df['Failure_Probability'] = probs_display

        st.dataframe(display_df[['timestamp', 'operational_hours', 'load_factor', 'sensor_1_vibration',
                                 'sensor_2_temperature', 'vibration_rolling_mean_3h', 'temperature_diff',
                                 'cumulative_mileage', 'Predicted_Failure', 'Failure_Probability']].style.format(subset=['Failure_Probability'], formatter="{:.2%}"),
                      use_container_width=True)
    else:
        st.info("No data to display in the Fleet Overview. Please ensure data is loaded.")

with tab2:
    st.header("Interactive Prediction")
    st.write("Adjust sensor readings and operational parameters to see a real-time failure prediction.")

    # Input widgets for features
    col1, col2, col3 = st.columns(3)
    with col1:
        op_hours = st.slider("Operational Hours", 1, 100, 50)
        load_factor = st.slider("Load Factor", 0.5, 1.0, 0.75, 0.01)
    with col2:
        vibration = st.slider("Sensor 1 Vibration", 40.0, 70.0, 55.0, 0.1)
        temperature = st.slider("Sensor 2 Temperature", 75.0, 90.0, 82.0, 0.1)
    with col3:
        # These are engineered features, will calculate them based on simple rules for demo
        # In a real app, these would come from an input pipeline or more sophisticated estimation
        vibration_rolling_mean = st.slider("Vibration Rolling Mean (3h)", 40.0, 70.0, 55.0, 0.1)
        temp_diff = st.slider("Temperature Difference", -5.0, 5.0, 0.0, 0.1)
        cumulative_mileage = st.slider("Cumulative Mileage", 10, 1000, 500, 10)

    input_data = pd.DataFrame([[op_hours, load_factor, vibration_rolling_mean, temp_diff, cumulative_mileage]],
                              columns=['operational_hours', 'load_factor', 'vibration_rolling_mean_3h', 'temperature_diff', 'cumulative_mileage'])

    if st.button("Predict Failure Status"):
        pred, prob = make_prediction(model, input_data)
        st.subheader("Prediction Result:")
        if pred[0] == 1:
            st.error(f"\n\n**FAILURE PREDICTED!** (Probability: {prob[0]:.2%})\n\nImmediate inspection recommended.")
        else:
            st.success(f"\n\n**Asset is currently HEALTHY.** (Probability of Failure: {prob[0]:.2%})\n\nContinue monitoring.")

with tab3:
    st.header("ROI & Potential Savings")
    st.write("Quantifying the financial impact of predictive maintenance.")

    st.subheader("Cost Savings Simulation")
    cost_unplanned_failure = st.number_input("Cost of an unplanned failure (e.g., breakdown, delays)", value=50000, min_value=0)
    cost_predictive_fix = st.number_input("Cost of a predictive fix (e.g., scheduled maintenance, part replacement)", value=5000, min_value=0)

    if st.button("Calculate Potential Savings"):
        if cost_unplanned_failure > cost_predictive_fix:
            potential_saving_per_event = cost_unplanned_failure - cost_predictive_fix
            st.success(f"By predicting and preventing one unplanned failure, you could save: **${potential_saving_per_event:,.2f}**")
            st.markdown("This demonstrates the significant return on investment (ROI) from transitioning to predictive maintenance.")
        else:
            st.warning("Predictive fix cost is not less than unplanned failure cost. Adjust values.")


'''

app_dir = os.path.join('Rail-Asset-Guard', 'app')
os.makedirs(app_dir, exist_ok=True)

streamlit_app_path = os.path.join(app_dir, 'streamlit_app.py')
with open(streamlit_app_path, 'w') as f:
    f.write(streamlit_app_content.strip())

print(f"Created Streamlit app at {streamlit_app_path}")
