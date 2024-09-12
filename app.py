import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Load the dataset
with st.expander('Data'):
  st.write('## Dataset')
  data= pd.read_csv('https://raw.githubusercontent.com/sumukhahe/ML_Project/main/data/dataset.csv')
  data
  
# Filter data by state
def get_state_data(data, state):
    return data[(data['State_x'] == state) | (data['State_y'] == state)]

# Predict future values using linear regression
def predict_future(data, column):
    # Use only non-null data for regression
    data = data[['year', column]].dropna()
    
    # Reshape the data
    X = data['year'].values.reshape(-1, 1)
    y = data[column].values
    
    # Create the regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future values for 2024 and 2025
    future_years = np.array([2024, 2025]).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    return future_years.flatten(), predictions

# Section for visualization
def visualize_state_data(state_data, selected_state):
    st.subheader(f"Visualizations for {selected_state}")

    # Line plot for MGNREGA demand across years
    if 'year' in state_data.columns and 'Employment_demanded' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(x='year', y='Employment_demanded', data=state_data, ax=ax)
        plt.title(f"MGNREGA Employment Demand Over Years in {selected_state}")
        st.pyplot(fig)

    # Line plot for crop production across years
    if 'year' in state_data.columns and 'Production_(in_Tonnes)' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(x='year', y='Production_(in_Tonnes)', data=state_data, ax=ax)
        plt.title(f"Crop Production Over Years in {selected_state}")
        st.pyplot(fig)

    # Line plot for rainfall across years
    if 'year' in state_data.columns and 'Annual_rainfall' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(x='year', y='Annual_rainfall', data=state_data, ax=ax)
        plt.title(f"Annual Rainfall Over Years in {selected_state}")
        st.pyplot(fig)

    # Line plot for MSP across years
    if 'year' in state_data.columns and 'MSP' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(x='year', y='MSP', data=state_data, ax=ax)
        plt.title(f"Minimum Support Price (MSP) Over Years in {selected_state}")
        st.pyplot(fig)

# Section for predictions
def display_predictions(state_data, selected_state):
    st.subheader(f"Predictions for 2024 and 2025 in {selected_state}")
    
    # List of columns for which predictions will be made
    columns_to_predict = ['Employment_demanded', 'Production_(in_Tonnes)', 'Annual_rainfall', 'MSP']

    for col in columns_to_predict:
        if col in state_data.columns:
            future_years, future_predictions = predict_future(state_data, col)
            st.write(f"Predicted {col} for 2024: {future_predictions[0]:.2f}")
            st.write(f"Predicted {col} for 2025: {future_predictions[1]:.2f}")
            st.markdown("---")

# Main function to render the Streamlit app
def main():
    st.title('MGNREGA and Crop Analysis by State')

    # Sidebar for state selection
    st.sidebar.header('State Selection')
    states = data['State_x'].unique()
    selected_state = st.sidebar.selectbox('Select a state', states)

    # Create sections for exploration, visualization, and prediction
    st.sidebar.markdown("### Sections")
    sections = ['Data Overview', 'Visualizations', 'Predictions for 2024 and 2025']
    selected_section = st.sidebar.radio('Go to', sections)

    # Filter data for the selected state
    state_data = get_state_data(data, selected_state)

    # Data overview section
    if selected_section == 'Data Overview':
        st.subheader(f"Data Overview for {selected_state}")
        st.write(f"Data for {selected_state}")
        st.dataframe(state_data)

        # Generate summary statistics
        st.subheader("Summary Statistics")
        st.write(state_data.describe())

        # QQ plot for normality test
        st.subheader("Normality Test (QQ Plot)")
        numerical_columns = ['Employment_demanded', 'Employment_offered', 'Employment_Availed', 
                             'Area_(in_Ha)', 'Production_(in_Tonnes)', 'Yield_(kg/Ha)', 
                             'Annual_rainfall', 'MSP']

        selected_column = st.selectbox("Select a column for QQ Plot", numerical_columns)
        qq_data = state_data[selected_column].dropna()

        fig, ax = plt.subplots()
        stats.probplot(qq_data, dist="norm", plot=ax)
        st.pyplot(fig)

    # Visualization section
    elif selected_section == 'Visualizations':
        visualize_state_data(state_data, selected_state)

    # Prediction section
    elif selected_section == 'Predictions for 2024 and 2025':
        display_predictions(state_data, selected_state)

if __name__ == "__main__":
    main()
