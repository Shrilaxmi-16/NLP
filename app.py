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
def predict_future(data, column, years_to_predict):
    # Use only non-null data for regression
    data = data[['year', column]].dropna()
    
    # Reshape the data
    X = data['year'].values.reshape(-1, 1)
    y = data[column].values
    
    # Create the regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future values
    future_years = np.array([2024, 2025]).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    return future_years.flatten(), predictions

# Main function to render the Streamlit app
def main():
    st.title('MGNREGA and Crop Analysis by State')

    # State selection
    states = data['State_x'].unique()
    selected_state = st.selectbox('Select a state', states)

    # Display table for selected state
    state_data = get_state_data(data, selected_state)
    st.write(f"Data for {selected_state}")
    st.dataframe(state_data)

    # Generate summary statistics
    st.subheader("Summary Statistics")
    st.write(state_data.describe())

    # Normality test using QQ plot
    st.subheader("Normality Test (QQ Plot)")
    numerical_columns = ['Employment_demanded', 'Employment_offered', 'Employment_Availed', 
                         'Area_(in_Ha)', 'Production_(in_Tonnes)', 'Yield_(kg/Ha)', 
                         'Annual_rainfall', 'MSP']

    selected_column = st.selectbox("Select a column for QQ Plot", numerical_columns)
    qq_data = state_data[selected_column].dropna()

    # Generate QQ plot
    fig, ax = plt.subplots()
    stats.probplot(qq_data, dist="norm", plot=ax)
    st.pyplot(fig)

    # Spearman correlation test
    st.subheader("Spearman Correlation Test")
    selected_corr_columns = st.multiselect("Select columns for correlation", numerical_columns)
    if len(selected_corr_columns) > 1:
        corr_matrix = state_data[selected_corr_columns].corr(method='spearman')
        st.write(corr_matrix)

        # Heatmap of correlation
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Line plot for MGNREGA demand across years
    st.subheader("MGNREGA Demand Over Years")
    if 'year' in state_data.columns and 'Employment_demanded' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(x='year', y='Employment_demanded', data=state_data, ax=ax)
        plt.title(f"MGNREGA Employment Demand Over Years in {selected_state}")
        st.pyplot(fig)

        # Predict future demand for 2024 and 2025
        future_years, future_predictions = predict_future(state_data, 'Employment_demanded', [2024, 2025])
        st.write(f"Predicted Employment Demand for 2024: {future_predictions[0]:.2f}")
        st.write(f"Predicted Employment Demand for 2025: {future_predictions[1]:.2f}")

    # Production of the state each year
    st.subheader("Crop Production Over Years")
    if 'year' in state_data.columns and 'Production_(in_Tonnes)' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(x='year', y='Production_(in_Tonnes)', data=state_data, ax=ax)
        plt.title(f"Crop Production Over Years in {selected_state}")
        st.pyplot(fig)

        # Predict future crop production for 2024 and 2025
        future_years, future_predictions = predict_future(state_data, 'Production_(in_Tonnes)', [2024, 2025])
        st.write(f"Predicted Crop Production for 2024: {future_predictions[0]:.2f}")
        st.write(f"Predicted Crop Production for 2025: {future_predictions[1]:.2f}")

    # Rainfall of the state each year
    st.subheader("Annual Rainfall Over Years")
    if 'year' in state_data.columns and 'Annual_rainfall' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(x='year', y='Annual_rainfall', data=state_data, ax=ax)
        plt.title(f"Annual Rainfall Over Years in {selected_state}")
        st.pyplot(fig)

        # Predict future rainfall for 2024 and 2025
        future_years, future_predictions = predict_future(state_data, 'Annual_rainfall', [2024, 2025])
        st.write(f"Predicted Rainfall for 2024: {future_predictions[0]:.2f} mm")
        st.write(f"Predicted Rainfall for 2025: {future_predictions[1]:.2f} mm")

    # Adjusted MSP
    st.subheader("Adjusted MSP Over Years")
    if 'year' in state_data.columns and 'MSP' in state_data.columns:
        fig, ax = plt.subplots()
        sns.lineplot(x='year', y='MSP', data=state_data, ax=ax)
        plt.title(f"Adjusted MSP Over Years in {selected_state}")
        st.pyplot(fig)

        # Predict future MSP for 2024 and 2025
        future_years, future_predictions = predict_future(state_data, 'MSP', [2024, 2025])
        st.write(f"Predicted MSP for 2024: {future_predictions[0]:.2f} INR")
        st.write(f"Predicted MSP for 2025: {future_predictions[1]:.2f} INR")

if __name__ == "__main__":
    main()
