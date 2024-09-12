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
  
# Section for predictions with visualization
def display_predictions_with_visualization(state_data, selected_state):
    st.subheader(f"Predictions for 2024 and 2025 in {selected_state}")
    
    # List of columns for which predictions will be made
    columns_to_predict = {
        'Employment_demanded': 'MGNREGA Employment Demand',
        'Production_(in_Tonnes)': 'Crop Production (in Tonnes)',
        'Annual_rainfall': 'Annual Rainfall (mm)',
        'MSP': 'Minimum Support Price (INR)'
    }

    for col, col_display_name in columns_to_predict.items():
        if col in state_data.columns:
            # Predict future values for 2024 and 2025
            future_years, future_predictions = predict_future(state_data, col)
            
            # Append predictions to the historical data
            historical_years = state_data['year'].values
            historical_values = state_data[col].values
            
            # Combine historical and predicted data
            combined_years = np.concatenate([historical_years, future_years])
            combined_values = np.concatenate([historical_values, future_predictions])

            # Create line plot for historical and predicted values
            fig, ax = plt.subplots()
            sns.lineplot(x=historical_years, y=historical_values, label='Historical', ax=ax)
            sns.lineplot(x=future_years, y=future_predictions, label='Predicted (2024, 2025)', ax=ax, linestyle="--", marker='o')
            plt.title(f"{col_display_name} Over Years in {selected_state}")
            plt.xlabel('Year')
            plt.ylabel(col_display_name)
            plt.legend()
            st.pyplot(fig)

            # Display the predicted values
            st.write(f"Predicted {col_display_name} for 2024: {future_predictions[0]:.2f}")
            st.write(f"Predicted {col_display_name} for 2025: {future_predictions[1]:.2f}")
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

    # Prediction section with visualizations
    elif selected_section == 'Predictions for 2024 and 2025':
        display_predictions_with_visualization(state_data, selected_state)

if __name__ == "__main__":
    main()

