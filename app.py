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

# Function to calculate year-over-year percentage change
def calculate_year_over_year_change(series):
    return series.pct_change() * 100

# Function to analyze MSP vs WPI trends for each crop in each state
def analyze_msp_vs_wpi_trends(df):
    results = []
    
    # Group by State, Crop, and Year
    grouped_data = df.groupby(['State_y', 'Crop'])
    
    for (state, crop), group in grouped_data:
        group = group.sort_values('year')  # Ensure the data is sorted by year
        
        # Calculate year-over-year percentage change for MSP and WPI
        group['MSP_Change'] = calculate_year_over_year_change(group['MSP'])
        group['WPI_Change'] = calculate_year_over_year_change(group['WPI'])
        
        # Analyze the trend by comparing MSP and WPI percentage changes
        for _, row in group.iterrows():
            msp_change = row['MSP_Change']
            wpi_change = row['WPI_Change']
            
            if pd.notna(msp_change) and pd.notna(wpi_change):
                if msp_change == 0 and wpi_change == 0:
                    continue  # Skip if both changes are zero
                
                if msp_change >= wpi_change:
                    status = 'MSP keeping up with WPI'
                else:
                    status = 'MSP lagging behind WPI'
                
                result = {
                    'State': state,
                    'Crop': crop,
                    'Year': row['year'],
                    'MSP_Change (%)': msp_change,
                    'WPI_Change (%)': wpi_change,
                    'Status': status
                }
                results.append(result)
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

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

# Section for MSP vs WPI trend analysis
def display_msp_vs_wpi_trends(data):
    st.subheader("MSP vs WPI Trends by State and Crop")
    
    # Analyze MSP vs WPI trends
    msp_wpi_trend_analysis_df = analyze_msp_vs_wpi_trends(data)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Create a FacetGrid to generate separate plots for each state
    g = sns.FacetGrid(msp_wpi_trend_analysis_df, col="State", hue="State", col_wrap=4, height=5, sharey=False)
    
    # Plot MSP and WPI changes
    g.map(sns.lineplot, 'Year', 'MSP_Change (%)', marker='o', linestyle='-', label='MSP Change')
    g.map(sns.lineplot, 'Year', 'WPI_Change (%)', marker='s', linestyle='--', label='WPI Change')
    
    # Add titles and labels
    g.set_axis_labels('Year', 'Percentage Change (%)')
    g.set_titles(col_template="{col_name}")
    g.add_legend(title='Metric')
    g.fig.suptitle('Year-over-Year Percentage Change in MSP and WPI by State', fontsize=16)
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.9)  # Adjust the title position
    
    # Show the plot
    st.pyplot(g.fig)

# Main function to render the Streamlit app
def main():
    st.title('MGNREGA and Crop Analysis by State')

    # Load the dataset
    data = load_data()

    # Sidebar for state selection
    st.sidebar.header('State Selection')
    states = data['State_x'].unique()
    selected_state = st.sidebar.selectbox('Select a state', states)

    # Create sections for exploration, visualization, and prediction
    st.sidebar.markdown("### Sections")
    sections = ['Data Overview', 'Visualizations', 'Predictions for 2024 and 2025', 'MSP vs WPI Trends']
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
    
    # MSP vs WPI analysis section
    elif selected_section == 'MSP vs WPI Analysis':
        msp_wpi_trend_analysis_df = analyze_msp_vs_wpi_trends(data)
        visualize_msp_wpi_trends(msp_wpi_trend_analysis_df)

if __name__ == "__main__":
    main()
