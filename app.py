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

# Visualize MSP vs WPI trends
def visualize_msp_wpi_trends(df):
    st.subheader('MSP vs. WPI Trends')
    
    # Set plot style
    sns.set(style="whitegrid")

    # Create a FacetGrid to generate separate plots for each state
    g = sns.FacetGrid(df, col="State", hue="State", col_wrap=4, height=5, sharey=False)

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

    # Sidebar for state selection
    st.sidebar.header('State Selection')
    states = data['State_x'].unique()
    selected_state = st.sidebar.selectbox('Select a state', states)

    # Create sections for exploration, visualization, and MSP vs WPI analysis
    st.sidebar.markdown("### Sections")
    sections = ['Data Overview', 'Visualizations', 'Predictions for 2024 and 2025', 'MSP vs WPI Analysis']
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
