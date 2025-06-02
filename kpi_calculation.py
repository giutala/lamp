import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr


def load_and_clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Rename columns to remove newline characters
    df.columns = [col.replace("\n", " ") for col in df.columns]

    return df


def plot_histogram(df):
    # Plot a histogram of the most seen values
    df.hist(figsize=(20, 15), bins=20)
    plt.tight_layout()
    plt.show()


def calculate_kpis(df):
    monthly_kpis = []

    # Iterate over each month and each equipaggio
    for (month, equipaggio), df_group in df.groupby(['Mese', 'Id Equipaggio']):
        # Ensure there are valid delta values
        delta_values = df_group['Delta (orario di fine montaggio EFFETTIVO - orario di fine montaggio CONCORDATO) [min]'].dropna()

        if delta_values.empty:
            continue  # Skip if there are no valid values

        # Handle outliers in Delta
        q1, q3 = np.percentile(delta_values, [25, 75])
        iqr_value = iqr(delta_values)
        lower_bound = q1 - 1.5 * iqr_value
        upper_bound = q3 + 1.5 * iqr_value

        # Winsorize: Cap extreme values
        filtered_delta = delta_values.clip(
            lower=lower_bound, upper=upper_bound)

        # Calculate Absolute Median Delta
        delta_median = abs(filtered_delta.median())

        # Calculate Rescheduling Impact
        df_group['Total Reschedules'] = (
            df_group['# consegne ripianificate per problemi nel MONTAGGIO di "TIPO 1" (camere, armadio e camerette)'] +
            df_group['# consegne ripianificate per problemi nel MONTAGGIO di "TIPO 2" (cucine e soggiorni)'] +
            df_group['# consegne ripianificate per problemi nel MONTAGGIO  di "TIPO 3" (arredo bagno, tavoli, sedie)']
        )

        df_group['Total Reschedules_'] = (
            df_group['# consegne ripianificate per problemi nel TRASPORTO'] +
            df_group['# consegne ripianificate per problemi nel MONTAGGIO di "TIPO 1" (camere, armadio e camerette)'] +
            df_group['# consegne ripianificate per problemi nel MONTAGGIO di "TIPO 2" (cucine e soggiorni)'] +
            df_group['# consegne ripianificate per problemi nel MONTAGGIO  di "TIPO 3" (arredo bagno, tavoli, sedie)'] +
            df_group['# consegne ripianificate per CAUSE LATO CLIENTE FINALE (assenza) e AZIENDA COMMITTENTE (es. pezzi mancanti)']
        )

        total_served = df_group['# clienti  serviti nel mese'].sum()
        total_reschedules = df_group['Total Reschedules'].sum()
        total_reschedules_ = df_group['Total Reschedules_'].sum()
        reschedule_ratio = total_reschedules / \
            (total_reschedules_ + total_served) if (total_reschedules +
                                                    total_served) > 0 else 0

        # Store base metrics for later normalization
        monthly_kpis.append({
            'Mese': month,
            'Id Equipaggio': equipaggio,
            'Delta Median': delta_median,
            'Reschedule Ratio': reschedule_ratio
        })

    # Convert to DataFrame for normalization
    kpi_df = pd.DataFrame(monthly_kpis).sort_values(
        by=['Mese', 'Id Equipaggio'])

    return kpi_df


def normalize_kpis(kpi_df):
    # Calculate normalization parameters
    delta_min = kpi_df['Delta Median'].min()
    delta_max = kpi_df['Delta Median'].max()
    resched_min = kpi_df['Reschedule Ratio'].min()
    resched_max = kpi_df['Reschedule Ratio'].max()

    # Define weights (adjust based on business priorities)
    W1 = 10  # Weight for Delta Median
    W2 = 20  # Weight for Reschedule Ratio

    # Normalize metrics and calculate composite score
    kpi_df['Norm Delta'] = kpi_df['Delta Median'].apply(
        lambda x: (delta_max - x) / (delta_max -
                                     delta_min) if delta_max != delta_min else 1.0
    )
    kpi_df['Norm Reschedule'] = kpi_df['Reschedule Ratio'].apply(
        lambda x: (resched_max - x) / (resched_max -
                                       resched_min) if resched_max != resched_min else 1.0
    )

    # Calculate weighted composite score
    kpi_df['Composite'] = (W1 * kpi_df['Norm Delta'] +
                           W2 * kpi_df['Norm Reschedule']) / (W1 + W2)

    # Normalize composite score to 0-100 KPI
    composite_min = kpi_df['Composite'].min()
    composite_max = kpi_df['Composite'].max()

    if composite_max != composite_min:
        kpi_df['KPI'] = 100 * (kpi_df['Composite'] -
                               composite_min) / (composite_max - composite_min)
    else:
        kpi_df['KPI'] = 100  # All scores are identical

    # Ensure KPI is within bounds
    kpi_df['KPI'] = kpi_df['KPI'].clip(0, 100).round(2)

    return kpi_df


def plot_kpi(kpi_df):
    # Plot Monthly KPI Values per Equipaggio
    plt.figure(figsize=(12, 6))
    for equipaggio in kpi_df['Id Equipaggio'].unique()[:5]:
        subset = kpi_df[kpi_df['Id Equipaggio'] == equipaggio]
        plt.plot(subset['Mese'], subset['KPI'], marker='o',
                 linestyle='-', label=f'Equipaggio {equipaggio}')

    plt.ylim(0, 100)
    plt.xlabel("Month")
    plt.ylabel("Efficiency Score")
    plt.title("Assembly Efficiency KPI by Month and Equipaggio")
    plt.legend()
    plt.grid()
    plt.show()


def plot_high_kpi_count(kpi_df):
    # Calculate high KPI count per month
    high_kpi_count_per_month = kpi_df[kpi_df['KPI'] > 90].groupby('Mese')[
        'Id Equipaggio'].nunique()

    # Plot the results
    plt.figure(figsize=(10, 6))
    high_kpi_count_per_month.plot(
        kind='bar', color='skyblue', edgecolor='black')
    plt.title('Number of Id Equipaggio with KPI > 90 per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Id Equipaggio')
    plt.xticks(range(12), [f'Month {i}' for i in range(1, 13)], rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # Load and clean the data
    df = load_and_clean_data('dataset.csv')

    # Display dataset information
    print(df.info())
    print(df.describe())

    # Plot histogram
    plot_histogram(df)

    # Calculate KPIs
    kpi_df = calculate_kpis(df)

    # Normalize KPIs
    kpi_df = normalize_kpis(kpi_df)

    # Plot KPI values
    plot_kpi(kpi_df)

    # Print KPI results
    print(kpi_df[['Mese', 'Id Equipaggio', 'KPI',
          'Delta Median', 'Reschedule Ratio']])

    # Plot high KPI count per month
    plot_high_kpi_count(kpi_df)


if __name__ == "__main__":
    main()
