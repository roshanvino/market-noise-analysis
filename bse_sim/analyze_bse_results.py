import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from matplotlib.ticker import MaxNLocator

# Set plot style
plt.style.use('ggplot')
sns.set_context("talk")

# Configuration
RESULTS_DIR = 'RESULTS_BSE_SIM'
AGENT_TYPES = ['MA', 'MOM_QL', 'RNDM']  # Can be extended
NOISE_LEVELS = ['LOW_NOISE', 'MED_NOISE', 'HIGH_NOISE']  # Can be extended
OUTPUT_DIR = 'analysis_results'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csv_data(filepath):
    """
    Load and parse BSE avg_balance.csv file
    """
    try:
        # From the error and screenshot, it seems the CSV files have a fixed format
        # with 10 columns instead of the expected structure
        df = pd.read_csv(filepath, header=None)

        # Based on the screenshot, the CSV format appears to be:
        # session_id, time, price1, price2, agent_type, total_balance, num_traders, avg_balance, ...

        # Extract the time series data
        if len(df) == 0:
            print(f"Warning: Empty file: {filepath}")
            return None

        # Create a proper dataframe for time series analysis
        result_df = pd.DataFrame()
        result_df['time'] = df[1].astype(float)  # Time is in column 1

        # Extract the agent type from the filepath rather than trying to parse from CSV
        # This is more reliable as we know the agent type from the directory structure
        agent_type = os.path.basename(os.path.dirname(os.path.dirname(filepath)))

        # Get the total balance (column 5) for this agent type
        result_df[f'{agent_type}_balance'] = df[5].astype(float)

        # Get the number of traders (column 6)
        result_df[f'{agent_type}_count'] = df[6].astype(int)

        # Get the average balance (column 7)
        result_df[f'{agent_type}_avg_balance'] = df[7].astype(float)

        # Add session_id from filename
        result_df['session_id'] = os.path.basename(filepath)

        # Add best bid/ask if available (columns 2 and 3)
        if df.shape[1] > 3:
            result_df['best_bid'] = df[2]
            result_df['best_ask'] = df[3]

        return result_df

    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        # Print the first few rows of the file to debug
        try:
            with open(filepath, 'r') as f:
                print(f"First few lines of {filepath}:")
                for i, line in enumerate(f):
                    if i < 3:  # Print first 3 lines
                        print(line.strip())
                    else:
                        break
        except:
            pass
        return None


def extract_metrics(df, agent_type):
    """
    Extract key metrics from the dataframe for a specific agent type
    """
    if df is None or df.empty:
        return {
            'final_profit': np.nan,
            'mean_balance': np.nan,
            'std_dev': np.nan,
            'max_balance': np.nan,
            'min_balance': np.nan
        }

    # Find columns related to this agent type
    balance_col = f'{agent_type}_balance'
    count_col = f'{agent_type}_count'
    avg_balance_col = f'{agent_type}_avg_balance'

    # Check if the columns exist
    if balance_col not in df.columns:
        print(f"Warning: {balance_col} not found in dataframe")
        return {
            'final_profit': np.nan,
            'mean_balance': np.nan,
            'std_dev': np.nan,
            'max_balance': np.nan,
            'min_balance': np.nan
        }

    # Extract metrics
    # For final profit, use the last non-NaN value
    final_profit = df[balance_col].iloc[-1] if not pd.isna(df[balance_col].iloc[-1]) else df[balance_col].dropna().iloc[
        -1] if not df[balance_col].dropna().empty else np.nan

    # For mean balance, calculate the mean of the avg_balance column
    if avg_balance_col in df.columns:
        mean_balance = df[avg_balance_col].mean()
    else:
        mean_balance = df[balance_col].mean()

    # Calculate standard deviation, max and min
    std_dev = df[balance_col].std()
    max_balance = df[balance_col].max()
    min_balance = df[balance_col].min()

    return {
        'final_profit': final_profit,
        'mean_balance': mean_balance,
        'std_dev': std_dev,
        'max_balance': max_balance,
        'min_balance': min_balance
    }


def process_all_results():
    """
    Process all results and return a DataFrame with all metrics
    """
    results = []

    for agent_type in AGENT_TYPES:
        for noise_level in NOISE_LEVELS:
            folder_path = os.path.join(RESULTS_DIR, agent_type, noise_level)
            if not os.path.exists(folder_path):
                print(f"Warning: Path does not exist: {folder_path}")
                continue

            # Find all avg_balance.csv files
            csv_files = glob.glob(os.path.join(folder_path, "*_avg_balance.csv"))

            if not csv_files:
                print(f"Warning: No avg_balance.csv files found in {folder_path}")
                continue

            # Process each file
            for csv_file in csv_files:
                df = load_csv_data(csv_file)
                if df is not None:
                    metrics = extract_metrics(df, agent_type)

                    # Get session info
                    session_id = df['session_id'].iloc[0] if 'session_id' in df.columns else os.path.basename(csv_file)

                    # Add to results
                    results.append({
                        'agent_type': agent_type,
                        'noise_level': noise_level,
                        'session_id': session_id,
                        'final_profit': metrics['final_profit'],
                        'mean_balance': metrics['mean_balance'],
                        'std_dev': metrics['std_dev'],
                        'max_balance': metrics['max_balance'],
                        'min_balance': metrics['min_balance']
                    })

    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        print("No results found!")
        return pd.DataFrame()


def plot_balance_over_time():
    """
    Create line plots of average balance over time for each agent and noise level
    """
    fig, axes = plt.subplots(len(NOISE_LEVELS), 1, figsize=(12, 5 * len(NOISE_LEVELS)), sharex=True)

    if len(NOISE_LEVELS) == 1:
        axes = [axes]  # Wrap in list for consistent indexing

    for i, noise_level in enumerate(NOISE_LEVELS):
        ax = axes[i]

        for agent_type in AGENT_TYPES:
            folder_path = os.path.join(RESULTS_DIR, agent_type, noise_level)
            if not os.path.exists(folder_path):
                continue

            # Find all avg_balance.csv files
            csv_files = glob.glob(os.path.join(folder_path, "*_avg_balance.csv"))

            if not csv_files:
                continue

            # Use the first file found
            csv_file = csv_files[0]
            print(f"Plotting data from {csv_file}")
            df = load_csv_data(csv_file)

            if df is not None and not df.empty:
                # Find columns related to this agent type
                balance_col = f'{agent_type}_balance'

                if balance_col in df.columns:
                    # Sort by time to ensure correct plotting order
                    df = df.sort_values('time')

                    # Plot the balance over time
                    ax.plot(df['time'], df[balance_col], label=f"{agent_type}",
                            marker='o', markersize=3, alpha=0.7)

                    # Add a smooth trend line
                    try:
                        # Use a rolling average for smoothing if enough data points
                        if len(df) > 5:
                            window_size = min(5, len(df) // 5)
                            df['smooth'] = df[balance_col].rolling(window=window_size, center=True).mean()
                            ax.plot(df['time'], df['smooth'], label=f"{agent_type} (trend)",
                                    linestyle='--', linewidth=2)
                    except Exception as e:
                        print(f"Warning: Could not create trend line: {e}")

        ax.set_title(f'Balance Over Time - {noise_level}')
        ax.set_ylabel('Balance')
        ax.set_xlabel('Time')
        ax.legend()

        # Format x-axis to show fewer ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'balance_over_time.png'), dpi=300)
    plt.close()


def plot_final_profit_comparison(results_df):
    """
    Create bar charts comparing final profit by agent type and noise level
    """
    if results_df.empty:
        print("No data for final profit comparison")
        return

    # Ensure we have final_profit column
    if 'final_profit' not in results_df.columns:
        print("final_profit column not found in results DataFrame")
        return

    # Create pivot table for easier plotting
    pivot_df = results_df.pivot_table(
        values='final_profit',
        index='agent_type',
        columns='noise_level',
        aggfunc='mean'
    )

    # Create bar chart
    ax = pivot_df.plot(kind='bar', figsize=(12, 8))
    ax.set_title('Final Profit by Agent Type and Noise Level')
    ax.set_ylabel('Final Profit')
    ax.set_xlabel('Agent Type')
    ax.legend(title='Noise Level')

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)

    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_profit_comparison.png'), dpi=300)
    plt.close()


def plot_volatility_comparison(results_df):
    """
    Create box plots showing volatility (std_dev) across agents and noise levels
    """
    if results_df.empty:
        print("No data for volatility comparison")
        return

    # Ensure we have std_dev column
    if 'std_dev' not in results_df.columns:
        print("std_dev column not found in results DataFrame")
        return

    # Create a figure for the box plots
    plt.figure(figsize=(12, 8))

    # Create box plot
    sns.boxplot(x='agent_type', y='std_dev', hue='noise_level', data=results_df)

    plt.title('Profit Volatility by Agent Type and Noise Level')
    plt.ylabel('Standard Deviation (Volatility)')
    plt.xlabel('Agent Type')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'volatility_comparison.png'), dpi=300)
    plt.close()


def create_performance_summary_table(results_df):
    """
    Create a summary table of agent performance across noise levels
    """
    if results_df.empty:
        print("No data for performance summary")
        return

    # Create summary table
    summary = results_df.groupby(['agent_type', 'noise_level']).agg({
        'final_profit': ['mean', 'std'],
        'mean_balance': ['mean', 'std'],
        'std_dev': 'mean'
    }).reset_index()

    # Format column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    # Rename columns for clarity
    summary = summary.rename(columns={
        'agent_type_': 'Agent Type',
        'noise_level_': 'Noise Level',
        'final_profit_mean': 'Avg Final Profit',
        'final_profit_std': 'Profit Std Dev',
        'mean_balance_mean': 'Avg Mean Balance',
        'mean_balance_std': 'Mean Balance Std Dev',
        'std_dev_mean': 'Avg Volatility'
    })

    # Save to CSV
    summary.to_csv(os.path.join(OUTPUT_DIR, 'performance_summary.csv'), index=False)

    return summary


def plot_combined_performance(results_df):
    """
    Create a combined plot showing performance metrics across agents and noise levels
    """
    if results_df.empty:
        print("No data for combined performance plot")
        return

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Final Profit
    sns.barplot(x='agent_type', y='final_profit', hue='noise_level', data=results_df, ax=axes[0])
    axes[0].set_title('Final Profit')
    axes[0].set_ylabel('Profit')
    axes[0].set_xlabel('Agent Type')

    # Plot 2: Mean Balance
    sns.barplot(x='agent_type', y='mean_balance', hue='noise_level', data=results_df, ax=axes[1])
    axes[1].set_title('Mean Balance')
    axes[1].set_ylabel('Balance')
    axes[1].set_xlabel('Agent Type')

    # Plot 3: Standard Deviation
    sns.barplot(x='agent_type', y='std_dev', hue='noise_level', data=results_df, ax=axes[2])
    axes[2].set_title('Balance Volatility')
    axes[2].set_ylabel('Standard Deviation')
    axes[2].set_xlabel('Agent Type')

    # Add a common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(NOISE_LEVELS))

    # Remove individual legends
    for ax in axes:
        ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_performance.png'), dpi=300)
    plt.close()


def plot_noise_impact(results_df):
    """
    Create a plot showing how different noise levels affect each agent type
    """
    if results_df.empty:
        print("No data for noise impact plot")
        return

    # Create a figure
    plt.figure(figsize=(12, 8))

    # Plot lines connecting the noise levels for each agent
    sns.pointplot(x='noise_level', y='final_profit', hue='agent_type', data=results_df,
                  dodge=True, markers=['o', 's', 'D'], linestyles=['-', '--', '-.'])

    plt.title('Impact of Market Noise on Agent Performance')
    plt.ylabel('Final Profit')
    plt.xlabel('Noise Level')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'noise_impact.png'), dpi=300)
    plt.close()


def main():
    """
    Main function to run the analysis
    """
    print("Starting analysis of BSE simulation results...")

    # First check that we can find the required folders
    print(f"Looking for results in: {RESULTS_DIR}")
    if not os.path.exists(RESULTS_DIR):
        print(f"ERROR: Results directory {RESULTS_DIR} does not exist!")
        return

    # Check that we can find subdirectories for each agent type
    for agent_type in AGENT_TYPES:
        agent_dir = os.path.join(RESULTS_DIR, agent_type)
        if not os.path.exists(agent_dir):
            print(f"WARNING: Agent directory {agent_dir} does not exist!")
        else:
            # Check noise level directories
            for noise_level in NOISE_LEVELS:
                noise_dir = os.path.join(agent_dir, noise_level)
                if not os.path.exists(noise_dir):
                    print(f"WARNING: Noise level directory {noise_dir} does not exist!")
                else:
                    # Check for CSV files
                    csv_files = glob.glob(os.path.join(noise_dir, "*_avg_balance.csv"))
                    if not csv_files:
                        print(f"WARNING: No avg_balance.csv files found in {noise_dir}")
                    else:
                        print(f"Found {len(csv_files)} CSV files in {noise_dir}")

    # Process all results
    results_df = process_all_results()

    # Save results to CSV
    if not results_df.empty:
        # Display a sample of the results
        print("\nSample of processed results:")
        print(results_df.head())

        results_df.to_csv(os.path.join(OUTPUT_DIR, 'all_results.csv'), index=False)
        print(f"\nSaved all results to {os.path.join(OUTPUT_DIR, 'all_results.csv')}")

        # Create summary table
        summary = create_performance_summary_table(results_df)
        print(f"\nCreated performance summary table:")
        print(summary)

        # Create plots
        print("\nCreating plots...")
        plot_balance_over_time()
        plot_final_profit_comparison(results_df)
        plot_volatility_comparison(results_df)
        plot_combined_performance(results_df)
        plot_noise_impact(results_df)

        print(f"All plots saved to {OUTPUT_DIR}")
    else:
        print("\nNo results to analyze! Check the error messages above for details.")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()