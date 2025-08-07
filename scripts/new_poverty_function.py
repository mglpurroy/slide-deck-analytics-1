def plot_poverty_trends_enhanced(data_dir: str, output_path: str = None) -> None:
    """
    Create a stacked bar chart showing poverty trends for FCS and other economies,
    replicating the format from the reference figure with $3.00/day poverty line.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Read the data
    data_path = os.path.join(data_dir, 'poverty_data.csv')
    df = pd.read_csv(data_path)
    
    # Filter for $3.0 poverty line and years >= 2005
    df = df[(df['povertyline'] == 3.0)].copy()
    
    # Use fcvFY25 variable directly (assuming it's already in the data)
    # fcvFY25 should be 1 for FCV countries, 0 for others
    df['fcs_mth2020'] = df['fcvFY25']
    
    # Calculate poor population (headcount should already be /100, population in millions)
    df['poor_tot'] = df['poorpop']
    
    # Aggregate by FCS status and year
    agg_df = df.groupby(['fcs_mth2020', 'year'])['poor_tot'].sum().reset_index()
    
    # Apply 3-year moving average within each FCS group
    def apply_moving_average(group):
        group = group.sort_values('year')
        group['poor_tot_ma3'] = group['poor_tot'].rolling(window=3, center=True, min_periods=1).mean()
        group['poor_tot_ma3'] = group['poor_tot_ma3'].fillna(group['poor_tot'])
        return group
    
    agg_df = agg_df.groupby('fcs_mth2020').apply(apply_moving_average).reset_index(drop=True)
    
    # Use the moving average values
    agg_df['poor_tot'] = agg_df['poor_tot_ma3']
    
    # Reshape data
    plot_df = agg_df.pivot(index='year', columns='fcs_mth2020', values='poor_tot')
    
    # Rename columns to match the reference figure
    plot_df.columns = ['Non-FCS (FY25)', 'FCS (FY25)']  # 0 = non-FCS, 1 = FCS
    
    # Fill NaN values with 0 for proper stacking
    plot_df = plot_df.fillna(0)
    
    # Create figure with white background
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Define colors matching the reference figure
    colors = ['#A8A8A8', '#4472C4']  # Gray for Non-FCS, Blue for FCS
    
    # Create stacked bar chart
    years = plot_df.index
    non_fcs_values = plot_df['Non-FCS (FY25)']
    fcs_values = plot_df['FCS (FY25)']
    
    # Create bars
    bar_width = 0.8
    bars1 = ax.bar(years, non_fcs_values, bar_width, 
                   label='Non-FCS (FY25)', color=colors[0], alpha=0.8)
    bars2 = ax.bar(years, fcs_values, bar_width, bottom=non_fcs_values,
                   label='FCS (FY25)', color=colors[1], alpha=0.8)
    
    # Add horizontal grid lines
    ax.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax.set_axisbelow(True)
    
    # Customize axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    # Set y-axis limits and ticks to match reference figure
    max_value = (plot_df.sum(axis=1)).max()
    y_max = int(np.ceil(max_value / 200) * 200)  # Round up to nearest 200
    ax.set_ylim(0, y_max)
    
    # Create y-axis ticks every 200 units
    y_ticks = list(range(0, y_max + 1, 200))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(tick):,}' for tick in y_ticks])
    
    # Set x-axis
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
    
    # Add title and subtitle
    ax.set_title('No. of extreme poor (below PPP$3.00 per day)\\nin FCS and non-FCS countries (millions)', 
                fontsize=16, fontweight='bold', pad=20, loc='center')
    
    # Add y-axis label (rotated to match reference)
    ax.set_ylabel('')  # Remove default ylabel since title includes the units
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add legend matching reference figure position
    legend = ax.legend(bbox_to_anchor=(0.85, 0.85), loc='upper left',
                      frameon=True, fancybox=False, shadow=False,
                      fontsize=12, ncol=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    
    # Add value labels on key years (similar to reference figure)
    forecast_start_year = 2025
    key_years = [2025, 2030, 2040]  # Years to show labels for
    
    for year in key_years:
        if year in plot_df.index:
            total_value = plot_df.loc[year].sum()
            fcs_value = plot_df.loc[year, 'FCS (FY25)']
            non_fcs_value = plot_df.loc[year, 'Non-FCS (FY25)']
            
            # Add total label at the top
            ax.text(year, total_value + y_max * 0.02, f'{int(total_value)}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='black', linewidth=0.5))
            
            # Add FCS value label in the middle of the FCS bar
            if fcs_value > 0:
                fcs_y_pos = non_fcs_value + fcs_value / 2
                ax.text(year, fcs_y_pos, f'{int(fcs_value)}', 
                       ha='center', va='center', fontweight='bold', fontsize=10,
                       color='white')
    
    # Add forecast arrow and label (matching reference figure)
    if forecast_start_year in years:
        forecast_years = years[years >= forecast_start_year]
        if len(forecast_years) > 0:
            # Add forecast arrow
            arrow_y = y_max * 0.1
            arrow_start = forecast_start_year - 0.5
            arrow_end = years.max() + 0.3
            
            ax.annotate('Forecast', xy=(arrow_end, arrow_y), xytext=(arrow_start, arrow_y),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                       fontsize=12, color='blue', ha='left', va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()
    
    plt.close()

# Usage example:
# plot_poverty_trends_enhanced(data_dir)