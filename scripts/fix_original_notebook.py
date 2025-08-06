#!/usr/bin/env python3
"""
Fix interactive plots in the original notebook (docs/notebooks/main.ipynb)
"""

import json
import re
import os

def fix_original_notebook():
    """Fix the interactive plot functions in the original notebook"""
    
    notebook_path = "docs/notebooks/main.ipynb"
    
    print("🔧 Fixing interactive plots in original notebook...")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # First, fix the data path issue
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'root_dir = os.path.abspath(os.path.join(os.getcwd(), \'..\'))' in source:
                print(f"📝 Fixing data paths in cell {i}")
                # Replace the path logic
                new_source = source.replace(
                    "root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))",
                    """# Handle both original location (docs/notebooks) and Jupyter Book location
# For docs/notebooks, go up one level to docs, then up one more to root
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))"""
                )
                new_source = new_source.replace(
                    "data_dir = os.path.join(root_dir, 'data')",
                    """data_dir = os.path.join(root_dir, 'data')
print(f"📁 Data directory: {data_dir}")
print(f"📂 Root directory: {root_dir}")
print(f"✓ Current working directory: {os.getcwd()}")"""
                )
                cell['source'] = new_source.split('\n')
                break
    
    # Find and fix the fragility plot function
    fragility_plot_code = '''def create_interactive_fragility_plot(file_path):
    """
    Create an interactive scatter plot with dropdown menus for selecting fragility metrics
    Also generates static plots and HTML export for Jupyter Book compatibility
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the States of Fragility 2022 dataset
    """
    # Import plotly at the function level
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        plotly_available = True
    except ImportError:
        plotly_available = False
        print("⚠️  Plotly not available, will show static plot only")
    
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name='Scores')
    
    # Clean column names by removing 'PC1'
    df.columns = [col.replace('.PC1', '') for col in df.columns]
    
    # Get list of numeric columns for dropdowns
    exclude_cols = ['iso3c', 'country', 'type']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create static plot as fallback (always visible)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Default metrics for static plot
    x_metric = 'Aggregate'
    y_metric = 'Political'
    
    # Create scatter plot
    scatter = ax.scatter(df[x_metric], df[y_metric], 
                        alpha=0.7, s=60, c='steelblue', edgecolors='white', linewidth=0.5)
    
    # Add country labels for extreme values
    for i, row in df.iterrows():
        if (row[x_metric] > df[x_metric].quantile(0.9) or 
            row[x_metric] < df[x_metric].quantile(0.1) or
            row[y_metric] > df[y_metric].quantile(0.9) or 
            row[y_metric] < df[y_metric].quantile(0.1)):
            ax.annotate(row['country'], (row[x_metric], row[y_metric]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
    
    ax.set_xlabel(f'{x_metric} Score', fontsize=12)
    ax.set_ylabel(f'{y_metric} Score', fontsize=12)
    ax.set_title(f'States of Fragility 2022: {x_metric} vs {y_metric}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create interactive plotly version
    if plotly_available:
        try:
            # Create interactive plotly scatter plot
            fig_interactive = px.scatter(df, x=x_metric, y=y_metric, 
                                       hover_data=['country', 'type'],
                                       title=f'Interactive States of Fragility 2022: {x_metric} vs {y_metric}',
                                       labels={x_metric: f'{x_metric} Score', 
                                              y_metric: f'{y_metric} Score'})
            
            fig_interactive.update_traces(marker=dict(size=8, opacity=0.7))
            fig_interactive.update_layout(height=600, showlegend=False)
            
            # Save interactive plot for Jupyter Book
            html_filename = 'interactive_fragility_plot.html'
            
            # Save to current directory for notebook development
            fig_interactive.write_html(html_filename)
            print(f"💾 Saved interactive plot: {html_filename}")
            
            # Display interactive plot if in notebook environment
            try:
                fig_interactive.show()
            except:
                print("📊 Interactive plot saved as HTML file")
                
        except Exception as e:
            print(f"⚠️  Error creating interactive plot: {e}")
    
    # Create widget-based interactive version (for local notebook use)
    try:
        from IPython.display import display, clear_output
        import ipywidgets as widgets
        
        # Create dropdown widgets
        x_dropdown = widgets.Dropdown(
            options=numeric_cols,
            value='Aggregate',  # Default value
            description='X-axis:',
            style={'description_width': 'initial'},
            layout={'width': 'auto'}
        )
        
        y_dropdown = widgets.Dropdown(
            options=numeric_cols,
            value='Political',  # Default value
            description='Y-axis:',
            style={'description_width': 'initial'},
            layout={'width': 'auto'}
        )
        
        def update_plot():
            clear_output(wait=True)
            
            x_metric = x_dropdown.value
            y_metric = y_dropdown.value
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(df[x_metric], df[y_metric], 
                               alpha=0.7, s=60, c='steelblue', edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel(f'{x_metric} Score')
            ax.set_ylabel(f'{y_metric} Score')
            ax.set_title(f'States of Fragility 2022: {x_metric} vs {y_metric}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Attach the update function to dropdown changes
        x_dropdown.observe(lambda change: update_plot(), names='value')
        y_dropdown.observe(lambda change: update_plot(), names='value')
        
        # Display widgets (only works in interactive notebook environment)
        try:
            display(widgets.HBox([x_dropdown, y_dropdown]))
            print("🎛️ Use the dropdowns above to explore different metrics")
        except:
            print("🎛️ Interactive widgets not available in this environment")
            
    except ImportError:
        print("⚠️ IPython widgets not available")'''
    
    # Find the cell containing the fragility plot function
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def create_interactive_fragility_plot(' in source:
                print(f"📝 Updating fragility plot function in cell {i}")
                cell['source'] = fragility_plot_code.split('\n')
                break
    
    # Add a markdown cell with link to interactive plot after the function call
    fragility_link_markdown = '''```{note}
**Interactive Fragility Plot Available**

The interactive version allows you to explore the States of Fragility 2022 data with hover details and zooming capabilities.

<a href="interactive_fragility_plot.html" target="_blank" style="display: inline-block; padding: 8px 16px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; margin: 4px 0;">🗺️ Open Interactive Fragility Plot</a>
```'''
    
    # Find where to insert the link (after the function call)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'create_interactive_fragility_plot(file_path)' in source:
                # Insert markdown cell after this code cell
                new_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": fragility_link_markdown.split('\n')
                }
                notebook['cells'].insert(i + 1, new_cell)
                print(f"📝 Added interactive plot link after cell {i}")
                break
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Original notebook fixed successfully!")
    return True

if __name__ == "__main__":
    fix_original_notebook()