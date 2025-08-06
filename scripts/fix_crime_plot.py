#!/usr/bin/env python3
"""
Fix the crime plot function in the main notebook
"""

import json
import re
import os

def fix_crime_plot():
    """Fix the crime plot function to work in static Jupyter Book builds"""
    
    notebook_path = "_sources/notebooks/main.ipynb"
    
    print("🔧 Fixing crime plot in main notebook...")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the crime plot function
    crime_plot_code = '''def create_interactive_crime_plot(file_path):
    """
    Create an interactive scatter plot with dropdown menus for selecting metrics
    Also generates static plots and HTML export for Jupyter Book compatibility
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing the dataset
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name='2023_dataset')
    
    # Get list of numeric columns for dropdowns
    exclude_cols = ['Country', 'Continent', 'Region']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create static plot as fallback (always visible)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Default metrics for static plot
    x_metric = numeric_cols[0] if numeric_cols else 'Index'
    y_metric = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
    
    # Create scatter plot
    scatter = ax.scatter(df[x_metric], df[y_metric], 
                        alpha=0.7, s=60, c='darkred', edgecolors='white', linewidth=0.5)
    
    # Add country labels for extreme values
    for i, row in df.iterrows():
        if (row[x_metric] > df[x_metric].quantile(0.9) or 
            row[x_metric] < df[x_metric].quantile(0.1) or
            row[y_metric] > df[y_metric].quantile(0.9) or 
            row[y_metric] < df[y_metric].quantile(0.1)):
            ax.annotate(row['Country'], (row[x_metric], row[y_metric]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
    
    ax.set_xlabel(f'{x_metric}', fontsize=12)
    ax.set_ylabel(f'{y_metric}', fontsize=12)
    ax.set_title(f'Global Crime Index: {x_metric} vs {y_metric}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create interactive plotly version
    try:
        # Create interactive plotly scatter plot
        fig_interactive = px.scatter(df, x=x_metric, y=y_metric, 
                                   hover_data=['Country', 'Continent', 'Region'],
                                   title=f'Interactive Global Crime Index: {x_metric} vs {y_metric}',
                                   labels={x_metric: x_metric, y_metric: y_metric},
                                   color='Region')
        
        fig_interactive.update_traces(marker=dict(size=8, opacity=0.7))
        fig_interactive.update_layout(height=600)
        
        # Save interactive plot for Jupyter Book
        html_filename = 'interactive_crime_plot.html'
        
        # Save to current directory for notebook development
        fig_interactive.write_html(html_filename)
        print(f"💾 Saved interactive crime plot: {html_filename}")
        
        # Also try to save to build directory if it exists
        try:
            from pathlib import Path
            build_dir = Path('_build/html')
            if build_dir.exists():
                build_path = build_dir / html_filename
                fig_interactive.write_html(str(build_path))
                print(f"💾 Also saved to: {build_path}")
        except:
            pass
            
        # Display interactive plot if in notebook environment
        try:
            fig_interactive.show()
        except:
            print("📊 Interactive crime plot saved as HTML file")
            
    except ImportError:
        print("⚠️  Plotly not available, showing static plot only")
    except Exception as e:
        print(f"⚠️  Error creating interactive crime plot: {e}")
    
    # Create widget-based interactive version (for local notebook use)
    try:
        from IPython.display import display, clear_output
        import ipywidgets as widgets
        
        # Create dropdown widgets
        x_dropdown = widgets.Dropdown(
            options=numeric_cols,
            value=x_metric,
            description='X-axis:',
            style={'description_width': 'initial'},
            layout={'width': 'auto'}
        )
        
        y_dropdown = widgets.Dropdown(
            options=numeric_cols,
            value=y_metric,
            description='Y-axis:',
            style={'description_width': 'initial'},
            layout={'width': 'auto'}
        )
        
        def update_plot():
            clear_output(wait=True)
            
            x_metric_local = x_dropdown.value
            y_metric_local = y_dropdown.value
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(df[x_metric_local], df[y_metric_local], 
                               alpha=0.7, s=60, c='darkred', edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel(f'{x_metric_local}')
            ax.set_ylabel(f'{y_metric_local}')
            ax.set_title(f'Global Crime Index: {x_metric_local} vs {y_metric_local}')
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
    
    # Find the cell containing the crime plot function
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def create_interactive_crime_plot(' in source:
                print(f"📝 Updating crime plot function in cell {i}")
                cell['source'] = crime_plot_code.split('\n')
                break
    
    # Add a markdown cell with link to interactive plot
    crime_link_markdown = '''```{note}
**Interactive Crime Plot Available**

The interactive version allows you to explore the Global Crime Index data with hover details and regional coloring.

<a href="interactive_crime_plot.html" target="_blank" style="display: inline-block; padding: 8px 16px; background-color: #dc3545; color: white; text-decoration: none; border-radius: 4px; margin: 4px 0;">🚨 Open Interactive Crime Plot</a>
```'''
    
    # Find where to insert the link (after the function call)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'create_interactive_crime_plot(file_path)' in source:
                # Insert markdown cell after this code cell
                new_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": crime_link_markdown.split('\n')
                }
                notebook['cells'].insert(i + 1, new_cell)
                print(f"📝 Added interactive crime plot link after cell {i}")
                break
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Crime plot fixed successfully!")
    return True

if __name__ == "__main__":
    fix_crime_plot()