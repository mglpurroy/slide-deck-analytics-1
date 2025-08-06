"""
FCV (Fragility, Conflict, and Violence) Style Guide
Provides consistent styling for data visualizations and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative, sequential, diverging

# FCV Color Palette
FCV_COLORS = {
    'primary': '#2E86AB',      # Blue - stability, peace
    'secondary': '#A23B72',    # Purple - complexity
    'accent': '#F18F01',       # Orange - attention, warning
    'danger': '#C73E1D',       # Red - conflict, high risk
    'success': '#4A7C59',      # Green - positive outcomes
    'warning': '#F4A261',      # Amber - moderate risk
    'info': '#264653',         # Dark teal - information
    'light': '#E9C46A',        # Light yellow - background
    'dark': '#1D3557',         # Navy - text
    'gray': '#6C757D'          # Gray - neutral
}

# Risk level colors
RISK_COLORS = {
    'very_low': '#4A7C59',     # Green
    'low': '#A8DADC',          # Light blue
    'moderate': '#F4A261',     # Amber
    'high': '#E76F51',         # Orange-red
    'very_high': '#C73E1D',    # Red
    'extreme': '#8B0000'       # Dark red
}

# Fragility state colors
FRAGILITY_COLORS = {
    'sustainable': '#4A7C59',
    'stable': '#A8DADC', 
    'warning': '#F4A261',
    'alert': '#E76F51',
    'crisis': '#C73E1D'
}

def apply_fcv_style():
    """Apply FCV styling to matplotlib and seaborn"""
    
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configure matplotlib parameters
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.edgecolor': FCV_COLORS['gray'],
        'axes.facecolor': 'white'
    })
    
    # Set seaborn palette
    sns.set_palette([
        FCV_COLORS['primary'],
        FCV_COLORS['secondary'], 
        FCV_COLORS['accent'],
        FCV_COLORS['success'],
        FCV_COLORS['warning'],
        FCV_COLORS['danger']
    ])

def get_plotly_theme():
    """Get Plotly theme configuration for FCV styling"""
    return {
        'layout': {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'title': {'font': {'size': 16}},
            'colorway': [
                FCV_COLORS['primary'],
                FCV_COLORS['secondary'],
                FCV_COLORS['accent'], 
                FCV_COLORS['success'],
                FCV_COLORS['warning'],
                FCV_COLORS['danger']
            ],
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'gridcolor': '#E5E5E5'
        }
    }

def get_risk_colorscale():
    """Get risk-based color scale for choropleth maps"""
    return [
        [0.0, RISK_COLORS['very_low']],
        [0.2, RISK_COLORS['low']], 
        [0.4, RISK_COLORS['moderate']],
        [0.6, RISK_COLORS['high']],
        [0.8, RISK_COLORS['very_high']],
        [1.0, RISK_COLORS['extreme']]
    ]

def get_fragility_colorscale():
    """Get fragility-based color scale"""
    return [
        [0.0, FRAGILITY_COLORS['sustainable']],
        [0.25, FRAGILITY_COLORS['stable']],
        [0.5, FRAGILITY_COLORS['warning']],
        [0.75, FRAGILITY_COLORS['alert']],
        [1.0, FRAGILITY_COLORS['crisis']]
    ]

def style_plotly_figure(fig, title=None, height=600):
    """Apply consistent styling to a Plotly figure"""
    theme = get_plotly_theme()
    
    fig.update_layout(
        **theme['layout'],
        height=height,
        title=title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Automatically apply styling when module is imported
apply_fcv_style()

print("✅ FCV styling applied successfully")
print(f"📊 Primary color: {FCV_COLORS['primary']}")
print(f"🎨 Available colors: {len(FCV_COLORS)} defined")
print(f"⚠️  Risk levels: {len(RISK_COLORS)} defined")