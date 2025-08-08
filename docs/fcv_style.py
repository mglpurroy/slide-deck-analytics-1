"""
FCV (Fragility, Conflict, and Violence) Style Guide - World Bank Template
Provides consistent styling for data visualizations and analysis following World Bank FCV presentation template
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative, sequential, diverging

# World Bank FCV Color Palette (based on template)
FCV_COLORS = {
    'primary': '#1D3557',      # Dark navy blue (main title color)
    'secondary': '#2E86AB',    # World Bank blue (accent and links)
    'accent': '#00BFFF',       # Light blue (cyan for highlights)
    'danger': '#C73E1D',       # Red - conflict, high risk
    'success': '#4A7C59',      # Green - positive outcomes
    'warning': '#F4A261',      # Amber - moderate risk
    'info': '#264653',         # Dark teal - information
    'light': '#F8F9FA',        # Very light gray - background
    'medium_gray': '#A8A8A8',  # Medium gray (from template)
    'dark': '#1D3557',         # Navy - main text
    'gray': '#6C757D',         # Gray - neutral/secondary text
    'white': '#FFFFFF'         # Pure white
}

# Risk level colors (maintaining hierarchy but matching template palette)
RISK_COLORS = {
    'very_low': '#4A7C59',     # Green
    'low': '#2E86AB',          # World Bank blue
    'moderate': '#F4A261',     # Amber
    'high': '#E76F51',         # Orange-red
    'very_high': '#C73E1D',    # Red
    'extreme': '#8B0000'       # Dark red
}

# Fragility state colors
FRAGILITY_COLORS = {
    'sustainable': '#4A7C59',  # Green
    'stable': '#2E86AB',       # World Bank blue
    'warning': '#F4A261',      # Amber
    'alert': '#E76F51',        # Orange-red
    'crisis': '#C73E1D'        # Red
}

# Chart colors following World Bank template (clean, professional)
CHART_COLORS = {
    'primary_series': '#2E86AB',    # World Bank blue
    'secondary_series': '#A8A8A8',  # Medium gray (like in template)
    'tertiary_series': '#00BFFF',   # Light blue
    'quaternary_series': '#1D3557', # Dark navy
    'accent_series': '#F4A261'      # Amber accent
}

def apply_fcv_style():
    """Apply World Bank FCV styling to matplotlib and seaborn"""
    
    # Set matplotlib style to clean, minimal
    plt.style.use('default')
    
    # Configure matplotlib parameters to match World Bank template
    plt.rcParams.update({
        'figure.figsize': (16, 9),        # 16:9 ratio like presentation
        'figure.dpi': 300,                # High quality
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        
        # Fonts - clean, professional
        'font.size': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
        
        # Titles and labels
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.titlecolor': FCV_COLORS['dark'],
        'axes.labelsize': 14,
        'axes.labelweight': 'normal',
        'axes.labelcolor': FCV_COLORS['dark'],
        
        # Tick labels
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.color': FCV_COLORS['gray'],
        'ytick.color': FCV_COLORS['gray'],
        
        # Legend
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 1.0,
        'legend.facecolor': 'white',
        'legend.edgecolor': FCV_COLORS['gray'],
        
        # Spines and grid - minimal, clean
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1,
        'axes.edgecolor': FCV_COLORS['gray'],
        
        # Grid - subtle horizontal lines like in template
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.alpha': 0.3,
        'grid.color': FCV_COLORS['gray'],
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',
        
        # Colors
        'axes.prop_cycle': plt.cycler('color', [
            CHART_COLORS['primary_series'],
            CHART_COLORS['secondary_series'],
            CHART_COLORS['tertiary_series'],
            CHART_COLORS['quaternary_series'],
            CHART_COLORS['accent_series']
        ])
    })
    
    # Set seaborn palette to match
    sns.set_palette([
        CHART_COLORS['primary_series'],
        CHART_COLORS['secondary_series'], 
        CHART_COLORS['tertiary_series'],
        CHART_COLORS['quaternary_series'],
        CHART_COLORS['accent_series']
    ])

def get_plotly_theme():
    """Get Plotly theme configuration for World Bank FCV styling"""
    return {
        'layout': {
            'font': {
                'family': 'Arial, sans-serif', 
                'size': 12,
                'color': FCV_COLORS['dark']
            },
            'title': {
                'font': {'size': 18, 'color': FCV_COLORS['dark']},
                'x': 0.5,  # Center title
                'xanchor': 'center'
            },
            'colorway': [
                CHART_COLORS['primary_series'],
                CHART_COLORS['secondary_series'],
                CHART_COLORS['tertiary_series'],
                CHART_COLORS['quaternary_series'],
                CHART_COLORS['accent_series']
            ],
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'gridcolor': FCV_COLORS['gray'],
            'gridwidth': 0.5,
            'showgrid': True,
            'zeroline': False,
            
            # Axes styling
            'xaxis': {
                'showgrid': False,
                'showline': True,
                'linecolor': FCV_COLORS['gray'],
                'linewidth': 1,
                'ticks': 'outside',
                'tickcolor': FCV_COLORS['gray']
            },
            'yaxis': {
                'showgrid': True,
                'gridcolor': FCV_COLORS['gray'],
                'gridwidth': 0.5,
                'showline': False,
                'zeroline': False,
                'ticks': 'outside',
                'tickcolor': FCV_COLORS['gray']
            }
        }
    }

def style_poverty_chart():
    """Specific styling for poverty trend charts matching the template"""
    return {
        'fcs_color': CHART_COLORS['primary_series'],     # Blue for FCS
        'non_fcs_color': CHART_COLORS['secondary_series'], # Gray for Non-FCS
        'title_color': FCV_COLORS['dark'],
        'subtitle_color': FCV_COLORS['gray'],
        'grid_color': FCV_COLORS['gray'],
        'background_color': 'white',
        'text_color': FCV_COLORS['dark'],
        'accent_color': CHART_COLORS['tertiary_series']   # Light blue for highlights
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

def style_plotly_figure(fig, title=None, height=600, show_world_bank_logo=True):
    """Apply consistent World Bank FCV styling to a Plotly figure"""
    theme = get_plotly_theme()
    
    fig.update_layout(
        **theme['layout'],
        height=height,
        title=title,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='white',
            bordercolor=FCV_COLORS['gray'],
            borderwidth=1
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    # Add World Bank branding if requested
    if show_world_bank_logo:
        fig.add_annotation(
            text="FRAGILITY, CONFLICT & VIOLENCE<br>WORLD BANK GROUP",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=8, color=FCV_COLORS['gray']),
            align="left"
        )
    
    return fig

def create_presentation_title_slide(title, subtitle, presenter_name=None, 
                                  presenter_title=None, event_name=None, 
                                  event_details=None):
    """Create a title slide matching the World Bank FCV template"""
    
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background
    fig.patch.set_facecolor('white')
    
    # Main title (large, navy blue)
    ax.text(9.5, 8.5, title, fontsize=48, fontweight='bold', 
            color=FCV_COLORS['dark'], ha='right', va='top')
    
    # Subtitle (smaller, gray)
    if subtitle:
        ax.text(9.5, 7.8, subtitle, fontsize=32, 
                color=FCV_COLORS['gray'], ha='right', va='top')
    
    # Presenter information
    if presenter_name:
        ax.text(9.5, 5.5, presenter_name, fontsize=24, 
                color=CHART_COLORS['tertiary_series'], ha='right', va='top')
    
    if presenter_title:
        ax.text(9.5, 5.0, presenter_title, fontsize=16, 
                color=FCV_COLORS['gray'], ha='right', va='top')
    
    # Event information
    if event_name:
        ax.text(9.5, 3.5, event_name, fontsize=24, 
                color=CHART_COLORS['tertiary_series'], ha='right', va='top')
    
    if event_details:
        ax.text(9.5, 3.0, event_details, fontsize=16, 
                color=FCV_COLORS['gray'], ha='right', va='top')
    
    # Add World Bank globe icon placeholder (simplified)
    # This would be replaced with actual logo in practice
    circle = plt.Circle((2.5, 5), 2, fill=False, linewidth=8, 
                       color=CHART_COLORS['primary_series'], alpha=0.8)
    ax.add_patch(circle)
    
    # Globe lines
    ax.plot([0.5, 4.5], [5, 5], color=CHART_COLORS['primary_series'], linewidth=4, alpha=0.6)
    ax.plot([2.5, 2.5], [3, 7], color=CHART_COLORS['primary_series'], linewidth=4, alpha=0.6)
    
    # Footer branding
    ax.text(0.1, 0.3, "FRAGILITY, CONFLICT & VIOLENCE", fontsize=10, 
            color=FCV_COLORS['gray'], ha='left', va='bottom', fontweight='bold')
    ax.text(0.1, 0.1, "WORLD BANK GROUP", fontsize=14, 
            color=FCV_COLORS['dark'], ha='left', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Automatically apply styling when module is imported
apply_fcv_style()

print("✅ World Bank FCV styling applied successfully")
print(f"📊 Primary color (WB Blue): {FCV_COLORS['primary']}")
print(f"🎨 Chart colors: {len(CHART_COLORS)} defined")
print(f"⚠️  Risk levels: {len(RISK_COLORS)} defined")
print(f"🌍 Template: World Bank FCV Presentation Style")