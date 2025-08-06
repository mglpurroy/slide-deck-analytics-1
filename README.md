# FCV Slide Deck Analytics

A comprehensive Jupyter Book for analyzing fragility, conflict, and violence (FCV) data with interactive visualizations.

## 🎯 Overview

This repository contains a data analytics slide deck that provides insights into:
- **States of Fragility 2022**: Interactive analysis of global fragility metrics
- **Crime Index Visualization**: Global crime data with interactive plots
- **Population Trends**: UN population data analysis by lending groups
- **Conflict Analysis**: UCDP conflict data with temporal and spatial analysis

## 🗺️ Interactive Features

- **Interactive Maps**: Plotly-based visualizations with hover details and zooming
- **Dynamic Plots**: Explore different metrics using dropdown controls
- **Static Fallbacks**: Professional matplotlib plots always visible
- **Clean Presentation**: Code is hidden by default for a clean slide deck experience

## 📊 Data Sources

- **States of Fragility 2022**: OECD fragility data
- **Global Crime Index**: Crime statistics by country
- **UN Population Data**: World population trends and projections
- **UCDP Data**: Uppsala Conflict Data Program conflict information
- **World Bank**: Lending group classifications

## 🚀 Deployment

This Jupyter Book is automatically built and deployed to GitHub Pages using GitHub Actions.

**Live Site**: https://mglpurroy.github.io/slide-deck-analytics-1/

## 🛠️ Development

### Local Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Build the book: `jupyter-book build docs`

### Structure

```
├── docs/                    # Jupyter Book source
│   ├── notebooks/          # Main analysis notebook
│   ├── _config.yml         # Jupyter Book configuration
│   └── _toc.yml           # Table of contents
├── data/                   # Data files (25MB+ datasets)
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
└── .github/workflows/     # Deployment automation
```

## 📈 Features

- **Professional Presentation**: Clean, publication-ready output
- **Interactive Visualizations**: Explore data with dynamic controls
- **Comprehensive Analysis**: Multiple data sources and methodologies
- **Automated Deployment**: Continuous integration and deployment
- **Mobile Responsive**: Works on all device sizes

## 🔧 Maintenance

- **Hide/Show Code**: Use `python scripts/hide_code_cells.py [show]`
- **Update Plots**: Modify notebook and rebuild
- **Add Data**: Place files in `data/` directory
- **Deploy**: Push to main branch triggers automatic deployment

## 📄 License

This project is open source and available under the MIT License.

## 👥 Contributors

- Miguel Purroy - Data Analysis and Visualization
- Automated deployment and repository management

---

*Built with Jupyter Book, deployed on GitHub Pages*
