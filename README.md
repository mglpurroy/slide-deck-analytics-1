# 📊 FCV Slide Deck Analytics

A comprehensive data analytics platform for analyzing **Fragility, Conflict, and Violence (FCV)** data with interactive visualizations and professional presentation.

## 🌐 Live Site
**https://mglpurroy.github.io/slide-deck-analytics-1/**

## 🎯 Overview

This repository provides deep insights into global fragility and conflict patterns through:

- **🗺️ States of Fragility 2022**: Interactive OECD fragility analysis
- **🔍 Crime Index Visualization**: Global crime statistics with dynamic plots  
- **👥 Population Trends**: UN demographic data by World Bank lending groups
- **⚔️ Conflict Analysis**: UCDP temporal and spatial conflict data
- **🏛️ FCS Map - FY25**: World Bank Fragile and Conflict-affected States mapping

## ✨ Key Features

### 🎨 **Interactive Visualizations**
- **Plotly-powered maps** with hover details and zoom controls
- **Dynamic dropdowns** for exploring different metrics
- **Professional matplotlib fallbacks** for static viewing
- **Mobile-responsive design** works on all devices

### 🔧 **Technical Excellence**
- **Single-branch workflow** - everything in `main`
- **Automated deployment** via GitHub Actions
- **Cached execution** for faster builds
- **Professional styling** with FCV color schemes
- **Clean presentation** - code hidden by default

### 📊 **Comprehensive Data Sources**
- **OECD**: States of Fragility 2022 dataset
- **Global Crime Index**: Country-level crime statistics
- **UN Population Division**: World population trends and projections
- **UCDP**: Uppsala Conflict Data Program conflict records
- **World Bank**: Lending group classifications and FCS designations

## 🚀 Quick Start

### **View the Analysis**
Simply visit: **https://mglpurroy.github.io/slide-deck-analytics-1/**

### **Local Development**
```bash
# Clone the repository
git clone https://github.com/mglpurroy/slide-deck-analytics-1.git
cd slide-deck-analytics-1

# Install dependencies
pip install -r requirements.txt

# Build the Jupyter Book
jupyter-book build docs

# View locally
open _build/html/index.html
```

## 📁 Repository Structure

```
├── 📂 docs/                    # Jupyter Book source
│   ├── 📓 notebooks/main.ipynb # Main analysis notebook
│   ├── 🎨 fcv_style.py        # Professional styling
│   ├── 📂 src/                 # Utility modules
│   ├── ⚙️ _config.yml          # Jupyter Book configuration
│   └── 📋 _toc.yml             # Table of contents
├── 📂 data/                    # Data files (25MB+ datasets)
├── 📂 scripts/                 # Maintenance and utility scripts
├── 📄 requirements.txt         # Python dependencies
├── 🔄 .github/workflows/       # Automated deployment
└── 📖 README.md               # This file
```

## 🔄 Automated Workflow

### **Development Process**
1. **Edit** content in `main` branch
2. **Commit** and push changes
3. **GitHub Actions** automatically builds
4. **Site updates** within 2-3 minutes

### **Deployment Pipeline**
- ✅ **Build**: Jupyter Book processes notebooks
- ✅ **Execute**: Notebooks run with 20-minute timeout
- ✅ **Generate**: Interactive plots and static fallbacks
- ✅ **Deploy**: Site updates automatically
- ✅ **Cache**: Faster subsequent builds

## 🎨 Styling & Presentation

### **FCV Color Palette**
- **Primary Blue** (#2E86AB): Stability and peace
- **Secondary Purple** (#A23B72): Complexity analysis
- **Accent Orange** (#F18F01): Attention and warnings
- **Danger Red** (#C73E1D): Conflict and high risk
- **Success Green** (#4A7C59): Positive outcomes

### **Professional Features**
- **Code cells hidden** for clean presentation
- **Interactive elements** with fallback static plots
- **Responsive design** for mobile and desktop
- **Publication-ready** styling throughout

## 🛠️ Maintenance

### **Update Analysis**
```bash
# Edit the main notebook
vim docs/notebooks/main.ipynb

# Push changes (auto-deploys)
git add . && git commit -m "Update analysis" && git push
```

### **Add New Data**
```bash
# Place files in data directory
cp new_dataset.csv data/

# Update notebook to reference new data
# Push changes for auto-deployment
```

### **Troubleshooting**
- **Build failing?** Check GitHub Actions logs
- **Plots not showing?** Verify data file paths
- **Styling issues?** Check `fcv_style.py` imports

## 📈 Performance

- **Fast Loading**: Optimized static generation
- **Efficient Caching**: Smart build system
- **Mobile Optimized**: Responsive across devices
- **SEO Friendly**: Proper metadata and structure

## 🤝 Contributing

This is a professional analytics platform. To contribute:

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** locally with `jupyter-book build docs`
5. **Submit** a pull request

## 📄 License

This project is available under the **MIT License**.

## 👨‍💻 Author

**Miguel Purroy** - Data Analysis and Visualization Specialist

---

## 🎯 Repository Highlights

- ✅ **Single Branch**: Everything in `main` for simplicity
- ✅ **Auto-Deploy**: Push to main → automatic site update  
- ✅ **Professional**: Publication-ready visualizations
- ✅ **Interactive**: Explore data with dynamic controls
- ✅ **Comprehensive**: Multiple data sources and methodologies
- ✅ **Maintainable**: Clean code and clear documentation

**Built with Jupyter Book • Deployed on GitHub Pages • Powered by Python**
