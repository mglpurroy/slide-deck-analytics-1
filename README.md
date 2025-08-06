# FCV Analytics Notebook

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/github/license/mglpurroy/slide-deck-analytics-1)
![Build Status](https://github.com/mglpurroy/slide-deck-analytics-1/workflows/Build%20and%20Deploy%20Jupyter%20Book/badge.svg)

A comprehensive Jupyter Book containing analytics and visualizations for the World Bank's Fragility, Conflict, and Violence (FCV) slide deck. This repository processes various datasets and produces key insights related to conflict trends, armed conflict duration, coup attempts, forcibly displaced persons, organized crime, and development indicators.

## 📖 Live Documentation

Visit the live Jupyter Book: **[https://mglpurroy.github.io/slide-deck-analytics-1/](https://mglpurroy.github.io/slide-deck-analytics-1/)**

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mglpurroy/slide-deck-analytics-1.git
   cd slide-deck-analytics-1
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv fcv-env
   source fcv-env/bin/activate  # On Windows: fcv-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Build the Jupyter Book locally**
   ```bash
   jupyter-book build _sources
   ```

5. **View the book**
   ```bash
   # Open _build/html/index.html in your browser
   # Or serve it locally:
   python -m http.server 8000 -d _build/html
   ```

## 📁 Repository Structure

```
├── _sources/                 # Source files for Jupyter Book
│   ├── index.md             # Main landing page
│   ├── notebooks/           # Jupyter notebooks
│   │   └── main.ipynb       # Main FCV analytics notebook
│   └── README.md            # Additional documentation
├── .github/                 # GitHub Actions workflows
│   └── workflows/
│       └── deploy.yml       # Automated build and deploy
├── _config.yml              # Jupyter Book configuration
├── _toc.yml                 # Table of contents structure
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Jupyter Book Configuration (`_config.yml`)

The main configuration includes:
- **Execution settings**: Notebooks are re-executed on each build with a 20-minute timeout
- **Repository integration**: GitHub buttons for issues, editing, and repository access
- **Theme customization**: Modern book theme with navigation enhancements
- **Extensions**: MyST markdown extensions for enhanced content

### Table of Contents (`_toc.yml`)

Defines the book structure and navigation. Currently includes:
- Landing page (`index.md`)
- Main FCV Analytics Dashboard (`notebooks/main.ipynb`)

## 🚀 Deployment

### Automatic Deployment (Recommended)

The repository is configured with GitHub Actions for automatic deployment:

1. **Push to main branch** - Triggers automatic build and deployment
2. **Pull requests** - Builds the book for testing (no deployment)
3. **Manual trigger** - Can be triggered manually from GitHub Actions tab

### Manual Deployment

For manual deployment to GitHub Pages:

```bash
# Build the book
jupyter-book build _sources

# Deploy to gh-pages branch (requires ghp-import)
pip install ghp-import
ghp-import -n -p -f _build/html
```

## 📊 Data Sources

The notebook analyzes data from multiple authoritative sources:

- **[ACLED](https://acleddata.com/)**: Armed Conflict Location & Event Data
- **[UCDP](https://ucdp.uu.se/)**: Uppsala Conflict Data Program
- **[World Bank Development Indicators](https://databank.worldbank.org/source/world-development-indicators)**: Socioeconomic data
- **[UNHCR](https://www.unhcr.org/data.html)**: Forcibly displaced persons data
- **[OECD States of Fragility 2022](https://www.oecd.org/dac/states-of-fragility-2022-bc0ab39e-en.htm)**: Fragility metrics
- **[UN Population Data](https://population.un.org/wpp/)**: Demographic data
- **[Global Instances of Coups (GIC)](https://oefresearch.org/publications/global-instances-coups)**: Coup attempt data
- **[Global Criminality Index](https://globalinitiative.net/analysis/global-organized-crime-index-2023/)**: Organized crime assessment

## 🛠️ Development

### Adding New Content

1. **Add notebooks**: Place new `.ipynb` files in `_sources/notebooks/`
2. **Update table of contents**: Add entries to `_toc.yml`
3. **Test locally**: Run `jupyter-book build _sources` to test
4. **Commit and push**: Changes to `main` branch will trigger automatic deployment

### Notebook Best Practices

- **Use appropriate timeouts**: Long-running cells should complete within the 20-minute limit
- **Hide complex code**: Use `tags: ["hide-input"]` for code cells that generate visualizations
- **Add documentation**: Include markdown cells explaining analysis steps
- **Optimize performance**: Cache intermediate results when possible

### Troubleshooting

**Notebook execution timeout:**
- Increase timeout in `_config.yml` under `execute.timeout`
- Optimize code for better performance
- Consider breaking long-running cells into smaller chunks

**Build failures:**
- Check the GitHub Actions logs for detailed error messages
- Ensure all required dependencies are in `requirements.txt`
- Test the build locally before pushing

**Missing data:**
- Verify data source URLs and API endpoints
- Check for authentication requirements
- Consider adding data caching mechanisms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test locally
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/mglpurroy/slide-deck-analytics-1/issues)
- **Discussions**: Join the conversation in [GitHub Discussions](https://github.com/mglpurroy/slide-deck-analytics-1/discussions)
- **Documentation**: Visit the [Jupyter Book documentation](https://jupyterbook.org/) for technical details

## 📈 Monitoring

- **Build Status**: Check the [Actions tab](https://github.com/mglpurroy/slide-deck-analytics-1/actions) for build history
- **Live Site**: Monitor the [GitHub Pages site](https://mglpurroy.github.io/slide-deck-analytics-1/) for updates
- **Performance**: Book builds typically take 15-25 minutes due to data processing requirements

---

**Note**: This is a preliminary version under active development. Content, figures, and analyses may change as improvements are made. Please check back for updates and provide feedback!