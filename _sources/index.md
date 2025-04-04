![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/github/license/mglpurroy/slide-deck-analytics-1)

# Welcome to the FCV Analytics Notebook

:::{note}
⚠️ **This is a preliminary version and is under constant development.** The content, figures, and analyses may change as improvements are made. Please check back for updates and feel free to provide feedback!
:::

This repository contains the Jupyter Notebook used to generate key figures and insights for the FCV (Fragility, Conflict, and Violence) slide deck. The notebook processes various datasets and produces visualizations related to conflict trends, armed conflict duration, coup attempts, forcibly displaced persons, organized crime, and development indicators.


# Figures and Tables

📌 **Please CLICK on the sections below** to explore the figures, tables, and analysis.

```{tableofcontents}
```

# Data Sources


The analysis uses multiple datasets, including:

- **[ACLED (Armed Conflict Location & Event Data Project)](https://acleddata.com/)**: The Armed Conflict Location & Event Data Project (ACLED) is a disaggregated data collection, analysis, and crisis mapping project. ACLED collects information on the dates, actors, locations, fatalities, and types of all reported political violence and protest events around the world. The raw data is available through a license obtained by the World Bank.

- **[UCDP (Uppsala Conflict Data Program)](https://ucdp.uu.se/)**: Offers detailed conflict data, including battle-related deaths.

- **[World Bank Development Indicators](https://databank.worldbank.org/source/world-development-indicators)**: Covers socioeconomic data such as population projections, education, and sanitation.

- **[UNHCR (United Nations High Commissioner for Refugees)](https://www.unhcr.org/data.html)**: Data on forcibly displaced persons.

- **[OECD States of Fragility 2022](https://www.oecd.org/dac/states-of-fragility-2022-bc0ab39e-en.htm)**: Contains scores and metrics related to various dimensions of fragility, including political, economic, and social factors.

- **[UN Population Data](https://population.un.org/wpp/)**: Provides demographic data, including population estimates and projections.

- **[Poverty, Prosperity and Planet Report 2024](https://www.worldbank.org/en/publication/poverty-and-shared-prosperity)**: Covers global poverty trends and shared prosperity insights.

- **[Global Instances of Coups (GIC)](https://oefresearch.org/publications/global-instances-coups)**: Powell, Jonathan & Clayton Thyne. 2011. *Global Instances of Coups from 1950-Present*. Journal of Peace Research 48(2):249-259. Tracks coup attempts globally.

- **[Global Criminality Index](https://globalinitiative.net/analysis/global-organized-crime-index-2023/)**: Published by the Global Initiative Against Transnational Organized Crime, the *Global Organized Crime Index 2023* provides a comprehensive assessment of organized crime worldwide.



## Methodology

- **Data Preprocessing**: Raw data is cleaned, filtered, and structured for visualization.
- **Visualization**: Uses Python libraries (e.g., Matplotlib, Seaborn) to create figures.
- **Aggregation**: Data is grouped by key dimensions (e.g., region, income group, conflict type) to provide meaningful comparisons.

