# Slide Deck Analytics

This repository contains the Jupyter Notebook used to generate key figures and insights for the FCV (Fragility, Conflict, and Violence) slide deck. The notebook processes various datasets and produces visualizations related to conflict trends, armed conflict duration, coup attempts, forcibly displaced persons, organized crime, and development indicators.

## Data Sources

The analysis uses multiple datasets, including:

- **ACLED (Armed Conflict Location & Event Data Project)**: Provides real-time data on political violence and protests.
- **UCDP (Uppsala Conflict Data Program)**: Offers detailed conflict data, including battle-related deaths.
- **World Bank Development Indicators**: Covers socioeconomic data such as population projections, education, and sanitation.
- **UNHCR (United Nations High Commissioner for Refugees)**: Data on forcibly displaced persons.

## Figures Generated

- **Number of Armed Conflicts by Type**: Tracks different conflict types over time (extra-systemic, inter-state, internal, and internationalized internal conflicts).
- **Average Duration of Armed Conflicts (1965-2023)**: Shows the changing trends in conflict longevity.
- **Global Coup Attempts**: Maps coup occurrences and trends.
- **Conflict-Induced Fatalities**: Visualizes fatalities by region (ACLED) and by type of conflict (UCDP).
- **Forcibly Displaced Persons**: Highlights migration patterns due to conflicts.
- **Global Organized Crime**: Provides insights into the spread and severity of organized crime.
- **Development Indicators**: Includes population projections, education completion rates, and sanitation trends.

## Methodology

- **Data Preprocessing**: Raw data is cleaned, filtered, and structured for visualization.
- **Visualization**: Uses Python libraries (e.g., Matplotlib, Seaborn) to create figures.
- **Aggregation**: Data is grouped by key dimensions (e.g., region, income group, conflict type) to provide meaningful comparisons.

## How to Use

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook main.ipynb
```

## Contact

For any questions or contributions, feel free to open an issue or submit a pull request.