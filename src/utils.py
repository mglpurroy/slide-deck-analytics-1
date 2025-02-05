"""
UCDP data processing utilities
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pycountry
import certifi
import urllib3
import warnings
import time


class UCDP:
    """Class for handling UCDP conflict data"""
    
    def __init__(self):
        """Initialize UCDP class"""
        self.base_url = "https://ucdpapi.pcr.uu.se/api"
        
    def fetch_ucdp_data(self, pagesize: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch conflict data from UCDP API.
        """
        url = f"{self.base_url}/ucdpprioconflict/24.1"
        params = {
            'pagesize': pagesize,
            'page': 1
        }
        
        all_data = []
        while True:
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                all_data.extend(data['Result'])
                
                if len(data['Result']) < pagesize:
                    print(f"Retrieved all {len(all_data)} records")
                    break
                else:
                    print(f"{len(all_data)} records fetched so far...")
                    params['page'] += 1
                    
            except requests.exceptions.RequestException as e:
                print(f"Error on page {params['page']}: {str(e)}")
                if 'response' in locals():
                    print(f"Response status: {response.status_code}")
                    print(f"Response text: {response.text[:200]}")
                break
                
        print(f"Total records fetched: {len(all_data)}")
        return all_data

    def fetch_fatalities_data(self, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            countries: Optional[List[int]] = None,
                            pagesize: int = 1000) -> pd.DataFrame:
        """
        Fetch fatalities data from UCDP GED API
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        countries : List[int], optional
            List of Gleditsch & Ward country codes
        pagesize : int
            Number of records per page (max 1000)
            
        Returns:
        --------
        pd.DataFrame
            Processed fatalities data
        """
        url = f"{self.base_url}/gedevents/24.1"
        params = {'pagesize': min(pagesize, 1000), 'page': 1}
        
        # Add optional filters
        if start_date:
            params['StartDate'] = start_date
        if end_date:
            params['EndDate'] = end_date
        if countries:
            params['Country'] = ','.join(map(str, countries))
        
        all_data = []
        while True:
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                all_data.extend(data['Result'])
                
                if len(data['Result']) < pagesize:
                    print(f"Retrieved all {len(all_data)} records")
                    break
                else:
                    print(f"{len(all_data)} records fetched so far...")
                    params['page'] += 1
                    
            except requests.exceptions.RequestException as e:
                print(f"Error on page {params['page']}: {str(e)}")
                if 'response' in locals():
                    print(f"Response status: {response.status_code}")
                    print(f"Response text: {response.text[:200]}")
                break
        
        return self.process_fatalities_data(all_data)

    def process_fatalities_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process fatalities data from UCDP GED API
        
        Parameters:
        -----------
        data : List[Dict]
            Raw data from UCDP GED API
            
        Returns:
        --------
        pd.DataFrame
            Processed data with yearly fatalities by type
        """
        if not data:
            print("No data to process")
            return pd.DataFrame()
        
        # Create records with fatality counts
        records = []
        for record in data:
            try:
                year = int(record['year'])
                deaths_a = int(record.get('deaths_a', 0) or 0)  # Government/Side A
                deaths_b = int(record.get('deaths_b', 0) or 0)  # Rebels/Side B
                deaths_civilians = int(record.get('deaths_civilians', 0) or 0)
                deaths_unknown = int(record.get('deaths_unknown', 0) or 0)
                type_of_violence = int(record.get('type_of_violence', 0) or 0)
                
                records.append({
                    'year': year,
                    'deaths_a': deaths_a,
                    'deaths_b': deaths_b,
                    'deaths_civilians': deaths_civilians,
                    'deaths_unknown': deaths_unknown,
                    'type_of_violence': type_of_violence,
                    'total_deaths': deaths_a + deaths_b + deaths_civilians + deaths_unknown
                })
            except (KeyError, ValueError, TypeError) as e:
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Create pivot table with deaths by year and violence type
        df_pivot = pd.pivot_table(
            df,
            index='year',
            columns='type_of_violence',
            values=['deaths_a', 'deaths_b', 'deaths_civilians', 'deaths_unknown', 'total_deaths'],
            aggfunc='sum',
            fill_value=0
        )
        
        # Flatten column names
        df_pivot.columns = [f'{col[0]}_{col[1]}' for col in df_pivot.columns]
        
        # Rename violence types
        violence_types = {
            1: 'state_based',
            2: 'non_state',
            3: 'one_sided'
        }
        
        renamed_cols = []
        for col in df_pivot.columns:
            for type_num, type_name in violence_types.items():
                if str(type_num) in col:
                    renamed_cols.append(col.replace(str(type_num), type_name))
                    break
            else:
                renamed_cols.append(col)
        
        df_pivot.columns = renamed_cols
        
        # Add total fatalities column
        df_pivot['total_fatalities'] = df_pivot[[col for col in df_pivot.columns 
                                               if col.startswith('total_deaths')]].sum(axis=1)
        
        print(f"\nProcessed fatalities data summary:")
        print(f"Years covered: {df_pivot.index.min()}-{df_pivot.index.max()}")
        print(f"Total fatalities: {int(df_pivot['total_fatalities'].sum()):,}")
        
        return df_pivot.sort_index()

    def process_conflict_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process conflict data into yearly counts by type"""
        if not data:
            print("No data to process")
            return pd.DataFrame()
        
        # Create a DataFrame from all records
        records = []
        for record in data:
            try:
                records.append({
                    'year': int(record['year']),
                    'type_of_conflict': int(record['type_of_conflict'])
                })
            except (KeyError, ValueError, TypeError) as e:
                continue
        
        # Convert records to DataFrame
        df_records = pd.DataFrame(records)
        
        # Group by year and conflict type to get counts
        df_pivot = pd.pivot_table(
            df_records,
            index='year',
            columns='type_of_conflict',
            aggfunc='size',
            fill_value=0
        )
        
        # Ensure all conflict types are present
        for i in range(1, 5):
            if i not in df_pivot.columns:
                df_pivot[i] = 0
        
        # Sort columns and rename
        df_pivot = df_pivot[[1, 2, 3, 4]]
        df_pivot = df_pivot.rename(columns={
            1: 'Extra-systemic conflicts',
            2: 'Inter-state conflicts',
            3: 'Internal conflicts',
            4: 'Internationalized-internal conflicts'
        })
        
        # Sort by year
        df_pivot = df_pivot.sort_index()
        
        print(f"\nProcessed data summary:")
        print(f"Years covered: {df_pivot.index.min()}-{df_pivot.index.max()}")
        print(f"Number of years: {len(df_pivot)}")
        print(f"Total conflicts: {df_pivot.sum().sum()}")
        
        return df_pivot

    def calculate_average_duration(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate the average duration of conflicts by start year"""
        records = []
        for record in data:
            try:
                records.append({
                    'conflict_id': record.get('conflict_id'),
                    'start_date': record.get('start_date'),
                    'ep_end_date': record.get('ep_end_date')
                })
            except (KeyError, ValueError, TypeError) as e:
                continue
        
        df_records = pd.DataFrame(records)
        df_records['start_date'] = pd.to_datetime(df_records['start_date'], errors='coerce')
        df_records['ep_end_date'] = pd.to_datetime(df_records['ep_end_date'], errors='coerce')
        
        df_records = df_records.groupby('conflict_id').agg({
            'start_date': 'first',
            'ep_end_date': 'max'
        }).reset_index()
        
        df_records = df_records.dropna(subset=['ep_end_date'])
        df_records['duration'] = (df_records['ep_end_date'] - df_records['start_date']).dt.days
        
        avg_duration = df_records.groupby(df_records['start_date'].dt.year)['duration'].mean()
        avg_duration.name = 'Average Duration'
        
        return avg_duration.reset_index().rename(columns={'start_date': 'Year'})

    @staticmethod
    def get_conflict_colors() -> Dict[str, str]:
        """Get color mapping for conflict types"""
        return {
            'Extra-systemic conflicts': '#2ecc71',
            'Inter-state conflicts': '#3498db',
            'Internal conflicts': '#f1c40f',
            'Internationalized-internal conflicts': '#9b59b6'
        }
    
        
class RegionMapper:
    """
    A class to map countries to their regions using various methods
    """
    
    def __init__(self):
        """
        Initialize the RegionMapper with predefined region mappings
        """
        # Predefined region mappings
        self._manual_regions = {
            # North America
            'United States': 'Americas',
            'Canada': 'Americas',
            'Mexico': 'Americas',
            
            # Central America
            'Guatemala': 'Americas',
            'Honduras': 'Americas',
            'El Salvador': 'Americas',
            'Nicaragua': 'Americas',
            'Costa Rica': 'Americas',
            'Panama': 'Americas',
            
            # Caribbean
            'Cuba': 'Americas',
            'Haiti': 'Americas',
            'Dominican Republic': 'Americas',
            'Jamaica': 'Americas',
            'Bahamas': 'Americas',
            'Barbados': 'Americas',
            'Trinidad and Tobago': 'Americas',
            
            # South America
            'Brazil': 'Americas',
            'Argentina': 'Americas',
            'Colombia': 'Americas',
            'Peru': 'Americas',
            'Venezuela': 'Americas',
            'Chile': 'Americas',
            'Ecuador': 'Americas',
            'Bolivia': 'Americas',
            'Paraguay': 'Americas',
            'Uruguay': 'Americas',
            
            # Africa regions
            'Egypt': 'Africa',
            'Nigeria': 'Africa',
            'South Africa': 'Africa',
            'Kenya': 'Africa',
            'Ethiopia': 'Africa',
            'Ghana': 'Africa',
            'Senegal': 'Africa',
            'Algeria': 'Africa',
            'Morocco': 'Africa',
            'Tunisia': 'Africa',
        }
        
        # Load World Bank region data if available
        try:
            self._load_world_bank_regions()
        except Exception as e:
            print(f"Could not load World Bank regions: {e}")
            self._world_bank_regions = {}
    
    def _load_world_bank_regions(self):
        """
        Fetch and store World Bank region data
        """
        # World Bank API endpoint for country classifications
        url = "http://api.worldbank.org/v2/country?format=json&per_page=1000"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()[1]
            self._world_bank_regions = {
                country['name']: country.get('region', {}).get('value', 'Other')
                for country in data
            }
        else:
            print("Failed to fetch World Bank region data")
            self._world_bank_regions = {}
    
    def get_region(self, country_name):
        """
        Determine the region for a given country name
        
        Parameters:
        -----------
        country_name : str
            Name of the country
        
        Returns:
        --------
        str
            Region of the country (Americas, Africa, Other)
        """
        # Check manual mappings first
        if country_name in self._manual_regions:
            return self._manual_regions[country_name]
        
        # Check World Bank regions
        if country_name in self._world_bank_regions:
            region = self._world_bank_regions[country_name]
            if region in ['North America', 'Latin America & Caribbean']:
                return 'Americas'
            elif region == 'Sub-Saharan Africa' or region == 'Middle East & North Africa':
                return 'Africa'
            else:
                return 'Other'
        
        # Fallback to manual classification based on known regions
        for key, value in self._manual_regions.items():
            if key in country_name:
                return value
        
        # Default to Other if no region found
        return 'Other'

class GIC:
    """
    Governmental Instability and Coups (GIC) Data Analysis Class
    """
    
    def __init__(self, data_path='http://www.uky.edu/~clthyn2/coup_data/powell_thyne_coups_final.txt'):
        """
        Initialize the GIC class with coup data and region mapping
        
        Parameters:
        -----------
        data_path : str, optional
            Path or URL to the coup dataset
        """
        # Initialize region mapper
        self.region_mapper = RegionMapper()
        
        # Read the raw data
        self.raw_data = pd.read_csv(data_path, sep='\t', dtype={
            'country': str, 
            'year': int, 
            'month': int, 
            'coup': int
        })
        
        # Add region column
        self.raw_data['region'] = self.raw_data['country'].apply(self.region_mapper.get_region)
        
        # Process the data
        self.processed_data = self._process_data()

    def _process_data(self, start_year=1950, end_year=None):
        """
        Process the raw coup data
        
        Parameters:
        -----------
        start_year : int, optional
            Starting year for data processing (default is 1950)
        end_year : int, optional
            Ending year for data processing (default is None, which uses the max year in data)
        
        Returns:
        --------
        pandas.DataFrame
            Processed coup data
        """
        # Create a copy of the raw data
        gic = self.raw_data.copy()
        
        # Filter years
        gic = gic[gic['year'] >= start_year]
        if end_year:
            gic = gic[gic['year'] <= end_year]
        
        # Group by country, year, month, and region
        gic_grouped = gic.groupby(['country', 'year', 'month', 'region']).agg({
            'coup': ['count', 'max']
        }).reset_index()
        
        # Flatten column names
        gic_grouped.columns = ['country', 'year', 'month', 'region', 'coup_count', 'coup_status']
        
        # Create coup indicators
        gic_grouped['GIC_coup_successful'] = (gic_grouped['coup_status'] == 2).astype(int)
        gic_grouped['GIC_coup_failed'] = (gic_grouped['coup_status'] == 1).astype(int)
        
        return gic_grouped
    
class ACLEDDataFetcher:
    """
    A class to fetch and process ACLED (Armed Conflict Location & Event Data) 
    """
    
    def __init__(self, api_key, email):
        """
        Initialize the ACLED data fetcher
        """
        self.api_key = api_key
        self.email = email
        self.base_url = "https://api.acleddata.com/acled/read/"
        
        # Selected fields matching the R script
        self.fields = [
            "event_id_cnty", "iso", "event_date", "event_type", "fatalities", 
            "disorder_type", "sub_event_type", "actor1", "actor2", 
            "assoc_actor_1", "assoc_actor_2", "latitude", "longitude"
        ]
        
        # Suppress SSL warnings at instance level
        warnings.simplefilter('ignore', urllib3.exceptions.InsecureRequestWarning)
    
    def fetch_data(self, start_date, end_date, max_retries=3):
        """
        Fetch data from ACLED API with improved error handling
        """
        # Remove fields parameter - let's get all fields by default
        params = {
            'key': self.api_key,
            'email': self.email,
            'event_date': f"{start_date.strftime('%Y-%m-%d')}|{end_date.strftime('%Y-%m-%d')}",
            'event_date_where': 'BETWEEN',
            'limit': 0
        }
        
        session = requests.Session()
        retry = requests.packages.urllib3.util.Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        
        try:
            response = session.get(
                self.base_url, 
                params=params, 
                verify=False  # Explicitly disable SSL verification
            )
            response.raise_for_status()
            
            # Debug response
            json_response = response.json()
            if 'error' in json_response:
                print(f"API Error: {json_response['error']}")
                return pd.DataFrame()
                
            if 'data' not in json_response:
                print(f"Unexpected API response format. Response keys: {list(json_response.keys())}")
                print(f"Raw response: {response.text[:500]}...")
                return pd.DataFrame()
            
            return pd.DataFrame(json_response['data'])
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
        except ValueError as e:
            print(f"Error parsing JSON response: {str(e)}")
            print(f"Raw response: {response.text[:500]}...")
            return pd.DataFrame()
        finally:
            session.close()

    def _is_gang_related(self, row):
        """
        Determine if an event is gang-related
        
        Parameters:
        -----------
        row : pd.Series
            A single row of event data
        
        Returns:
        --------
        bool
            True if the event is gang-related, False otherwise
        """
        gang_keywords = ["Unidentified Gang", "Unidentified Armed Group"]
        
        # Check all actor fields for gang-related keywords
        actor_fields = ['actor1', 'actor2', 'assoc_actor_1', 'assoc_actor_2']
        return any(
            any(keyword in str(row[field]) for keyword in gang_keywords) 
            for field in actor_fields
        )
                
    def process_data(self, acled_data):
        """
        Process raw ACLED data
        
        Parameters:
        -----------
        acled_data : pd.DataFrame
            Raw ACLED data
        
        Returns:
        --------
        pd.DataFrame
            Processed ACLED data
        """
        # Convert data types
        acled_data['iso'] = pd.to_numeric(acled_data['iso'], errors='coerce')
        acled_data['fatalities'] = pd.to_numeric(acled_data['fatalities'], errors='coerce')
        
        # Convert event date
        acled_data['event_date'] = pd.to_datetime(acled_data['event_date'])
        
        # Add year and month columns
        acled_data['year'] = acled_data['event_date'].dt.year
        acled_data['month'] = acled_data['event_date'].dt.month
        
        # Convert country code
        def convert_country_code(iso):
            try:
                return pycountry.countries.get(numeric=str(iso)).alpha_3
            except (AttributeError, ValueError):
                return np.nan
        
        acled_data['iso3'] = acled_data['iso'].apply(convert_country_code)
        
        # Add gang-related flag
        acled_data['gang'] = acled_data.apply(self._is_gang_related, axis=1)
        
        # Categorize event types and sub-types
        for col in ['event_type', 'disorder_type', 'sub_event_type']:
            acled_data[col] = acled_data[col].astype('category')
        
        return acled_data
    
    def get_conflict_related_deaths(self, acled_data):
        """
        Calculate conflict-related deaths
        
        Parameters:
        -----------
        acled_data : pd.DataFrame
            Processed ACLED data
        
        Returns:
        --------
        pd.DataFrame
            Monthly conflict-related deaths by country
        """
        # Filter for conflict-related events with fatalities, excluding gang events
        conflict_deaths = acled_data[
            (acled_data['event_type'].isin(['Battles', 'Violence against civilians', 'Explosions/Remote violence'])) &
            (acled_data['fatalities'] > 0) &
            (~acled_data['gang'])
        ]
        
        # Aggregate deaths by country, year, and month
        return conflict_deaths.groupby(['iso3', 'year', 'month'])['fatalities'].sum().reset_index(name='ACLED_conflict_related_deaths')
    
    def get_event_counts(self, acled_data):
        """
        Calculate event counts by type
        
        Parameters:
        -----------
        acled_data : pd.DataFrame
            Processed ACLED data
        
        Returns:
        --------
        pd.DataFrame
            Monthly event counts by country and type
        """
        # Exclude gang-related events
        non_gang_data = acled_data[~acled_data['gang']]
        
        # Count events by type
        event_counts = non_gang_data.groupby(['iso3', 'year', 'month', 'event_type']).size().unstack(fill_value=0)
        event_counts.columns = [f'ACLED_{col.lower().replace(" ", "_")}' for col in event_counts.columns]
        
        # Reset index to make iso3, year, month regular columns
        return event_counts.reset_index()


class UNHCRDataFinder:
    """
    A class to fetch data from UNHCR's Refugee Data Finder API.
    No API key required - the API is open to all.
    """
    
    def __init__(self):
        """Initialize the UNHCR Data Finder client"""
        self.base_url = "https://api.unhcr.org/population/v1"
        self.headers = {"Accept": "application/json"}
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make a request to the UNHCR API with error handling
        
        Parameters:
        -----------
        endpoint : str
            API endpoint to call
        params : dict, optional
            Query parameters
            
        Returns:
        --------
        dict
            JSON response from the API
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {endpoint}: {str(e)}")
            return None
            
    def _fetch_paginated_data(self, endpoint: str, params: Dict = None) -> List[Dict]:
        """
        Fetch all pages of data from a paginated endpoint
        """
        if params is None:
            params = {}
            
        params['limit'] = params.get('limit', 1000)
        params['page'] = 1
        all_data = []
        
        while True:
            response = self._make_request(endpoint, params)
            if not response or 'items' not in response:
                break
                
            all_data.extend(response['items'])
            
            if params['page'] >= response.get('totalPages', params['page']):
                break
                
            params['page'] += 1
            time.sleep(0.1)  # Small delay to be nice to the API
            
        return all_data

    def get_metadata(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch metadata (countries, regions, years)
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing metadata DataFrames
        """
        metadata = {}
        
        # Fetch countries
        countries_data = self._make_request('countries')
        if countries_data and 'items' in countries_data:
            metadata['countries'] = pd.DataFrame(countries_data['items'])
            
        # Fetch regions
        regions_data = self._make_request('regions')
        if regions_data and 'items' in regions_data:
            metadata['regions'] = pd.DataFrame(regions_data['items'])
            
        # Fetch years
        years_data = self._make_request('years')
        if years_data and 'items' in years_data:
            metadata['years'] = pd.DataFrame(years_data['items'])
            
        return metadata

    def get_population_data(self,
                          year_from: Optional[int] = None,
                          year_to: Optional[int] = None,
                          years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch population statistics including refugees, asylum-seekers, and others
        needing international protection
        """
        params = {}
        if year_from:
            params['yearFrom'] = year_from
        if year_to:
            params['yearTo'] = year_to
        if years:
            params['year'] = ','.join(map(str, years))
            
        data = self._fetch_paginated_data('population', params)
        return pd.DataFrame(data) if data else pd.DataFrame()

    def get_unrwa_data(self,
                      year_from: Optional[int] = None,
                      year_to: Optional[int] = None,
                      years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch data on Palestine refugees under UNRWA's mandate
        """
        params = {}
        if year_from:
            params['yearFrom'] = year_from
        if year_to:
            params['yearTo'] = year_to
        if years:
            params['year'] = ','.join(map(str, years))
            
        data = self._fetch_paginated_data('unrwa', params)
        return pd.DataFrame(data) if data else pd.DataFrame()

    def get_idp_data(self,
                    year_from: Optional[int] = None,
                    year_to: Optional[int] = None,
                    years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch IDP data from IDMC
        """
        params = {}
        if year_from:
            params['yearFrom'] = year_from
        if year_to:
            params['yearTo'] = year_to
        if years:
            params['year'] = ','.join(map(str, years))
            
        data = self._fetch_paginated_data('idmc', params)
        return pd.DataFrame(data) if data else pd.DataFrame()

    def get_demographics(self,
                        year: int,
                        population_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch demographic data by age and sex
        
        Parameters:
        -----------
        year : int
            Year for demographic data
        population_types : List[str], optional
            List of population types (e.g., ['refugees', 'oip'])
        """
        params = {'year': year}
        if population_types:
            params['columns[]'] = population_types
            
        data = self._fetch_paginated_data('demographics', params)
        return pd.DataFrame(data) if data else pd.DataFrame()

    def get_all_displacement_data(self,
                                year_from: int,
                                year_to: int) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive displacement statistics including:
        - Refugees and others under UNHCR's mandate
        - Palestine refugees under UNRWA's mandate
        - IDPs (from IDMC)
        
        Parameters:
        -----------
        year_from : int
            Start year
        year_to : int
            End year
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing DataFrames for each type of data
        """
        return {
            'unhcr_population': self.get_population_data(year_from, year_to),
            'unrwa_refugees': self.get_unrwa_data(year_from, year_to),
            'idps': self.get_idp_data(year_from, year_to)
        }
        
    def process_displacement_data(self,
                                year_from: int,
                                year_to: int) -> pd.DataFrame:
        """
        Process and combine displacement data into a format suitable for visualization,
        with five categories:
        1. Internally displaced people
        2. Refugees under UNHCR's mandate
        3. Palestine refugees under UNRWA's mandate
        4. Asylum-seekers
        5. Other people in need of international protection
        
        Parameters:
        -----------
        year_from : int
            Start year
        year_to : int
            End year
            
        Returns:
        --------
        pd.DataFrame
            Processed data with columns for each category by year
        """
        # Fetch raw data
        raw_data = self.get_all_displacement_data(year_from, year_to)
        
        # Process UNHCR population data
        unhcr_df = raw_data['unhcr_population']
        population_data = unhcr_df[['year', 'refugees', 'asylum_seekers', 'oip']].fillna(0)
        
        # Process UNRWA data
        unrwa_df = raw_data['unrwa_refugees']
        unrwa_data = unrwa_df[['year', 'total']].rename(columns={'total': 'palestine_refugees'})
        
        # Process IDP data
        idp_df = raw_data['idps']
        idp_data = idp_df[['year', 'total']].rename(columns={'total': 'idps'})
        
        # Merge all datasets on year
        merged_df = (population_data
                    .merge(unrwa_data, on='year', how='outer')
                    .merge(idp_data, on='year', how='outer')
                    .fillna(0))
        
        # Ensure all years are present
        all_years = range(year_from, year_to + 1)
        merged_df = (merged_df.set_index('year')
                    .reindex(all_years)
                    .fillna(0)
                    .reset_index())
        
        # Rename columns to match documentation
        merged_df = merged_df.rename(columns={
            'year': 'year',
            'refugees': 'refugees_unhcr',
            'asylum_seekers': 'asylum_seekers',
            'oip': 'other_protection',
            'palestine_refugees': 'refugees_unrwa',
            'idps': 'idps'
        })
        
        return merged_df.sort_values('year')