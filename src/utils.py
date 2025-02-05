"""
UCDP data processing utilities
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime, timedelta
import pycountry
import certifi
import urllib3
import warnings

class UCDP:
    """Class for handling UCDP conflict data"""
    
    @staticmethod
    def fetch_ucdp_data(pagesize: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch conflict data from UCDP API.
        
        Args:
            pagesize (int): Number of records to fetch per page
            
        Returns:
            List[Dict]: List of conflict records
        """
        base_url = "https://ucdpapi.pcr.uu.se/api/ucdpprioconflict/24.1"
        params = {
            'pagesize': pagesize,
            'page': 1
        }
        
        all_data = []
        while True:
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()  # Raise exception for bad status codes
                data = response.json()
                all_data.extend(data['Result'])
                
                # Check if there are more records to fetch
                if len(data['Result']) < pagesize:
                    print(f"Retrieved all {len(all_data)} records")
                    break
                else:
                    print(f"{len(all_data)} records fetched so far, fetching next page...")
                    params['page'] += 1
                    
            except requests.exceptions.RequestException as e:
                print(f"Error on page {params['page']}: {str(e)}")
                if 'response' in locals():
                    print(f"Response status: {response.status_code}")
                    print(f"Response text: {response.text[:200]}")
                break
            
        print(f"Total records fetched: {len(all_data)}")
        return all_data
    
    @staticmethod
    def process_conflict_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
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

    @staticmethod
    def calculate_average_duration(data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate the average duration of conflicts by start year, excluding active conflicts.
        
        Args:
            data (List[Dict]): Raw conflict data from UCDP API
            
        Returns:
            pd.DataFrame: DataFrame with average duration of conflicts by start year
        """
        # Create a DataFrame from all records
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
        
        # Convert records to DataFrame
        df_records = pd.DataFrame(records)
        
        # Convert dates to datetime
        df_records['start_date'] = pd.to_datetime(df_records['start_date'], errors='coerce')
        df_records['ep_end_date'] = pd.to_datetime(df_records['ep_end_date'], errors='coerce')
        
        # Keep the maximum ep_end_date by conflict_id
        df_records = df_records.groupby('conflict_id').agg({
            'start_date': 'first',
            'ep_end_date': 'max'
        }).reset_index()
        
        # Exclude active conflicts (where ep_end_date is NaT)
        df_records = df_records.dropna(subset=['ep_end_date'])
        
        # Calculate duration
        df_records['duration'] = (df_records['ep_end_date'] - df_records['start_date']).dt.days
        
        # Calculate average duration by start year
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