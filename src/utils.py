"""
UCDP data processing utilities
"""
import os
import sys
import re
import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pycountry
import certifi
import urllib3
import warnings
import time
import socket
import glob
from openpyxl import load_workbook
import country_converter as coco
import logging
from urllib.parse import urljoin, urlparse, parse_qs, urlencode



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
        Fetch fatalities data from UCDP GED API and GED Candidate
        
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
            Processed fatalities data combining both GED and GED Candidate
        """
        def fetch_from_endpoint(endpoint: str, params: dict) -> List[Dict]:
            """Helper function to fetch data from a specific endpoint"""
            url = f"{self.base_url}/{endpoint}"
            params = params.copy()
            params['page'] = 1
            all_data = []
            
            while True:
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    all_data.extend(data['Result'])
                    
                    if len(data['Result']) < pagesize:
                        print(f"Retrieved all {len(all_data)} records from {endpoint}")
                        break
                    else:
                        print(f"{len(all_data)} records fetched from {endpoint} so far...")
                        params['page'] += 1
                        
                except requests.exceptions.RequestException as e:
                    print(f"Error on page {params['page']} from {endpoint}: {str(e)}")
                    break
                    
            return all_data
        
        # Prepare base parameters
        params = {'pagesize': min(pagesize, 1000)}
        if start_date:
            params['StartDate'] = start_date
        if end_date:
            params['EndDate'] = end_date
        if countries:
            params['Country'] = ','.join(map(str, countries))
            
        # Fetch from regular GED dataset (24.1)
        print("\nFetching from GED dataset 24.1...")
        ged_data = fetch_from_endpoint('gedevents/24.1', params)
        
        # Fetch from latest GED Candidate
        print("\nFetching from GED Candidate 24.01.24.12...")
        candidate_data = fetch_from_endpoint('gedevents/24.01.24.12', params)
        
        # Combine the data, prioritizing GED over GED Candidate for overlapping dates
        if ged_data:
            ged_df = pd.DataFrame(ged_data)
            ged_df['source'] = 'ged'
        else:
            ged_df = pd.DataFrame()
            
        if candidate_data:
            candidate_df = pd.DataFrame(candidate_data)
            candidate_df['source'] = 'candidate'
        else:
            candidate_df = pd.DataFrame()
            
        # Combine datasets
        if not ged_df.empty and not candidate_df.empty:
            # Convert date columns to datetime
            ged_df['date_start'] = pd.to_datetime(ged_df['date_start'])
            candidate_df['date_start'] = pd.to_datetime(candidate_df['date_start'])
            
            # Find the latest date in GED dataset
            latest_ged_date = ged_df['date_start'].max()
            
            # Only use candidate data after the latest GED date
            candidate_df = candidate_df[candidate_df['date_start'] > latest_ged_date]
            
            # Combine the datasets
            combined_data = pd.concat([ged_df, candidate_df], ignore_index=True)
        else:
            combined_data = ged_df if not ged_df.empty else candidate_df
            
        if combined_data.empty:
            print("No data retrieved from either source")
            return pd.DataFrame()
            
        print(f"\nTotal records after combining: {len(combined_data)}")
        print(f"Date range: {combined_data['date_start'].min()} to {combined_data['date_start'].max()}")
        
        return self.process_fatalities_data(combined_data.to_dict('records'))

    def process_fatalities_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process fatalities data from UCDP GED API
        """
        if not data:
            print("No data to process")
            return pd.DataFrame()
        
        # Create records with fatality counts
        records = []
        for record in data:
            try:
                year = int(record['year'])
                deaths_a = int(record.get('deaths_a', 0) or 0)
                deaths_b = int(record.get('deaths_b', 0) or 0)
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


class RefugeeAnalyzer(UNHCRDataFinder):
    """
    A class to analyze refugee and IDP flows between countries using UNHCR data.
    """
    
    def get_refugee_flows(self, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get refugee origin and host country statistics for a specific year."""
        params = {
            'year': year,
            'limit': 1000,
            'coo_all': 'true',  # String 'true' instead of boolean True
            'coa_all': 'true'
        }
        
        data = self._fetch_paginated_data('population', params)
        if not data:
            return pd.DataFrame(), pd.DataFrame()
        
        # Print sample data for debugging
        sample_data = pd.DataFrame(data[:1])
        print(f"\nSample data columns: {sample_data.columns.tolist()}")
        
        # Create DataFrame and convert numeric columns
        df = pd.DataFrame(data)
        numeric_columns = ['refugees', 'asylum_seekers', 'idps', 'returned_idps', 
                         'stateless', 'ooc', 'oip']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Process origin countries (exclude summary rows)
        origin_stats = (df[
            # Keep rows with valid origin country
            (df['coo_iso'].notna()) & 
            (df['coo_iso'] != '-') & 
            # Exclude rows where both origin and asylum are the same
            (df['coo_iso'] != df['coa_iso'])
        ].groupby(['coo_iso', 'coo_name'], as_index=False)
        .agg({
            'refugees': 'sum',
            'asylum_seekers': 'sum',
            'idps': 'sum'
        })
        .rename(columns={
            'coo_iso': 'iso3',
            'coo_name': 'country',
            'refugees': 'refugees_originated',
            'asylum_seekers': 'asylum_seekers_originated',
            'idps': 'idps'
        }))
        
        # Process host countries (exclude summary rows)
        host_stats = (df[
            # Keep rows with valid asylum country
            (df['coa_iso'].notna()) & 
            (df['coa_iso'] != '-') & 
            # Exclude rows where both origin and asylum are the same
            (df['coo_iso'] != df['coa_iso'])
        ].groupby(['coa_iso', 'coa_name'], as_index=False)
        .agg({
            'refugees': 'sum',
            'asylum_seekers': 'sum'
        })
        .rename(columns={
            'coa_iso': 'iso3',
            'coa_name': 'country',
            'refugees': 'refugees_hosted',
            'asylum_seekers': 'asylum_seekers_hosted'
        }))
        
        print(f"\nFound {len(origin_stats)} origin countries and {len(host_stats)} host countries")
        
        # Print top 5 origin and host countries for verification
        print("\nTop 5 origin countries:")
        print(origin_stats.nlargest(5, 'refugees_originated')[['country', 'refugees_originated']])
        print("\nTop 5 host countries:")
        print(host_stats.nlargest(5, 'refugees_hosted')[['country', 'refugees_hosted']])
        
        return origin_stats, host_stats
    
    def create_displacement_summary(self, year: int) -> pd.DataFrame:
        """Create a comprehensive summary of displacement statistics by country."""
        try:
            # Get refugee and IDP statistics
            origin_stats, host_stats = self.get_refugee_flows(year)
            
            # Merge statistics
            summary = origin_stats.merge(host_stats, on=['iso3', 'country'], how='outer').fillna(0)
            
            # Add total displacement column
            summary['total_displacement'] = (
                summary['refugees_originated'] +
                summary['asylum_seekers_originated'] +
                summary['idps']
            )
            
            # Sort by total displacement
            summary = summary.sort_values('total_displacement', ascending=False)
            
            # Remove rows where iso3 is '-' or empty
            summary = summary[summary['iso3'].str.len() == 3].reset_index(drop=True)
            
            return summary
            
        except Exception as e:
            print(f"Error creating displacement summary for {year}: {str(e)}")
            return pd.DataFrame()
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics from the displacement data."""
        return {
            'total_refugees': df['refugees_originated'].sum(),
            'total_asylum_seekers': df['asylum_seekers_originated'].sum(),
            'total_idps': df['idps'].sum(),
            'total_displacement': df['total_displacement'].sum(),
            'num_origin_countries': df[df['refugees_originated'] > 0]['iso3'].nunique(),
            'num_host_countries': df[df['refugees_hosted'] > 0]['iso3'].nunique()
        }
    
    def analyze_historical_trends(self, start_year: int = 2010, end_year: int = 2022) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze historical trends in displacement statistics.
        """
        all_years_data = []
        yearly_stats = []
        
        print(f"\nAnalyzing displacement trends from {start_year} to {end_year}...")
        
        for year in range(start_year, end_year + 1):
            print(f"Processing year {year}...")
            summary = self.create_displacement_summary(year)
            
            if not summary.empty:
                # Get yearly statistics
                stats = self.get_summary_stats(summary)
                stats['year'] = year
                yearly_stats.append(stats)
                
                # Add year to detailed data
                summary['year'] = year
                all_years_data.append(summary)
        
        # Combine yearly statistics
        stats_df = pd.DataFrame(yearly_stats)
        
        # Combine all detailed data
        detailed_df = pd.concat(all_years_data, ignore_index=True) if all_years_data else pd.DataFrame()
        
        return stats_df, detailed_df
    
    def get_trend_analysis(self, stats_df: pd.DataFrame) -> Dict:
        """Calculate trends and changes in displacement statistics."""
        # Calculate year-over-year changes
        stats_df = stats_df.sort_values('year')
        
        for col in ['total_refugees', 'total_asylum_seekers', 'total_idps']:
            stats_df[f'{col}_yoy_change'] = stats_df[col].pct_change() * 100
        
        # Calculate compound annual growth rate (CAGR)
        years = stats_df['year'].max() - stats_df['year'].min()
        trends = {}
        
        for col in ['total_refugees', 'total_asylum_seekers', 'total_idps']:
            start_val = stats_df[col].iloc[0]
            end_val = stats_df[col].iloc[-1]
            if start_val > 0:
                cagr = (((end_val / start_val) ** (1/years)) - 1) * 100
            else:
                cagr = np.nan
            trends[f'{col}_cagr'] = cagr
            
            # Calculate total change
            total_change = ((end_val - start_val) / start_val * 100) if start_val > 0 else np.nan
            trends[f'{col}_total_change'] = total_change
        
        return trends
    
    def get_country_trends(self, detailed_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze trends for individual countries."""
        # Get the most recent year's data
        latest_year = detailed_df['year'].max()
        earliest_year = detailed_df['year'].min()
        
        # Calculate country-level trends
        country_trends = []
        
        for country in detailed_df['country'].unique():
            country_data = detailed_df[detailed_df['country'] == country]
            
            latest = country_data[country_data['year'] == latest_year].iloc[0] if not country_data[country_data['year'] == latest_year].empty else None
            earliest = country_data[country_data['year'] == earliest_year].iloc[0] if not country_data[country_data['year'] == earliest_year].empty else None
            
            if latest is not None and earliest is not None:
                trend = {
                    'country': country,
                    'iso3': latest['iso3'],
                    'refugees_change': ((latest['refugees_originated'] - earliest['refugees_originated']) / 
                                    earliest['refugees_originated'] * 100) if earliest['refugees_originated'] > 0 else np.nan,
                    'current_refugees': latest['refugees_originated'],
                    'current_idps': latest['idps'],
                    'hosting_change': ((latest['refugees_hosted'] - earliest['refugees_hosted']) /
                                    earliest['refugees_hosted'] * 100) if earliest['refugees_hosted'] > 0 else np.nan,
                    'current_hosting': latest['refugees_hosted']
                }
                country_trends.append(trend)
        
        return pd.DataFrame(country_trends)


class WorldBankAPI:
    """
    A class to interact with the World Bank API and retrieve various indicators and data.
    """
    
    def __init__(self):
        self.base_url = "http://api.worldbank.org/v2"
        self.format = "json"
        self._all_countries = None  # Cache for country codes
    
    def get_country_list(self) -> pd.DataFrame:
        """
        Retrieve a list of all available countries and their codes.
        
        Returns:
            pandas.DataFrame containing country information
        """
        url = f"{self.base_url}/country"
        params = {'format': self.format, 'per_page': 300}
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
            
        data = response.json()
        return pd.DataFrame(data[1])
    
    def _get_all_country_codes(self) -> List[str]:
        """
        Get a list of all country codes.
        
        Returns:
            List of country codes
        """
        if self._all_countries is None:
            df = self.get_country_list()
            # Filter out aggregates and regions, keep only countries
            self._all_countries = df[df['region'].notna()]['id'].tolist()
        return self._all_countries
    
    def get_indicator_data(
        self,
        country_code: Union[str, List[str], None] = None,
        indicator: str = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve data for a specific indicator and country/countries.
        
        Args:
            country_code: ISO country code(s) (e.g., 'US' for United States) or None for all countries
            indicator: World Bank indicator code (e.g., 'NY.GDP.MKTP.CD' for GDP)
            start_year: Starting year for the data (optional)
            end_year: Ending year for the data (optional)
            
        Returns:
            pandas.DataFrame containing the requested data
        """
        # Handle the case where country_code is None (all countries)
        if country_code is None:
            country_code = 'all'
            
        # Handle multiple country codes
        elif isinstance(country_code, list):
            country_code = ';'.join(country_code)
            
        # Build the URL
        url = f"{self.base_url}/country/{country_code}/indicator/{indicator}"
        
        # Parameters for the request
        params = {
            'format': self.format,
            'per_page': 1000
        }
        
        # Add date range if specified
        if start_year:
            params['date'] = f"{start_year}"
            if end_year:
                params['date'] = f"{start_year}:{end_year}"
        
        all_data = []
        page = 1
        while True:
            params['page'] = page
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}")
                
            data = response.json()
            
            # Check if we have data
            if len(data) < 2 or not data[1]:
                break
                
            all_data.extend(data[1])
            page += 1
            
            # Check if we've reached the last page
            if len(data[1]) < params['per_page']:
                break
        
        if not all_data:
            return pd.DataFrame()
            
        # Convert to DataFrame and clean up nested structures
        df = pd.DataFrame(all_data)
        
        # Extract values from nested dictionaries
        df['country'] = df['country'].apply(lambda x: x['value'])
        df['countryiso3code'] = df['countryiso3code'].astype(str)
        
        # Clean up the DataFrame
        df['date'] = pd.to_datetime(df['date'], format='%Y')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Select and rename columns
        df = df[['country', 'countryiso3code', 'date', 'value']].copy()
        df = df.rename(columns={
            'value': indicator,
            'countryiso3code': 'iso3'
        })
        
        return df

    def search_indicators(self, query: str) -> pd.DataFrame:
        """
        Search for indicators based on a keyword query.
        
        Args:
            query: Search term for indicators
            
        Returns:
            pandas.DataFrame containing matching indicators
        """
        url = f"{self.base_url}/indicator"
        params = {
            'format': self.format,
            'per_page': 1000,
            'search': query
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
            
        data = response.json()
        return pd.DataFrame(data[1])



class FSIDataProcessor:
    """A class to process Fragile States Index (FSI) data and create monthly country-level datasets."""
    
    def __init__(self, source_dir: str, output_dir: str):
        """Initialize the FSI data processor."""
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.cc = coco.CountryConverter()
        
        if not os.path.exists(self.source_dir):
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
    
    def _get_fsi_files(self) -> Dict[str, str]:
        """Get all FSI files from the source directory."""
        pattern = os.path.join(self.source_dir, "*.xlsx")
        files = glob.glob(pattern)
        
        if not files:
            raise ValueError(f"No Excel files found in {self.source_dir}")
            
        file_dict = {}
        for f in files:
            year = self._extract_year(f)
            if year:
                file_dict[year] = f
                
        if not file_dict:
            raise ValueError("No valid FSI files with year in filename found")
            
        return file_dict
    
    def _extract_year(self, filepath: str) -> Optional[str]:
        """Extract year from filename."""
        import re
        match = re.search(r'\d{4}', os.path.basename(filepath))
        return match.group(0) if match else None
    
    def _read_excel_file(self, filepath: str) -> pd.DataFrame:
        """Read FSI data from Excel file."""
        try:
            df = pd.read_excel(filepath, dtype=str)
            return df[['Country', 'Total']].copy()
        except Exception:
            return pd.DataFrame()
    
    def _convert_country_to_iso3(self, country_names: pd.Series) -> pd.Series:
        """Convert country names to ISO3 codes."""
        iso3_codes = []
        for name in country_names:
            try:
                # Special handling for Israel and West Bank
                if name == 'Israel and West Bank':
                    iso3_codes.append('ISR')
                else:
                    code = self.cc.convert(name, to='ISO3')
                    iso3_codes.append(code)
            except:
                iso3_codes.append(None)
        return pd.Series(iso3_codes)
    
    def _create_monthly_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create monthly dataset from annual FSI data."""
        if df.empty:
            raise ValueError("No data available to create monthly dataset")
            
        monthly_data = []
        for year in df['year'].unique():
            year_data = df[df['year'] == year].copy()
            for month in range(1, 13):
                month_data = year_data.copy()
                month_data['month'] = month
                month_data['date'] = pd.to_datetime(
                    month_data[['year', 'month']].assign(day=1)
                )
                monthly_data.append(month_data)
        
        return pd.concat(monthly_data, ignore_index=True)
    
    def process_fsi_data(self) -> pd.DataFrame:
        """Process FSI data files and create monthly country-level dataset."""
        fsi_files = self._get_fsi_files()
        
        dfs = []
        for year, filepath in fsi_files.items():
            df = self._read_excel_file(filepath)
            if not df.empty:
                df['year'] = int(year)
                df['FSI'] = pd.to_numeric(df['Total'], errors='coerce')
                df['iso3'] = self._convert_country_to_iso3(df['Country'])
                df = df.dropna(subset=['iso3'])
                df = df.drop(['Country', 'Total'], axis=1)
                dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        monthly_df = self._create_monthly_dataset(combined_df)
        return monthly_df.sort_values(['iso3', 'date'])
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = 'fsi_monthly.csv') -> bool:
        """Save processed data to CSV file if output directory exists."""
        if not os.path.exists(self.output_dir):
            return False
            
        try:
            output_path = os.path.join(self.output_dir, filename)
            df.to_csv(output_path, index=False)
            return True
        except:
            return False
        


class FragilityClassifier:
    """
    A class to process and manage country fragility classifications across years.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the FragilityClassifier with the data directory path."""
        self.data_dir = data_dir
        self.classifications = {}
        self._load_classifications()
    
    def _clean_rtf_content(self, content: str) -> str:
        """Clean RTF file content by removing RTF formatting."""
        # Remove RTF escape sequences
        content = content.replace('\\\'', "'")
        content = re.sub(r'\\[a-z]+', ' ', content)
        # Remove other RTF formatting
        content = content.replace('\\', '')
        return content
    
    def _extract_iso_codes(self, content: str, pattern: str) -> List[str]:
        """Extract ISO codes from the file content using the given pattern."""
        try:
            # Clean content first
            content = self._clean_rtf_content(content)
            # Look for the pattern in cleaned content
            match = re.search(pattern + r'\s*<-\s*c\((.*?)\)', content, re.DOTALL)
            if match:
                # Extract codes and clean them
                codes_str = match.group(1)
                # Extract quoted ISO codes
                codes = re.findall(r'"([A-Z]{3})"', codes_str)
                return codes
        except Exception as e:
            print(f"Error extracting {pattern}: {str(e)}")
        return []
    
    def _load_file(self, year: int) -> Optional[str]:
        """Load the content of a classification file for a given year."""
        filename = f"{year}_List.rtf"
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Successfully loaded {filename}")
                return content
        except FileNotFoundError:
            print(f"Warning: File not found for year {year}: {filepath}")
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
        return None
    
    def _load_classifications(self):
        """Load all classification files and process them."""
        print("Loading classifications...")
        # Process files for years 2016-2022
        for year in range(2016, 2023, 2):
            print(f"\nProcessing year {year}")
            content = self._load_file(year)
            if content:
                # Extract ISO codes for different categories
                extreme = self._extract_iso_codes(content, f"fragile_{year}_iso_extreme")
                not_extreme = self._extract_iso_codes(content, f"fragile_{year}_iso_notextreme")
                all_fragile = self._extract_iso_codes(content, f"fragile_{year}_iso_all")
                
                print(f"Found {len(extreme)} extreme, {len(not_extreme)} not extreme, {len(all_fragile)} total fragile countries")
                
                self.classifications[year] = {
                    'extreme': extreme,
                    'not_extreme': not_extreme,
                    'all': all_fragile
                }
    
    def get_classification(self, year: int) -> Dict[str, List[str]]:
        """
        Get the classification for a specific year.
        
        Args:
            year: The year to get classifications for
            
        Returns:
            Dictionary with 'extreme', 'not_extreme', and 'all' country lists
        """
        if year not in self.classifications:
            print(f"Warning: No classification data available for year {year}")
            return {'extreme': [], 'not_extreme': [], 'all': []}
        return self.classifications[year]
    
    def get_country_status(self, iso3: str, year: int) -> str:
        """
        Get the fragility status of a country for a specific year.
        
        Args:
            iso3: The ISO3 country code
            year: The year to check
            
        Returns:
            'extreme', 'fragile', or 'not_fragile'
        """
        if year not in self.classifications:
            return 'unknown'
            
        if iso3 in self.classifications[year]['extreme']:
            return 'extreme'
        elif iso3 in self.classifications[year]['not_extreme']:
            return 'fragile'
        return 'not_fragile'
    
    def get_all_countries(self) -> List[str]:
        """Get a list of all countries that have ever been classified as fragile."""
        all_countries = set()
        for year_data in self.classifications.values():
            all_countries.update(year_data['all'])
        return sorted(list(all_countries))
    
    def create_classification_panel(self) -> pd.DataFrame:
        """
        Create a panel dataset of classifications for all countries and years.
        
        Returns:
            DataFrame with columns: iso3, year, status
        """
        data = []
        countries = self.get_all_countries()
        
        for year in sorted(self.classifications.keys()):
            for iso3 in countries:
                status = self.get_country_status(iso3, year)
                data.append({
                    'iso3': iso3,
                    'year': year,
                    'status': status
                })
        
        df = pd.DataFrame(data)
        # Convert status to categorical with specific order
        df['status'] = pd.Categorical(df['status'], 
                                    categories=['extreme', 'fragile', 'not_fragile', 'unknown'],
                                    ordered=True)
        return df
    


import requests
import pandas as pd
import time
import socket
import urllib3
import os
import sys
import logging
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('un_population_api.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class UNPopulationDataPortalAPI:
    def __init__(self, base_url="https://population.un.org/dataportalapi/api/v1"):
        """
        Initialize the UN Population Data Portal API client
        
        :param base_url: Base URL for the API
        """
        self.base_url = base_url
        
        # Disable SSL warnings if needed
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Session for persistent connection
        self.session = requests.Session()
        
        # Detailed logging of initialization
        logger.info(f"Initialized UN Population Data Portal API with base URL: {base_url}")
    
    def _resolve_url(self, url):
        """
        Carefully resolve potentially problematic URLs
        
        :param url: URL to resolve
        :return: Resolved URL
        """
        # Parse the original URL
        parsed_url = urlparse(url)
        
        # If URL is to an internal address, reconstruct using base URL
        if parsed_url.netloc in ['10.208.38.29', '127.0.0.1', 'localhost']:
            # Extract query parameters
            query_params = parse_qs(parsed_url.query)
            
            # Reconstruct URL using original base URL
            resolved_url = urljoin(self.base_url, parsed_url.path)
            
            # Add back original query parameters
            if query_params:
                resolved_url += f"?{urlencode(query_params, doseq=True)}"
            
            logger.info(f"Resolved internal URL: {url} -> {resolved_url}")
            return resolved_url
        
        # If URL seems to be an incorrect domain, rebuild with original base
        if 'www.un.org' in parsed_url.netloc or 'un.org' in parsed_url.netloc:
            # Extract the path and query from the problematic URL
            path = parsed_url.path.replace('/development/desa/pd', '/dataportalapi')
            query_params = parse_qs(parsed_url.query)
            
            # Reconstruct URL using base domain and corrected path
            resolved_url = f"https://population.un.org{path}"
            
            # Add back original query parameters
            if query_params:
                resolved_url += f"?{urlencode(query_params, doseq=True)}"
            
            logger.info(f"Corrected domain URL: {url} -> {resolved_url}")
            return resolved_url
        
        return url
    
    def _make_request(self, method, url, **kwargs):
        """
        Centralized request method with comprehensive error handling
        
        :param method: HTTP method (get, post, etc.)
        :param url: Target URL
        :param kwargs: Additional arguments for requests
        :return: Response object
        """
        # Resolve any problematic URLs
        url = self._resolve_url(url)
        
        # Default request parameters
        default_kwargs = {
            'timeout': 30,
            'verify': False,  # Disable SSL verification for testing
        }
        
        # Update with any user-provided kwargs
        default_kwargs.update(kwargs)
        
        try:
            # Make the request using the session
            logger.info(f"Making {method.upper()} request to {url}")
            response = getattr(self.session, method)(url, **default_kwargs)
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            return response
        
        except requests.exceptions.RequestException as e:
            # Detailed logging of request errors
            logger.error(f"Request failed: {e}")
            logger.error(f"Request details - Method: {method.upper()}, URL: {url}")
            
            # Specific error handling
            if isinstance(e, requests.exceptions.ConnectionError):
                logger.error("Connection Error: Unable to connect to the server")
            elif isinstance(e, requests.exceptions.Timeout):
                logger.error("Timeout Error: The request timed out")
            elif isinstance(e, requests.exceptions.HTTPError):
                logger.error(f"HTTP Error: {e.response.status_code} - {e.response.reason}")
            
            raise
    
    def test_api_connection(self):
        """
        Test basic API connectivity with detailed diagnostics
        
        :return: Boolean indicating connection success
        """
        try:
            logger.info("Testing API Connection...")
            
            # Test endpoint
            response = self._make_request('get', f"{self.base_url}/indicators")
            
            logger.info(f"API Connection Successful. Response Status: {response.status_code}")
            return True
        
        except Exception as e:
            logger.error(f"API Connection Failed: {e}")
            return False
    
    def get_country_ids(self, max_retries=3, page_size=500):
        """
        Retrieve IDs for all countries with robust error handling
        
        :param max_retries: Maximum number of retry attempts
        :param page_size: Number of results per page
        :return: List of country IDs
        """
        country_ids = []
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to retrieve country IDs (Attempt {attempt + 1})...")
                
                # Initial request with custom page size
                response = self._make_request(
                    'get', 
                    f"{self.base_url}/locations",
                    params={'pageSize': page_size}
                )
                
                # Parse response
                data = response.json()
                
                # Collect country IDs (locationTypeId 4 represents countries)
                batch_country_ids = [
                    location['id'] for location in data['data'] 
                    if location.get('locationTypeId') == 4
                ]
                country_ids.extend(batch_country_ids)
                
                # Handle pagination
                current_page = 1
                while data.get('nextPage'):
                    try:
                        current_page += 1
                        logger.info(f"Fetching page {current_page}")
                        
                        # Resolve and use nextPage URL
                        next_url = self._resolve_url(data['nextPage'])
                        response = self._make_request('get', next_url)
                        
                        data = response.json()
                        
                        batch_country_ids = [
                            location['id'] for location in data['data'] 
                            if location.get('locationTypeId') == 4
                        ]
                        country_ids.extend(batch_country_ids)
                    
                    except Exception as page_error:
                        logger.error(f"Error fetching page {current_page}: {page_error}")
                        break
                
                logger.info(f"Successfully retrieved {len(country_ids)} country IDs")
                return country_ids
            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                # Wait before retrying with exponential backoff
                time.sleep(2 ** attempt)
        
        raise Exception("Failed to retrieve country IDs after multiple attempts")
    
    def get_population_data(self, start_year=1950, end_year=2050, indicator_id=36):
        """
        Retrieve population estimation data for countries
        
        :param start_year: Start year for data retrieval
        :param end_year: End year for data retrieval
        :param indicator_id: Indicator ID for population (default 36 for total population)
        :return: DataFrame of population data
        """
        # Get country IDs
        country_ids = self.get_country_ids()
        
        # Prepare to collect data
        all_population_data = []
        
        # Batch processing
        batch_size = 5
        for i in range(0, len(country_ids), batch_size):
            batch_ids = country_ids[i:i+batch_size]
            location_ids = ','.join(map(str, batch_ids))
            
            try:
                # Construct API target URL
                target = (f"{self.base_url}/data/indicators/{indicator_id}/"
                          f"locations/{location_ids}/"
                          f"start/{start_year}/end/{end_year}")
                
                logger.info(f"Processing batch {i//batch_size + 1}")
                
                # Make request
                response = self._make_request('get', target)
                
                # Parse response
                data = response.json()
                
                # Collect data
                all_population_data.extend(data['data'])
                
                # Handle pagination
                while data.get('nextPage'):
                    try:
                        # Resolve and use nextPage URL
                        next_url = self._resolve_url(data['nextPage'])
                        response = self._make_request('get', next_url)
                        
                        data = response.json()
                        all_population_data.extend(data['data'])
                    except Exception as page_error:
                        logger.error(f"Error fetching additional pages: {page_error}")
                        break
                
                # Short pause between batches
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_population_data)
        
        return df
    
