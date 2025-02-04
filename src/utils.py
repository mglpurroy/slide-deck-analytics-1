"""
UCDP data processing utilities
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any

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
    def get_conflict_colors() -> Dict[str, str]:
        """Get color mapping for conflict types"""
        return {
            'Extra-systemic conflicts': '#2ecc71',
            'Inter-state conflicts': '#3498db',
            'Internal conflicts': '#f1c40f',
            'Internationalized-internal conflicts': '#9b59b6'
        }