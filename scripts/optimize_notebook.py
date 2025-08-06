#!/usr/bin/env python3
"""
Script to optimize the main notebook for better performance and timeout handling.
"""

import json
import sys
from pathlib import Path

def optimize_notebook(notebook_path):
    """
    Optimize notebook cells for better performance and timeout handling.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    optimized_cells = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            # Add timeout handling and optimization tags
            source = cell['source']
            
            # Check if this is a long-running cell that might need optimization
            if any(keyword in ''.join(source) for keyword in [
                'comprehensive_data', 'requests.get', 'time.sleep', 
                'for i in range', 'while ', 'api.', 'download'
            ]):
                # Add timeout handling and caching suggestions
                if 'metadata' not in cell:
                    cell['metadata'] = {}
                
                # Add tags for long-running cells
                if 'tags' not in cell['metadata']:
                    cell['metadata']['tags'] = []
                
                if 'hide-input' not in cell['metadata']['tags']:
                    cell['metadata']['tags'].append('hide-input')
                
                # Add execution timeout metadata
                cell['metadata']['execution'] = {
                    'timeout': 300,  # 5 minutes per cell max
                    'allow_errors': False
                }
                
                # Add caching suggestion comment at the beginning
                cache_comment = [
                    "# Performance optimization: Consider caching results\n",
                    "# import pickle\n",
                    "# cache_file = 'cache/cell_{}_cache.pkl'\n".format(i),
                    "# if Path(cache_file).exists():\n",
                    "#     with open(cache_file, 'rb') as f:\n",
                    "#         cached_result = pickle.load(f)\n",
                    "# else:\n",
                    "#     # Original code here\n",
                    "#     # ... your computation ...\n",
                    "#     # Save to cache\n",
                    "#     Path(cache_file).parent.mkdir(exist_ok=True)\n",
                    "#     with open(cache_file, 'wb') as f:\n",
                    "#         pickle.dump(result, f)\n",
                    "\n"
                ]
                
                # Only add cache comment if not already present
                if not any('cache' in line.lower() for line in source[:5]):
                    cell['source'] = cache_comment + source
        
        optimized_cells.append(cell)
    
    notebook['cells'] = optimized_cells
    
    # Update notebook metadata
    if 'metadata' not in notebook:
        notebook['metadata'] = {}
    
    notebook['metadata']['jupyterbook'] = {
        'execution_timeout': 1200,  # 20 minutes total
        'execution_allow_errors': False
    }
    
    # Write optimized notebook
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    notebook_path.rename(backup_path)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Optimized notebook saved to {notebook_path}")
    print(f"Original backup saved to {backup_path}")
    
    return notebook

def create_cache_directory():
    """Create cache directory for storing intermediate results."""
    cache_dir = Path('_sources/notebooks/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create .gitignore for cache directory
    gitignore_path = cache_dir / '.gitignore'
    with open(gitignore_path, 'w') as f:
        f.write("# Cache files - ignore all contents\n*\n!.gitignore\n")
    
    print(f"Created cache directory: {cache_dir}")

if __name__ == "__main__":
    notebook_path = Path("_sources/notebooks/main.ipynb")
    
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        sys.exit(1)
    
    create_cache_directory()
    optimize_notebook(notebook_path)
    print("Notebook optimization complete!")