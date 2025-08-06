#!/usr/bin/env python3
"""
Add hide-input tags to code cells in the notebook to hide code and show only outputs
"""

import json
import os

def hide_code_cells():
    """Add hide-input tags to code cells in the notebook"""
    
    notebook_path = "docs/notebooks/main.ipynb"
    
    print("🔧 Adding hide-input tags to notebook cells...")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells_modified = 0
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            # Initialize metadata if it doesn't exist
            if 'metadata' not in cell:
                cell['metadata'] = {}
            
            # Initialize tags if they don't exist
            if 'tags' not in cell['metadata']:
                cell['metadata']['tags'] = []
            
            # Add hide-input tag if not already present
            if 'hide-input' not in cell['metadata']['tags']:
                cell['metadata']['tags'].append('hide-input')
                cells_modified += 1
                print(f"📝 Added hide-input tag to cell {i}")
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Added hide-input tags to {cells_modified} code cells!")
    print("📖 Code will be hidden in Jupyter Book, but outputs/plots will be visible")
    return True

def show_code_cells():
    """Remove hide-input tags from code cells to show code"""
    
    notebook_path = "docs/notebooks/main.ipynb"
    
    print("🔧 Removing hide-input tags from notebook cells...")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells_modified = 0
    
    # Process each cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            if 'metadata' in cell and 'tags' in cell['metadata']:
                if 'hide-input' in cell['metadata']['tags']:
                    cell['metadata']['tags'].remove('hide-input')
                    cells_modified += 1
                    print(f"📝 Removed hide-input tag from cell {i}")
                
                # Clean up empty tags list
                if not cell['metadata']['tags']:
                    del cell['metadata']['tags']
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Removed hide-input tags from {cells_modified} code cells!")
    print("📖 Code will be visible in Jupyter Book")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show_code_cells()
    else:
        hide_code_cells()