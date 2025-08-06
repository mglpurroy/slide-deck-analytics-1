#!/usr/bin/env python3
"""
Simplified script to fix map rendering issues in the FCV Analytics Notebook.
Focuses on path fixes and link improvements without requiring external dependencies.
"""

import json
import re
from pathlib import Path

def fix_notebook_maps():
    """Fix map rendering issues in the main notebook."""
    notebook_path = Path("_sources/notebooks/main.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    print("🔍 Reading notebook...")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print("🔧 Analyzing and fixing map rendering issues...")
    
    fixes_applied = 0
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix 1: Update fragility plot save path
            if 'interactive_fragility_plot.html' in source and '_build' in source:
                print(f"🔧 Fixing fragility plot paths in cell {i}")
                
                # Find and replace the path construction
                if "build_dir = os.path.join('_build', 'html')" in source:
                    # Replace the entire path section
                    new_source = re.sub(
                        r"        # Create _build/html directory if it doesn't exist\s*\n"
                        r"        build_dir = os\.path\.join\('_build', 'html'\)\s*\n"
                        r"        os\.makedirs\(build_dir, exist_ok=True\)\s*\n"
                        r"        \s*\n"
                        r"        # Save the file\s*\n"
                        r"        html_filename = 'interactive_fragility_plot\.html'\s*\n"
                        r"        html_path = os\.path\.join\(build_dir, html_filename\)\s*\n"
                        r"        interactive_fig\.write_html\(html_path\)",
                        """        # Save interactive plot for Jupyter Book
        html_filename = 'interactive_fragility_plot.html'
        
        # Save to current directory for notebook development
        interactive_fig.write_html(html_filename)
        print(f"💾 Saved interactive plot: {html_filename}")
        
        # Also try to save to build directory if it exists
        try:
            from pathlib import Path
            build_dir = Path('_build/html')
            if build_dir.exists():
                build_path = build_dir / html_filename
                interactive_fig.write_html(str(build_path))
                print(f"💾 Also saved to: {build_path}")
        except:
            pass""",
                        source,
                        flags=re.MULTILINE
                    )
                    
                    if new_source != source:
                        cell['source'] = new_source.split('\n')
                        fixes_applied += 1
            
            # Fix 2: Update FCS map save path  
            elif 'interactive_fcs_map.html' in source and 'save_dirs' in source:
                print(f"🔧 Fixing FCS map paths in cell {i}")
                
                # Replace the save_dirs logic
                new_source = re.sub(
                    r"        save_dirs = \[\s*\n"
                    r"            os\.path\.join\('docs', '_build', 'html'\),\s*\n"
                    r"            os\.path\.join\('docs', 'notebooks'\)\s*\n"
                    r"        \]\s*\n"
                    r"        \s*\n"
                    r"        html_filename = 'interactive_fcs_map\.html'\s*\n"
                    r"        \s*\n"
                    r"        # Save to each directory\s*\n"
                    r"        for dir_path in save_dirs:\s*\n"
                    r"            # Create directory if it doesn't exist\s*\n"
                    r"            os\.makedirs\(dir_path, exist_ok=True\)\s*\n"
                    r"            \s*\n"
                    r"            # Save the file\s*\n"
                    r"            html_path = os\.path\.join\(dir_path, html_filename\)\s*\n"
                    r"            interactive_fig\.write_html\(html_path\)",
                    """        # Save interactive FCS map for Jupyter Book
        html_filename = 'interactive_fcs_map.html'
        
        # Save to current directory for notebook development
        interactive_fig.write_html(html_filename)
        print(f"💾 Saved interactive FCS map: {html_filename}")
        
        # Also try to save to build directory if it exists
        try:
            from pathlib import Path
            build_dir = Path('_build/html')
            if build_dir.exists():
                build_path = build_dir / html_filename
                interactive_fig.write_html(str(build_path))
                print(f"💾 Also saved to: {build_path}")
        except:
            pass""",
                    source,
                    flags=re.MULTILINE
                )
                
                if new_source != source:
                    cell['source'] = new_source.split('\n')
                    fixes_applied += 1
        
        elif cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            
            # Fix 3: Update markdown links for fragility plot
            if '[Click here for interactive version](interactive_fragility_plot.html)' in source:
                print(f"🔧 Fixing fragility plot link in cell {i}")
                
                new_content = """```{note}
**Interactive Fragility Plot Available**

The interactive version allows you to explore the States of Fragility 2022 data with hover details and zooming capabilities.

<a href="interactive_fragility_plot.html" target="_blank" style="display: inline-block; padding: 8px 16px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; margin: 4px 0;">🗺️ Open Interactive Fragility Plot</a>
```"""
                
                cell['source'] = [new_content]
                fixes_applied += 1
            
            # Fix 4: Update markdown links for FCS map
            elif '[Click here for interactive FCS map](interactive_fcs_map.html)' in source:
                print(f"🔧 Fixing FCS map link in cell {i}")
                
                new_content = """```{note}
**Interactive FCS Map Available**

Explore the interactive FCS (Fragile and Conflict-affected Situations) map with detailed country classifications and information.

<a href="interactive_fcs_map.html" target="_blank" style="display: inline-block; padding: 8px 16px; background-color: #28a745; color: white; text-decoration: none; border-radius: 4px; margin: 4px 0;">🌍 Open Interactive FCS Map</a>
```"""
                
                cell['source'] = [new_content]
                fixes_applied += 1
    
    # Write the fixed notebook back
    print("💾 Saving fixed notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Applied {fixes_applied} fixes to the notebook")
    return fixes_applied > 0

def update_jupyter_book_config():
    """Update Jupyter Book configuration to better handle interactive content."""
    config_path = Path("_config.yml")
    
    if not config_path.exists():
        print("⚠️ _config.yml not found, skipping config updates")
        return
    
    print("🔧 Updating Jupyter Book configuration for better map support...")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Add HTML file copying configuration
    html_config = """
# Additional configuration for interactive HTML files
sphinx:
  config:
    html_static_path: ['_static']
    html_extra_path: ['_sources/notebooks/*.html']
    html_copy_source: false
    html_show_sourcelink: false
"""
    
    # Check if we need to add the HTML config
    if 'html_extra_path' not in content:
        content += html_config
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("✅ Updated _config.yml to include HTML file handling")
    else:
        print("✅ _config.yml already configured for HTML files")

def main():
    """Main function to fix all map rendering issues."""
    print("🗺️ Fixing Map Rendering Issues in FCV Analytics Notebook")
    print("=" * 60)
    
    # Step 1: Fix notebook code
    notebook_fixed = fix_notebook_maps()
    
    # Step 2: Update Jupyter Book configuration
    update_jupyter_book_config()
    
    if notebook_fixed:
        print("\n✅ Map rendering fixes applied successfully!")
        print("\n📋 What was fixed:")
        print("• ✅ Updated file paths to work with Jupyter Book structure")
        print("• ✅ Simplified HTML file saving to current directory")
        print("• ✅ Improved interactive map links with better styling")
        print("• ✅ Added fallback path handling for build directories")
        print("• ✅ Updated Jupyter Book configuration for HTML files")
        
        print("\n🚀 Next steps:")
        print("1. The maps should now render properly in the built book")
        print("2. Interactive HTML files will be saved alongside the notebook")
        print("3. Links will have better styling and accessibility")
        print("4. Test by building the book: jupyter-book build _sources")
    else:
        print("\n❌ No fixes were applied. The notebook may already be correct.")
        print("If maps are still not rendering, please check:")
        print("• Data files are available in the expected locations")
        print("• Plotly is installed and working properly")
        print("• No errors occur during notebook execution")
    
    return notebook_fixed

if __name__ == "__main__":
    main()