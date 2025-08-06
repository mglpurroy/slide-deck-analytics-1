#!/usr/bin/env python3
"""
Final cleanup and polish of the main branch
"""

import os
import json
import shutil
import glob
from pathlib import Path

def final_cleanup():
    """Perform final cleanup and optimization of main branch"""
    
    print("🎯 FINAL CLEANUP AND POLISH OF MAIN BRANCH")
    print("=" * 50)
    
    # 1. Remove any remaining unnecessary files
    print("🧹 Removing any remaining unnecessary files...")
    
    cleanup_patterns = [
        "*.tmp",
        "*.temp",
        "*~",
        "*.bak",
        ".DS_Store",
        "Thumbs.db",
        "*.log",
        "*.pyc",
        "*.pyo",
        "__pycache__/*",
        ".pytest_cache/*",
        ".ipynb_checkpoints/*",
        "*/.ipynb_checkpoints/*",
        "docs/.ipynb_checkpoints/*",
        "docs/notebooks/.ipynb_checkpoints/*"
    ]
    
    cleaned = 0
    for pattern in cleanup_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"🗑️  Removed directory: {file_path}")
                else:
                    os.remove(file_path)
                    print(f"🗑️  Removed file: {file_path}")
                cleaned += 1
            except Exception as e:
                print(f"⚠️  Could not remove {file_path}: {e}")
    
    # 2. Optimize notebook metadata
    print("\n🔧 Optimizing notebook metadata...")
    
    notebook_path = "docs/notebooks/main.ipynb"
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Clean up notebook metadata
    if 'metadata' in notebook:
        # Remove unnecessary metadata
        unnecessary_keys = ['orig_nbformat', 'vscode']
        for key in unnecessary_keys:
            if key in notebook['metadata']:
                del notebook['metadata'][key]
                print(f"🧹 Removed {key} from notebook metadata")
    
    # Ensure all code cells have hide-input tags
    hidden_cells = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            if 'metadata' not in cell:
                cell['metadata'] = {}
            if 'tags' not in cell['metadata']:
                cell['metadata']['tags'] = []
            if 'hide-input' not in cell['metadata']['tags']:
                cell['metadata']['tags'].append('hide-input')
                hidden_cells += 1
    
    # Save optimized notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ Optimized notebook metadata and ensured {hidden_cells} cells are hidden")
    
    # 3. Validate final structure
    print("\n🔍 Validating final repository structure...")
    
    essential_files = {
        "docs/notebooks/main.ipynb": "Main analysis notebook",
        "docs/_config.yml": "Jupyter Book configuration",
        "docs/_toc.yml": "Table of contents",
        "docs/index.md": "Landing page",
        "requirements.txt": "Python dependencies",
        "README.md": "Project documentation",
        ".github/workflows/deploy.yml": "Deployment workflow",
        "data/": "Data directory"
    }
    
    missing_files = []
    for file_path, description in essential_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_path} ({description})")
        else:
            print(f"✅ {file_path} - {description}")
    
    if missing_files:
        print(f"❌ Missing essential files: {missing_files}")
        return False
    
    # 4. Count and validate data files
    data_files = list(Path("data").glob("*.xlsx")) + list(Path("data").glob("*.csv"))
    data_size = sum(f.stat().st_size for f in data_files) / (1024 * 1024)  # MB
    
    print(f"📊 Data validation:")
    print(f"   - {len(data_files)} data files found")
    print(f"   - Total size: {data_size:.1f} MB")
    
    for data_file in data_files:
        print(f"   ✅ {data_file.name} ({data_file.stat().st_size / (1024*1024):.1f} MB)")
    
    # 5. Check scripts directory
    script_files = list(Path("scripts").glob("*.py"))
    print(f"🔧 Scripts: {len(script_files)} utility scripts available")
    
    # 6. Final summary
    print(f"\n🎉 FINAL CLEANUP COMPLETE!")
    print("=" * 50)
    print(f"✅ Cleaned {cleaned} unnecessary files")
    print(f"✅ Optimized notebook metadata")
    print(f"✅ Validated all essential files present")
    print(f"✅ Confirmed {len(data_files)} data files ({data_size:.1f} MB)")
    print(f"✅ Repository structure optimized")
    print(f"✅ Main branch is clean and ready")
    
    # 7. Repository statistics
    print(f"\n📈 REPOSITORY STATISTICS:")
    print("=" * 50)
    
    # Count files by type
    all_files = list(Path(".").rglob("*"))
    file_types = {}
    total_size = 0
    
    for file_path in all_files:
        if file_path.is_file() and not any(part.startswith('.git') for part in file_path.parts):
            ext = file_path.suffix.lower() or 'no-extension'
            if ext not in file_types:
                file_types[ext] = {'count': 0, 'size': 0}
            file_types[ext]['count'] += 1
            try:
                size = file_path.stat().st_size
                file_types[ext]['size'] += size
                total_size += size
            except:
                pass
    
    print(f"📁 Total repository size: {total_size / (1024*1024):.1f} MB")
    print(f"📄 File breakdown:")
    
    for ext, info in sorted(file_types.items(), key=lambda x: x[1]['size'], reverse=True)[:10]:
        size_mb = info['size'] / (1024*1024)
        print(f"   {ext}: {info['count']} files ({size_mb:.1f} MB)")
    
    print(f"\n🚀 MAIN BRANCH READY FOR DEPLOYMENT!")
    return True

if __name__ == "__main__":
    final_cleanup()