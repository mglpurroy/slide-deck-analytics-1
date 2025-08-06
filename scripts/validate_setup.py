#!/usr/bin/env python3
"""
Validation script to check if the FCV Analytics repository is properly configured
for Jupyter Book deployment.
"""

import json
import sys
from pathlib import Path
import yaml

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} MISSING: {filepath}")
        return False

def validate_config_file():
    """Validate _config.yml file."""
    config_path = Path("_config.yml")
    if not config_path.exists():
        print("❌ _config.yml file missing")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['title', 'author', 'execute', 'repository']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ _config.yml missing required keys: {missing_keys}")
            return False
        
        print("✅ _config.yml is properly configured")
        return True
    except Exception as e:
        print(f"❌ Error reading _config.yml: {e}")
        return False

def validate_toc_file():
    """Validate _toc.yml file."""
    toc_path = Path("_toc.yml")
    if not toc_path.exists():
        print("❌ _toc.yml file missing")
        return False
    
    try:
        with open(toc_path, 'r') as f:
            toc = yaml.safe_load(f)
        
        if 'root' not in toc:
            print("❌ _toc.yml missing 'root' key")
            return False
        
        print("✅ _toc.yml is properly configured")
        return True
    except Exception as e:
        print(f"❌ Error reading _toc.yml: {e}")
        return False

def validate_notebook():
    """Validate the main notebook exists and has proper structure."""
    notebook_path = Path("_sources/notebooks/main.ipynb")
    if not notebook_path.exists():
        print("❌ Main notebook missing: _sources/notebooks/main.ipynb")
        return False
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        if 'cells' not in notebook:
            print("❌ Notebook has invalid structure (missing cells)")
            return False
        
        cell_count = len(notebook['cells'])
        print(f"✅ Main notebook found with {cell_count} cells")
        return True
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return False

def validate_github_actions():
    """Validate GitHub Actions workflow."""
    workflow_path = Path(".github/workflows/deploy.yml")
    if not workflow_path.exists():
        print("❌ GitHub Actions workflow missing")
        return False
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        if 'jobs' not in workflow or 'build' not in workflow['jobs']:
            print("❌ GitHub Actions workflow has invalid structure")
            return False
        
        print("✅ GitHub Actions workflow is properly configured")
        return True
    except Exception as e:
        print(f"❌ Error reading GitHub Actions workflow: {e}")
        return False

def main():
    """Main validation function."""
    print("🔍 Validating FCV Analytics Repository Setup...")
    print("=" * 50)
    
    checks = [
        # Core files
        (lambda: check_file_exists("_config.yml", "Jupyter Book config"), "Config file"),
        (lambda: check_file_exists("_toc.yml", "Table of contents"), "TOC file"),
        (lambda: check_file_exists("requirements.txt", "Dependencies"), "Requirements"),
        (lambda: check_file_exists("README.md", "Documentation"), "README"),
        (lambda: check_file_exists(".gitignore", "Git ignore rules"), "Gitignore"),
        
        # Source files
        (lambda: check_file_exists("_sources/index.md", "Main index page"), "Index page"),
        (lambda: check_file_exists("_sources/notebooks/main.ipynb", "Main notebook"), "Notebook"),
        
        # Scripts
        (lambda: check_file_exists("scripts/build_book.sh", "Build script"), "Build script"),
        (lambda: check_file_exists("scripts/optimize_notebook.py", "Optimization script"), "Optimization"),
        
        # GitHub Actions
        (lambda: check_file_exists(".github/workflows/deploy.yml", "Deployment workflow"), "Workflow"),
        
        # Validation functions
        (validate_config_file, "Config validation"),
        (validate_toc_file, "TOC validation"),
        (validate_notebook, "Notebook validation"),
        (validate_github_actions, "Workflow validation"),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_func, description in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {description} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Repository is properly configured for deployment!")
        print("\n📋 Next steps:")
        print("1. Push changes to main branch to trigger deployment")
        print("2. Check GitHub Actions tab for build status")
        print("3. Visit the GitHub Pages URL when deployment completes")
        return True
    else:
        print("⚠️  Some issues need to be addressed before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)