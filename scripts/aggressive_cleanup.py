#!/usr/bin/env python3
"""
Aggressive cleanup to remove all useless files and build artifacts
"""

import os
import shutil
import glob
from pathlib import Path

def aggressive_cleanup():
    """Remove all build artifacts and useless files"""
    
    print("🧹 AGGRESSIVE CLEANUP - REMOVING USELESS FILES")
    print("=" * 50)
    
    # Files and directories that should NOT be in the root
    useless_items = [
        # Build artifacts
        "genindex.html",
        "index.html", 
        "search.html",
        "searchindex.js",
        "objects.inv",
        ".buildinfo",
        "README.html",
        ".nojekyll",
        
        # Duplicate config files (should only be in docs/)
        "_config.yml",
        "_toc.yml",
        
        # Build directories
        "_static/",
        "_sphinx_design_static/",
        "_images/",
        "notebooks/",  # This should only be in docs/
        
        # Other artifacts
        "DEPLOYMENT_SUMMARY.md",  # Not needed anymore
        
        # Any remaining cache/temp files
        "*.pyc",
        "*.pyo",
        "__pycache__/",
        ".pytest_cache/",
        "*.tmp",
        "*.temp",
        "*~",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    cleaned = 0
    for item in useless_items:
        if "*" in item:
            # Handle glob patterns
            for file_path in glob.glob(item, recursive=True):
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
        else:
            if os.path.exists(item):
                try:
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                        print(f"🗑️  Removed directory: {item}")
                    else:
                        os.remove(item)
                        print(f"🗑️  Removed file: {item}")
                    cleaned += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {item}: {e}")
    
    print(f"\n✅ Removed {cleaned} useless items!")
    
    # Show what should remain in root
    print("\n📁 WHAT SHOULD REMAIN IN ROOT:")
    print("=" * 30)
    
    essential_root_files = [
        "README.md",
        "requirements.txt", 
        ".gitignore",
        "docs/",
        "data/",
        "scripts/",
        ".github/"
    ]
    
    print("✅ Essential files/directories:")
    for item in essential_root_files:
        if os.path.exists(item):
            print(f"   ✅ {item}")
        else:
            print(f"   ❌ {item} (MISSING!)")
    
    # Check what's actually in root now
    print("\n📂 CURRENT ROOT CONTENTS:")
    print("=" * 25)
    
    root_items = []
    for item in os.listdir("."):
        if not item.startswith('.git'):  # Ignore .git directory
            root_items.append(item)
    
    for item in sorted(root_items):
        if item in ["README.md", "requirements.txt", ".gitignore", "docs", "data", "scripts", ".github"]:
            print(f"   ✅ {item} (essential)")
        else:
            print(f"   ⚠️  {item} (questionable)")
    
    print(f"\n🎯 ROOT DIRECTORY CLEANUP COMPLETE!")
    print(f"Removed {cleaned} useless files/directories")
    
    return True

if __name__ == "__main__":
    aggressive_cleanup()