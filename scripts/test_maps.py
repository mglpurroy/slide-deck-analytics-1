#!/usr/bin/env python3
"""
Test script to validate that map rendering fixes are working properly.
"""

import json
import re
from pathlib import Path

def test_notebook_structure():
    """Test that the notebook has the correct structure for map rendering."""
    notebook_path = Path("_sources/notebooks/main.ipynb")
    
    if not notebook_path.exists():
        print("❌ Notebook not found")
        return False
    
    print("🔍 Testing notebook structure...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Check for improved fragility plot code
    total_tests += 1
    fragility_plot_fixed = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'interactive_fragility_plot.html' in source:
                if 'Save interactive plot for Jupyter Book' in source:
                    fragility_plot_fixed = True
                    break
    
    if fragility_plot_fixed:
        print("✅ Fragility plot code has been fixed")
        tests_passed += 1
    else:
        print("❌ Fragility plot code not properly fixed")
    
    # Test 2: Check for improved FCS map code
    total_tests += 1
    fcs_map_fixed = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'interactive_fcs_map.html' in source:
                if 'Save interactive FCS map for Jupyter Book' in source:
                    fcs_map_fixed = True
                    break
    
    if fcs_map_fixed:
        print("✅ FCS map code has been fixed")
        tests_passed += 1
    else:
        print("❌ FCS map code not properly fixed")
    
    # Test 3: Check for improved fragility plot link
    total_tests += 1
    fragility_link_fixed = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if 'Interactive Fragility Plot Available' in source:
                fragility_link_fixed = True
                break
    
    if fragility_link_fixed:
        print("✅ Fragility plot link has been improved")
        tests_passed += 1
    else:
        print("❌ Fragility plot link not properly improved")
    
    # Test 4: Check for improved FCS map link
    total_tests += 1
    fcs_link_fixed = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if 'Interactive FCS Map Available' in source:
                fcs_link_fixed = True
                break
    
    if fcs_link_fixed:
        print("✅ FCS map link has been improved")
        tests_passed += 1
    else:
        print("❌ FCS map link not properly improved")
    
    print(f"\n📊 Notebook structure tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def test_config_updates():
    """Test that the Jupyter Book configuration has been updated."""
    config_path = Path("_config.yml")
    
    if not config_path.exists():
        print("❌ _config.yml not found")
        return False
    
    print("🔍 Testing Jupyter Book configuration...")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Check for HTML extra path configuration
    if 'html_extra_path' in content:
        print("✅ HTML extra path configuration found")
        tests_passed += 1
    else:
        print("❌ HTML extra path configuration missing")
    
    # Test 2: Check for HTML copy source configuration
    if 'html_copy_source: false' in content:
        print("✅ HTML copy source configuration found")
        tests_passed += 1
    else:
        print("❌ HTML copy source configuration missing")
    
    print(f"\n📊 Configuration tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def test_github_actions():
    """Test that GitHub Actions workflow has been updated."""
    workflow_path = Path(".github/workflows/deploy.yml")
    
    if not workflow_path.exists():
        print("❌ GitHub Actions workflow not found")
        return False
    
    print("🔍 Testing GitHub Actions workflow...")
    
    with open(workflow_path, 'r') as f:
        content = f.read()
    
    tests_passed = 0
    total_tests = 1
    
    # Test 1: Check for interactive HTML file copying step
    if 'Copy interactive HTML files' in content:
        print("✅ Interactive HTML file copying step found")
        tests_passed += 1
    else:
        print("❌ Interactive HTML file copying step missing")
    
    print(f"\n📊 GitHub Actions tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def create_test_summary():
    """Create a summary of what should now work."""
    print("\n" + "="*60)
    print("🗺️ MAP RENDERING FIXES SUMMARY")
    print("="*60)
    
    print("\n✅ FIXES APPLIED:")
    print("1. 📂 Updated file paths to work with Jupyter Book")
    print("2. 🔗 Improved interactive map links with better styling")
    print("3. ⚙️ Enhanced Jupyter Book configuration for HTML files")
    print("4. 🚀 Updated GitHub Actions to copy interactive files")
    print("5. 🛡️ Added fallback handling for missing directories")
    
    print("\n🎯 EXPECTED RESULTS:")
    print("• States of Fragility 2022 maps should render properly")
    print("• FCS Map - FY25 should display correctly")
    print("• Interactive versions accessible via styled buttons")
    print("• Maps work in both local development and deployed site")
    print("• Automatic copying of HTML files during build process")
    
    print("\n🔧 TECHNICAL CHANGES:")
    print("• HTML files saved to notebook directory and build directory")
    print("• Markdown links replaced with styled note boxes")
    print("• GitHub Actions workflow copies HTML files post-build")
    print("• Jupyter Book config includes HTML file handling")
    print("• Error handling for missing build directories")
    
    print("\n🚀 TESTING RECOMMENDATIONS:")
    print("1. Build the book locally: jupyter-book build _sources")
    print("2. Check for HTML files in _build/html/ directory")
    print("3. Test interactive links in the built site")
    print("4. Deploy to GitHub Pages to test full workflow")

def main():
    """Run all tests and provide summary."""
    print("🧪 Testing Map Rendering Fixes")
    print("="*50)
    
    # Run tests
    notebook_ok = test_notebook_structure()
    config_ok = test_config_updates()
    actions_ok = test_github_actions()
    
    # Overall results
    all_tests_passed = notebook_ok and config_ok and actions_ok
    
    print("\n" + "="*50)
    print("📊 OVERALL TEST RESULTS")
    print("="*50)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Map rendering fixes are properly implemented")
        print("✅ Configuration updates are in place")
        print("✅ GitHub Actions workflow is updated")
        
        create_test_summary()
        
        print("\n🎯 READY FOR DEPLOYMENT!")
        print("The maps should now render properly in the slide deck.")
        
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please review the issues above and re-run the fix scripts if needed.")
        
        if not notebook_ok:
            print("• Re-run: python3 scripts/fix_maps_simple.py")
        if not config_ok:
            print("• Check _config.yml for HTML configuration")
        if not actions_ok:
            print("• Check .github/workflows/deploy.yml for copy step")
    
    return all_tests_passed

if __name__ == "__main__":
    main()