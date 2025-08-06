#!/bin/bash
# Build script for FCV Analytics Jupyter Book

set -e  # Exit on any error

echo "🚀 Starting FCV Analytics Jupyter Book build..."

# Check if virtual environment exists
if [[ ! -d "fcv-env" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv fcv-env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source fcv-env/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf _build/
rm -rf _sources/_build/

# Create cache directory
echo "📁 Setting up cache directory..."
mkdir -p _sources/notebooks/cache

# Optimize notebook (optional - comment out if not needed)
echo "⚡ Optimizing notebook for performance..."
python scripts/optimize_notebook.py

# Build the book
echo "📚 Building Jupyter Book..."
jupyter-book build _sources --path-output ./ --builder html

# Check if build was successful
if [[ -d "_build/html" ]]; then
    echo "✅ Build successful! Book is ready at _build/html/index.html"
    echo "🌐 To serve locally, run: python -m http.server 8000 -d _build/html"
else
    echo "❌ Build failed! Check the output above for errors."
    exit 1
fi

# Optional: Open the book in browser (uncomment if desired)
# if command -v open &> /dev/null; then
#     open _build/html/index.html
# elif command -v xdg-open &> /dev/null; then
#     xdg-open _build/html/index.html
# fi

echo "🎉 Build complete!"