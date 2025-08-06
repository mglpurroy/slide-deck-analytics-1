#!/usr/bin/env python3
"""
Create a simple fallback HTML site if Jupyter Book build fails
"""

import os
from pathlib import Path

def create_fallback_site():
    """Create a simple static HTML fallback site"""
    
    print("🚨 Creating fallback HTML site...")
    
    # Create a simple index.html
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FCV Slide Deck Analytics</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }
        .header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .status {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 2rem;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        .feature {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .button {
            display: inline-block;
            padding: 12px 24px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 0.5rem;
            transition: background 0.3s;
        }
        .button:hover {
            background: #0056b3;
        }
        .data-info {
            background: #e7f3ff;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #eee;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🗺️ FCV Slide Deck Analytics</h1>
        <p>Comprehensive analysis of fragility, conflict, and violence data</p>
    </div>
    
    <div class="status">
        <h3>⚠️ Site Status</h3>
        <p><strong>The Jupyter Book is currently rebuilding.</strong> This is a temporary fallback page while the full interactive site is being deployed.</p>
        <p>The site includes interactive maps, data visualizations, and comprehensive analysis of global fragility metrics.</p>
    </div>
    
    <div class="features">
        <div class="feature">
            <h3>🗺️ Interactive Maps</h3>
            <p>Explore States of Fragility 2022 data with interactive plotly visualizations, hover details, and zooming capabilities.</p>
        </div>
        
        <div class="feature">
            <h3>📊 Crime Index Analysis</h3>
            <p>Global crime statistics with dynamic plots and regional comparisons using dropdown controls.</p>
        </div>
        
        <div class="feature">
            <h3>📈 Population Trends</h3>
            <p>UN population data analysis by lending groups with projections through 2050.</p>
        </div>
        
        <div class="feature">
            <h3>⚔️ Conflict Analysis</h3>
            <p>UCDP conflict data with temporal and spatial analysis of global violence patterns.</p>
        </div>
    </div>
    
    <div class="data-info">
        <h3>📊 Data Sources</h3>
        <ul>
            <li><strong>States of Fragility 2022</strong>: OECD fragility metrics (0.5 MB)</li>
            <li><strong>UN Population Data</strong>: World population trends (24.9 MB)</li>
            <li><strong>UCDP Conflict Data</strong>: Uppsala conflict database (6.2 MB)</li>
            <li><strong>Global Crime Index</strong>: Crime statistics by country (0.2 MB)</li>
            <li><strong>World Bank Groups</strong>: Lending classifications (0.1 MB)</li>
        </ul>
        <p><strong>Total:</strong> 31.9 MB of comprehensive datasets</p>
    </div>
    
    <div style="text-align: center; margin: 2rem 0;">
        <a href="https://github.com/mglpurroy/slide-deck-analytics-1" class="button">📂 View Repository</a>
        <a href="https://github.com/mglpurroy/slide-deck-analytics-1/actions" class="button">🔄 Check Build Status</a>
    </div>
    
    <div class="footer">
        <p>Built with Jupyter Book • Deployed on GitHub Pages</p>
        <p>Repository: <strong>slide-deck-analytics-1</strong> • Branch: <strong>main</strong></p>
        <p><em>This fallback page will be replaced once the full site build completes.</em></p>
    </div>
</body>
</html>"""
    
    # Create fallback directory
    fallback_dir = Path("fallback")
    fallback_dir.mkdir(exist_ok=True)
    
    # Write the HTML file
    with open(fallback_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Fallback site created at fallback/index.html")
    print("📄 This can be used as a temporary deployment if Jupyter Book fails")
    
    return True

if __name__ == "__main__":
    create_fallback_site()