#!/usr/bin/env python3
"""
Dashboard Creator for GoodHunt v3+
==================================
Interactive performance dashboards and visualizations
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger('GoodHunt.Dashboard')

def create_performance_dashboard(trades: List[Dict], equity_curve: List[float], 
                               output_file: str = None) -> str:
    """Create interactive performance dashboard"""
    try:
        if not output_file:
            output_file = f"logs/dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Process data
        df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_data = pd.Series(equity_curve) if equity_curve else pd.Series()
        
        # Calculate metrics
        total_return = ((equity_data.iloc[-1] - equity_data.iloc[0]) / equity_data.iloc[0] * 100) if len(equity_data) > 1 else 0
        win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) * 100 if len(df_trades) > 0 else 0
        
        # Create HTML dashboard
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GoodHunt Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metrics {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
        .metric {{ text-align: center; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .chart {{ margin-bottom: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🦊 GoodHunt Performance Dashboard</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <h3>Total Return</h3>
            <h2>{total_return:.2f}%</h2>
        </div>
        <div class="metric">
            <h3>Total Trades</h3>
            <h2>{len(df_trades)}</h2>
        </div>
        <div class="metric">
            <h3>Win Rate</h3>
            <h2>{win_rate:.1f}%</h2>
        </div>
    </div>
    
    <div id="equity-chart" class="chart"></div>
    <div id="trades-chart" class="chart"></div>
    
    <script>
        // Equity curve chart
        var equityData = [{
            x: {list(range(len(equity_data)))},
            y: {equity_data.tolist() if len(equity_data) > 0 else []},
            type: 'scatter',
            mode: 'lines',
            name: 'Equity Curve',
            line: {{color: 'blue'}}
        }];
        
        var equityLayout = {{
            title: 'Equity Curve',
            xaxis: {{title: 'Time'}},
            yaxis: {{title: 'Portfolio Value'}}
        }};
        
        Plotly.newPlot('equity-chart', equityData, equityLayout);
        
        // Trades distribution
        var tradesData = [{
            x: {df_trades['pnl'].tolist() if 'pnl' in df_trades.columns else []},
            type: 'histogram',
            nbinsx: 30,
            name: 'Trade P&L Distribution'
        }];
        
        var tradesLayout = {{
            title: 'Trade P&L Distribution',
            xaxis: {{title: 'P&L'}},
            yaxis: {{title: 'Frequency'}}
        }};
        
        Plotly.newPlot('trades-chart', tradesData, tradesLayout);
    </script>
</body>
</html>
        """
        
        # Save dashboard
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📊 Dashboard created: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Dashboard creation failed: {e}")
        return ""