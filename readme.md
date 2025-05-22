# Technical Trading Strategy Framework

## Overview
This project implements a comprehensive algorithmic trading framework that combines multiple technical indicators to generate buy and sell signals for financial market assets. The system uses Exponential Moving Averages (EMA), Relative Strength Index (RSI), and Average Directional Index (ADX) to identify potential trading opportunities based on trend strength and momentum.

![Trading Strategy Visualization](https://github.com/Initin0/Comp_finance/blob/main/data.png)

## Features
- **Modular Object-Oriented Design**: Easily extensible and maintainable codebase
- **Multi-Factor Signal Generation**: Combines trend and momentum indicators
- **Comprehensive Backtesting System**: Complete metrics and performance analysis 
- **Visualization Tools**: Detailed charts for prices, indicators, and performance
- **Risk Management**: Tracking of drawdowns, win rates, and other risk metrics

## Technical Indicators
The strategy incorporates three primary technical indicators:
1. **EMA Crossover**: Fast EMA (10-period) and Slow EMA (30-period) crossover for trend direction
2. **RSI (Relative Strength Index)**: 14-period RSI for identifying overbought/oversold conditions (buy threshold: 40, sell threshold: 60)
3. **ADX (Average Directional Index)**: 14-period ADX for measuring trend strength (threshold: 20)

## Performance Metrics (Last Backtest)
- **Total Return**: 0.20% (vs. Market Return: 1.62%)
- **Annualized Return**: 0.41% (vs. Market Annualized: 3.31%)
- **Sharpe Ratio**: 0.01
- **Maximum Drawdown**: 30.88%
- **Win Rate**: 50.00%
- **Profit Factor**: 1.40
- **Best Trade**: 15.23% gain (Nov 27, 2024 - Jan 29, 2025)
- **Worst Trade**: 10.86% loss (Feb 21, 2025 - May 9, 2025)

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- yfinance

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/trading-strategy.git
cd trading-strategy

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
Basic usage example:
```python
from trading_strategy import TradingStrategy

# Create a new strategy instance
strategy = TradingStrategy(ticker="MSFT", period="1y", interval="1d")

# Fetch data and calculate indicators
strategy.fetch_data()
strategy.calculate_indicators(fast_ema=12, slow_ema=26, adx_period=14, rsi_period=14)

# Generate trading signals
strategy.generate_signals(adx_threshold=25, rsi_buy=30, rsi_sell=70)

# Run backtest
strategy.backtest_strategy(initial_capital=10000)

# Print performance summary
strategy.summary()

# Plot results
fig = strategy.plot_results()
plt.show()
```

## Customization
The framework allows for easy customization of:
- Ticker symbols
- Time periods and data intervals
- Indicator parameters (EMA lengths, RSI thresholds, ADX periods)
- Initial capital for backtesting

## Future Improvements
- Add machine learning models for signal enhancement
- Implement portfolio optimization and position sizing
- Develop real-time trading capabilities
- Add more sophisticated risk management tools
- Incorporate fundamental data analysis

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This software is for educational purposes only. It is not intended to be investment advice. Past performance is not indicative of future results. Use at your own risk.
