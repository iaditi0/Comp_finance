import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from datetime import datetime

# Set plotting style
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')

class TradingStrategy:
    def __init__(self, ticker="AMZN", period="6mo", interval="1d"):
        """
        Initialize the trading strategy with default parameters
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (default: AMZN for Amazon)
        period : str
            Time period for data (default: 6mo for 6 months)
        interval : str
            Data interval (default: 1d for daily)
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = None
        self.signals = None
        self.trades = []
        self.performance = {}
    
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        print(f"Fetching {self.ticker} data for the past {self.period}...")
        try:
            self.data = yf.download(self.ticker, period=self.period, interval=self.interval, progress=False)
            print(f"Downloaded {len(self.data)} rows of data")
            # Check for empty data
            if len(self.data) == 0:
                print(f"Warning: No data downloaded for {self.ticker}")
                return None
            return self.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, fast_ema=12, slow_ema=26, adx_period=14, rsi_period=14):
        """
        Calculate technical indicators: EMA, ADX, and RSI
        
        Parameters:
        -----------
        fast_ema : int
            Fast EMA period (default: 12)
        slow_ema : int
            Slow EMA period (default: 26)
        adx_period : int
            ADX period (default: 14)
        rsi_period : int
            RSI period (default: 14)
        """
        if self.data is None or len(self.data) == 0:
            self.data = self.fetch_data()
            if self.data is None or len(self.data) == 0:
                return None
        
        # Make a copy to avoid modifying the original
        df = self.data.copy()
        
        # Calculate EMAs
        df['EMA_fast'] = df['Close'].ewm(span=fast_ema, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow_ema, adjust=False).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        # Prevent division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate ADX
        # True Range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        # Directional Movement
        df['DM+'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        
        df['DM-'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )
        
        # Smoothed TR and DM (using exponential moving average for smoother values)
        df['ATR'] = df['TR'].ewm(span=adx_period, adjust=False).mean()
        df['DI+'] = 100 * (df['DM+'].ewm(span=adx_period, adjust=False).mean() / df['ATR'])
        df['DI-'] = 100 * (df['DM-'].ewm(span=adx_period, adjust=False).mean() / df['ATR'])
        
        # Handle division by zero
        df['DI+'] = df['DI+'].replace([np.inf, -np.inf], np.nan)
        df['DI-'] = df['DI-'].replace([np.inf, -np.inf], np.nan)
        
        # Directional Index
        di_sum = df['DI+'] + df['DI-']
        df['DX'] = 100 * (abs(df['DI+'] - df['DI-']) / di_sum.replace(0, np.nan))
        
        # Average Directional Index
        df['ADX'] = df['DX'].ewm(span=adx_period, adjust=False).mean()
        
        # Forward fill and backward fill NaN values to handle missing data
        df = df.ffill().bfill()
        
        # Drop rows with remaining NaN values (typically only the first few rows)
        df = df.dropna()
        
        self.data = df
        return df
    
    def generate_signals(self, adx_threshold=20, rsi_buy=40, rsi_sell=60):
        """
        Generate buy/sell signals based on EMA crossover, ADX, and RSI
        
        Parameters:
        -----------
        adx_threshold : int
            ADX threshold value (default: 20)
        rsi_buy : int
            RSI oversold threshold for buy signal (default: 40)
        rsi_sell : int
            RSI overbought threshold for sell signal (default: 60)
        """
        if 'ADX' not in self.data.columns:
            self.calculate_indicators()
        
        if len(self.data) == 0:
            print("No data available to generate signals")
            return None
        
        # Create a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Initialize signal column
        df['Signal'] = 0
        
        # Buy signal: Fast EMA crosses above Slow EMA AND (ADX > threshold OR RSI < rsi_buy)
        buy_condition = (
            (df['EMA_fast'] > df['EMA_slow']) & 
            (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1)) &
            ((df['ADX'] > adx_threshold) | (df['RSI'] < rsi_buy))
        )
        df.loc[buy_condition, 'Signal'] = 1
        
        # Sell signal: Fast EMA crosses below Slow EMA AND (ADX > threshold OR RSI > rsi_sell)
        sell_condition = (
            (df['EMA_fast'] < df['EMA_slow']) & 
            (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1)) &
            ((df['ADX'] > adx_threshold) | (df['RSI'] > rsi_sell))
        )
        df.loc[sell_condition, 'Signal'] = -1
        
        # Add more buy/sell signals based on strong RSI conditions
        # Strong oversold condition - additional buy signal
        strong_buy = (df['RSI'] < rsi_buy - 10) & (df['RSI'].shift(1) >= rsi_buy - 10)
        df.loc[strong_buy, 'Signal'] = 1
        
        # Strong overbought condition - additional sell signal
        strong_sell = (df['RSI'] > rsi_sell + 10) & (df['RSI'].shift(1) <= rsi_sell + 10)
        df.loc[strong_sell, 'Signal'] = -1
        
        # Update class data
        self.data = df
        self.signals = df[df['Signal'] != 0].copy()
        
        return self.signals
    
    def backtest_strategy(self, initial_capital=10000):
        """
        Backtest the trading strategy
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital for the backtest (default: 10000)
        """
        if self.data is None or len(self.data) == 0:
            print("No data available for backtesting")
            return None
            
        if 'Signal' not in self.data.columns:
            self.generate_signals()
        
        # Create a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Initialize position and portfolio columns
        df['Position'] = 0
        df['Portfolio'] = initial_capital
        
        # Track current position and trades
        position = 0
        entry_price = 0
        entry_date = None
        self.trades = []
        
        # Instead of iterating through rows, use the signal column directly
        for i in range(len(df)):
            date = df.index[i]
            current_signal = df['Signal'].iloc[i]
            current_close = df['Close'].iloc[i]
            
            # Convert to float to ensure we're not dealing with a Series
            if hasattr(current_close, 'item'):
                current_close = float(current_close)
            
            # Update position based on signals
            if current_signal == 1 and position == 0:  # Buy signal when not holding
                position = 1
                entry_price = current_close
                entry_date = date
                df.at[date, 'Position'] = 1
            elif current_signal == -1 and position == 1:  # Sell signal when holding
                position = 0
                exit_price = current_close
                df.at[date, 'Position'] = 0
                
                # Ensure entry_price is a float (not a Series)
                if hasattr(entry_price, 'item'):
                    entry_price = float(entry_price)
                
                # Calculate profit/loss as a float
                pnl = (exit_price - entry_price) / entry_price
                
                # Record trade
                self.trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': f"{pnl:.2%}"  # Format as percentage
                })
            else:
                # Maintain previous position
                df.at[date, 'Position'] = position
        
        # Close out any open position at the end of the period
        if position == 1:
            exit_price = float(df['Close'].iloc[-1])
            exit_date = df.index[-1]
            
            # Ensure entry_price is a float
            if hasattr(entry_price, 'item'):
                entry_price = float(entry_price)
                
            pnl = (exit_price - entry_price) / entry_price
            
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': f"{pnl:.2%}"  # Format as percentage
            })
        
        # Calculate forward-fill positions to handle gaps in trading
        df['Position'] = df['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change().fillna(0)
        
        # Calculate strategy returns (today's position * today's return)
        df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
        df.loc[df.index[0], 'Strategy_Return'] = 0  # First day has no return
        
        # Calculate cumulative returns
        df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod() - 1
        df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
        
        # Calculate portfolio value
        df['Portfolio_Value'] = initial_capital * (1 + df['Cumulative_Strategy_Return'])
        
        # Calculate performance metrics
        total_days = (df.index[-1] - df.index[0]).days
        if total_days <= 0:
            total_days = 1  # Prevent division by zero
        
        total_return = df['Cumulative_Strategy_Return'].iloc[-1]
        market_return = df['Cumulative_Market_Return'].iloc[-1]
        
        # Annualize returns
        annual_return = ((1 + total_return) ** (365.0 / total_days)) - 1
        annual_market_return = ((1 + market_return) ** (365.0 / total_days)) - 1
        
        # Calculate Sharpe Ratio (assuming risk-free rate = 0)
        daily_returns_std = df['Strategy_Return'].std()
        if daily_returns_std > 0:
            sharpe_ratio = annual_return / (daily_returns_std * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        rolling_max = df['Portfolio_Value'].cummax()
        drawdown = (df['Portfolio_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        if self.trades:
            wins = sum(1 for trade in self.trades if trade['pnl'] > 0)
            win_rate = wins / len(self.trades)
        else:
            win_rate = 0
        
        # Calculate average profit per trade
        if self.trades:
            avg_profit = sum(trade['pnl'] for trade in self.trades) / len(self.trades)
        else:
            avg_profit = 0
        
        # Calculate profit factor
        gross_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trades if trade['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Store performance metrics
        self.performance = {
            'Total Return': f"{total_return:.2%}",
            'Market Return': f"{market_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Annual Market Return': f"{annual_market_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Number of Trades': len(self.trades),
            'Win Rate': f"{win_rate:.2%}" if win_rate != 'N/A' else 'N/A',
            'Average Profit per Trade': f"{avg_profit:.2%}",
            'Profit Factor': f"{profit_factor:.2f}" if profit_factor != float('inf') else 'Inf'
        }
        
        # Update data
        self.data = df
        
        return self.performance
    
    def plot_results(self):
        """Plot the strategy results"""
        if self.data is None or len(self.data) == 0:
            print("No data available for plotting")
            return None
            
        if 'Portfolio_Value' not in self.data.columns:
            self.backtest_strategy()
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 2]})
        
        # Plot 1: Price with EMAs and signals
        axs[0].plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.7)
        axs[0].plot(self.data.index, self.data['EMA_fast'], label=f'Fast EMA', color='orange')
        axs[0].plot(self.data.index, self.data['EMA_slow'], label=f'Slow EMA', color='red')
        
        # Buy signals
        buy_signals = self.data[self.data['Signal'] == 1]
        if not buy_signals.empty:
            axs[0].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal')
        
        # Sell signals
        sell_signals = self.data[self.data['Signal'] == -1]
        if not sell_signals.empty:
            axs[0].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell Signal')
        
        # Shade areas where we are holding position
        for trade in self.trades:
            axs[0].axvspan(trade['entry_date'], trade['exit_date'], alpha=0.2, color='green')
        
        axs[0].set_title(f'{self.ticker} Price with Trading Signals')
        axs[0].set_ylabel('Price ($)')
        axs[0].legend(loc='upper left')
        axs[0].grid(True)
        
        # Plot 2: ADX and RSI indicators
        axs[1].plot(self.data.index, self.data['ADX'], label='ADX', color='purple')
        axs[1].axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='ADX Threshold')
        axs[1].set_ylabel('ADX')
        axs[1].legend(loc='upper left')
        axs[1].grid(True)
        
        ax1_twin = axs[1].twinx()
        ax1_twin.plot(self.data.index, self.data['RSI'], label='RSI', color='blue')
        ax1_twin.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='RSI Sell')
        ax1_twin.axhline(y=40, color='green', linestyle='--', alpha=0.5, label='RSI Buy')
        ax1_twin.set_ylabel('RSI')
        ax1_twin.legend(loc='upper right')
        
        # Plot 3: Strategy performance
        axs[2].plot(self.data.index, (1 + self.data['Cumulative_Market_Return']) * 100, label='Buy and Hold', color='blue')
        axs[2].plot(self.data.index, (1 + self.data['Cumulative_Strategy_Return']) * 100, label='Strategy', color='green')
        axs[2].set_title('Strategy Performance')
        axs[2].set_ylabel('Value (starting at 100)')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].set_xlabel('Date')
        
        # Add performance metrics as text
        perf_text = "\n".join([f"{k}: {v}" for k, v in self.performance.items()])
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        fig.text(0.15, 0.01, perf_text, fontsize=10, bbox=props)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        return fig
    
    def summary(self):
        """Print a summary of the strategy and performance"""
        if not self.performance and self.data is not None:
            self.backtest_strategy()
        
        print(f"\n{'='*50}")
        print(f"TRADING STRATEGY SUMMARY FOR {self.ticker}")
        print(f"{'='*50}")
        print(f"Period: {self.period}, Interval: {self.interval}")
        
        if self.data is not None and len(self.data) > 0:
            date_format = '%Y-%m-%d'
            if isinstance(self.data.index[0], datetime):
                print(f"Data Range: {self.data.index[0].strftime(date_format)} to {self.data.index[-1].strftime(date_format)}")
            else:
                print(f"Data Range: {self.data.index[0]} to {self.data.index[-1]}")
            
            print(f"\nPerformance Metrics:")
            for key, value in self.performance.items():
                print(f"- {key}: {value}")
            
            print(f"\nTrading Activity:")
            buy_signals = len(self.data[self.data['Signal'] == 1])
            sell_signals = len(self.data[self.data['Signal'] == -1])
            print(f"- Buy signals generated: {buy_signals}")
            print(f"- Sell signals generated: {sell_signals}")
            print(f"- Completed trades: {len(self.trades)}")
            
            if self.trades:
                print(f"\nTrade Details:")
                for i, trade in enumerate(self.trades, 1):
                    print(f"Trade #{i}:")
                    print(f"  Entry: {trade['entry_date'].strftime(date_format) if isinstance(trade['entry_date'], datetime) else trade['entry_date']} at ${trade['entry_price']:.2f}")
                    print(f"  Exit: {trade['exit_date'].strftime(date_format) if isinstance(trade['exit_date'], datetime) else trade['exit_date']} at ${trade['exit_price']:.2f}")
                    print(f"  P&L: {trade['pnl_pct']}")
        else:
            print("No data available for analysis.")
        
        print(f"{'='*50}\n")
        
        return self.performance


# Example usage
if __name__ == "__main__":
    # Create strategy instance
    strategy = TradingStrategy(ticker="AMZN", period="6mo", interval="1d")
    
    # Fetch data and calculate indicators with adjusted parameters
    strategy.fetch_data()
    strategy.calculate_indicators(fast_ema=10, slow_ema=30, adx_period=14, rsi_period=14)
    
    # Generate signals with adjusted thresholds
    strategy.generate_signals(adx_threshold=20, rsi_buy=40, rsi_sell=60)
    
    # Backtest strategy
    strategy.backtest_strategy(initial_capital=10000)
    
    # Print summary
    strategy.summary()
    
    # Plot results
    fig = strategy.plot_results()
    plt.show()