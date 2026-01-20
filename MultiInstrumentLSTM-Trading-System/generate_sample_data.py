import pandas as pd
import numpy as np
import os

# Create sample data for multiple instruments
def generate_sample_data(instrument_name, start_date='2015-01-01', end_date='2025-12-31'):
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate synthetic price data with trend and noise
    np.random.seed(42)  # For reproducibility
    
    # Create a base price with a trend
    base_price = 100
    trend = np.linspace(0, 200, n_days)  # Upward trend
    volatility = 2.0
    
    # Generate returns with some autocorrelation
    returns = np.random.normal(0, volatility/100, n_days)
    prices = np.zeros(n_days)
    prices[0] = base_price
    
    for i in range(1, n_days):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    # Add trend to prices
    prices += trend
    
    # Generate OHLC data
    open_prices = prices
    close_prices = np.roll(prices, -1)
    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 1, n_days)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 1, n_days)
    
    # Generate volume data
    volume = np.random.randint(100000, 1000000, n_days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    # Remove the last row since we rolled the prices
    df = df.iloc[:-1]
    
    return df

# Generate data for multiple instruments
instruments = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data_dir = 'History_Data'

# Create data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Generate and save data for each instrument
for instrument in instruments:
    df = generate_sample_data(instrument)
    file_path = os.path.join(data_dir, f'{instrument}.csv')
    df.to_csv(file_path)
    print(f'Saved sample data for {instrument} to {file_path}')

print('\nSample data generation complete!')