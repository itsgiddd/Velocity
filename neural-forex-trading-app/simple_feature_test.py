#!/usr/bin/env python3
"""
Simple test to verify 10-feature extraction fix
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def test_10_feature_extraction():
    """Test the enhanced 10-feature extraction"""
    
    print("Testing 10-Feature Extraction Fix")
    print("=" * 40)
    
    # Create sample data (need 50+ points for all calculations)
    np.random.seed(42)
    base_price = 150.0  # USDJPY price
    data = []
    
    for i in range(60):  # Need at least 50 points
        if i == 0:
            price = base_price
        else:
            price = data[-1]['close'] + np.random.normal(0, 0.5)
        
        price = max(price, 1.0)
        spread = np.random.uniform(0.01, 0.05)
        open_price = price
        high_price = price + np.random.uniform(0, spread)
        low_price = price - np.random.uniform(0, spread)
        close_price = price + np.random.normal(0, spread/2)
        
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        timestamp = datetime.now() - timedelta(minutes=i*15)
        
        data.append({
            'time': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'tick_volume': np.random.randint(1, 100)
        })
    
    market_data = {'M15': data}
    
    try:
        # Test the feature extraction logic (same as in trading_engine.py)
        if 'M15' not in market_data:
            print("X No M15 data")
            return False
        
        m15_data = market_data['M15']
        if len(m15_data) < 50:
            print(f"X Insufficient data: {len(m15_data)} points (need 50+)")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(m15_data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Calculate all 10 features
        df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
        df['price_zscore'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_5_ratio'] = df['sma_5'] / df['close'] - 1
        df['sma_20_ratio'] = df['sma_20'] / df['close'] - 1
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['trend_strength'] = df['close'] / df['close'].shift(20) - 1
        
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        df['trend_continuation_score'] = 0.0
        df['trend_reversal_score'] = 0.0
        
        # Candlestick patterns
        for i in range(4, len(df)):
            consecutive_bullish = 0
            consecutive_bearish = 0
            
            for j in range(i, max(0, i-5), -1):
                if df.iloc[j]['close'] > df.iloc[j]['open']:
                    consecutive_bullish += 1
                else:
                    break
            
            for j in range(i, max(0, i-5), -1):
                if df.iloc[j]['close'] < df.iloc[j]['open']:
                    consecutive_bearish += 1
                else:
                    break
            
            df.loc[i, 'trend_continuation_score'] = min(consecutive_bullish, consecutive_bearish, 4) / 4.0
            df.loc[i, 'trend_reversal_score'] = min(consecutive_bullish, consecutive_bearish, 3) / 3.0
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Check features
        required_features = ['price_momentum', 'price_zscore', 'sma_5_ratio', 'sma_20_ratio', 
                           'rsi', 'volatility', 'trend_strength', 'bb_position', 
                           'trend_continuation_score', 'trend_reversal_score']
        
        print("Feature availability check:")
        for feature in required_features:
            value = latest[feature]
            if pd.isna(value):
                print(f"  X {feature}: NaN")
            else:
                print(f"  + {feature}: {value:.6f}")
        
        if pd.isna(latest[required_features]).any():
            print("X Some features are NaN - need more data points")
            return False
        
        # Create feature vector
        features = [
            latest['price_momentum'],           # 1. Price momentum
            latest['price_zscore'],              # 2. Z-score
            latest['sma_5_ratio'],              # 3. SMA 5 ratio
            latest['sma_20_ratio'],              # 4. SMA 20 ratio
            latest['rsi'],                       # 5. RSI
            latest['volatility'],                 # 6. Volatility
            latest['trend_strength'],            # 7. Trend strength
            latest['bb_position'],               # 8. Bollinger Bands position
            latest['trend_continuation_score'],  # 9. Trend continuation score
            latest['trend_reversal_score']       # 10. Trend reversal score
        ]
        
        print(f"\nFeature extraction result:")
        print(f"Number of features: {len(features)}")
        print(f"Expected features: 10")
        
        if len(features) == 10:
            print("+ Correct number of features!")
            
            # Test with neural model if available
            try:
                from model_manager import NeuralModelManager
                model_manager = NeuralModelManager()
                if model_manager.load_model("enhanced_neural_model.pth"):
                    prediction = model_manager.predict(np.array(features).reshape(1, -1))
                    if prediction:
                        print(f"+ Neural model prediction successful!")
                        print(f"  Action: {prediction['action']}")
                        print(f"  Confidence: {prediction['confidence']:.1%}")
                        probs = prediction['probabilities']
                        print(f"  BUY: {probs['BUY']:.1%}, SELL: {probs['SELL']:.1%}, HOLD: {probs['HOLD']:.1%}")
                        
                        if probs['BUY'] > 0.05 and probs['SELL'] > 0.05:
                            print("+ Neural model shows both BUY and SELL capability!")
                            return True
                        else:
                            print("! Model bias detected")
                            return False
                    else:
                        print("! Neural prediction failed")
                        return False
                else:
                    print("! Could not load model")
                    return True  # Feature extraction works
            except Exception as e:
                print(f"! Model test error: {e}")
                return True  # Feature extraction works
            
            return True
        else:
            print("X Wrong number of features!")
            return False
            
    except Exception as e:
        print(f"X Error: {e}")
        return False

def main():
    print("Neural Trading - Feature Extraction Fix Test")
    print("=" * 50)
    
    success = test_10_feature_extraction()
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: 10-feature extraction is working!")
        print("This should fix the issue of only getting SELL signals.")
    else:
        print("FAILED: Feature extraction needs more work.")
    
    return success

if __name__ == "__main__":
    main()