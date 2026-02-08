#!/usr/bin/env python3
"""
Enhanced Neural Network Training with Historical Data
================================================

Train the neural network on extensive historical MT5 data to learn
profitable patterns and enable frequent trading while maintaining profitability.

SEQUENTIAL THINKING: Historical Learning → Pattern Recognition → Frequent Profitable Trading
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import MetaTrader5 as mt5
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalTradingData(Dataset):
    """Custom dataset for historical trading data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EnhancedNeuralNetwork(nn.Module):
    """Enhanced neural network for forex prediction"""
    
    def __init__(self, input_size=6, hidden_sizes=[128, 64, 32], output_size=3):
        super(EnhancedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class EnhancedNeuralTrainer:
    """Train neural network on extensive historical data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("ENHANCED NEURAL TRAINER INITIALIZED")
        
        # Training parameters
        self.input_features = 10  # Price momentum, z-score, SMA ratios, RSI, volatility, trend, Bollinger Bands, Candlestick Patterns
        self.output_classes = 3  # BUY, SELL, HOLD
        self.hidden_layers = [256, 128, 64]  # Larger network for better learning
        
        # Data parameters
        self.min_data_points = 10000  # Minimum historical data points needed
        self.lookback_periods = [5, 15, 30, 60, 240]  # Multiple timeframes
        
        # Training settings
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epochs = 200
        self.validation_split = 0.2
        
        # Historical data storage
        self.historical_data = {}
        self.training_features = []
        self.training_labels = []
        
    def collect_extensive_historical_data(self):
        """Collect extensive historical data from MT5"""
        
        self.logger.info("COLLECTING EXTENSIVE HISTORICAL DATA FROM MT5")
        
        # Connect to MT5
        if not mt5.initialize():
            self.logger.error("Failed to initialize MT5")
            return False
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            mt5.shutdown()
            return False
        
        self.logger.info(f"Connected to MT5 - Account: {account_info.login}")
        self.logger.info(f"Server: {account_info.server}")
        
        # Focused trading pairs (USDJPY + USDCAD)
        trading_pairs = ['USDJPY', 'USDCAD']
        
        # Collect data for multiple years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)  # 3 years of data
        
        total_data_points = 0
        
        for pair in trading_pairs:
            self.logger.info(f"Collecting data for {pair}")
            
            pair_data = []
            
            # Collect data for each timeframe
            for period_name, period_minutes in [
                ('M5', 5), ('M15', 15), ('M30', 30), 
                ('H1', 60), ('H4', 240)
            ]:
                
                # Get rates
                rates = mt5.copy_rates_range(pair, period_minutes, start_date, end_date)
                
                if rates is not None and len(rates) > 100:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Calculate technical indicators
                    df = self._calculate_technical_indicators(df)
                    
                    # Generate labels (BUY/SELL/HOLD)
                    df = self._generate_trading_labels(df)
                    
                    # Store data
                    key = f"{pair}_{period_name}"
                    self.historical_data[key] = df
                    pair_data.append(df)
                    
                    self.logger.info(f"  {period_name}: {len(df)} data points")
                    total_data_points += len(df)
            
            if pair_data:
                self.logger.info(f"Total {pair} data points: {sum(len(df) for df in pair_data)}")
        
        self.logger.info(f"TOTAL HISTORICAL DATA POINTS: {total_data_points}")
        
        mt5.shutdown()
        return total_data_points >= self.min_data_points
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Z-score (price deviation from mean)
        df['price_zscore'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # SMA ratios
        df['sma_5_ratio'] = df['sma_5'] / df['close'] - 1
        df['sma_20_ratio'] = df['sma_20'] / df['close'] - 1
        df['sma_50_ratio'] = df['sma_50'] / df['close'] - 1
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        
        # Trend strength
        df['trend_strength'] = df['close'] / df['close'].shift(20) - 1
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Candlestick Pattern Recognition (User's Trading Knowledge)
        # 4 consecutive candles = trend continuation
        # 3 consecutive candles = trend reversal
        
        # Calculate consecutive candles
        df['prev_close'] = df['close'].shift(1)
        df['prev_open'] = df['open'].shift(1)
        
        # Initialize pattern columns
        df['trend_continuation_score'] = 0.0
        df['trend_reversal_score'] = 0.0
        
        # Calculate patterns for each row (starting from index 4 to have enough data)
        for i in range(4, len(df)):
            consecutive_bullish = 0
            consecutive_bearish = 0
            
            # Count consecutive bullish candles
            for j in range(i, max(0, i-5), -1):
                if df.iloc[j]['close'] > df.iloc[j]['open']:
                    consecutive_bullish += 1
                else:
                    break
            
            # Count consecutive bearish candles  
            for j in range(i, max(0, i-5), -1):
                if df.iloc[j]['close'] < df.iloc[j]['open']:
                    consecutive_bearish += 1
                else:
                    break
            
            # Pattern scores (normalized)
            df.loc[i, 'trend_continuation_score'] = min(consecutive_bullish, consecutive_bearish, 4) / 4.0
            df.loc[i, 'trend_reversal_score'] = min(consecutive_bullish, consecutive_bearish, 3) / 3.0
        
        return df
    
    def _generate_trading_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading labels based on future price movement"""
        
        # Look ahead 5 periods for price movement
        df['future_price'] = df['close'].shift(-5)
        df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
        
        # Create labels based on price movement
        df['label'] = 1  # HOLD default
        
        # BUY signal: price increases by more than 0.5%
        df.loc[df['price_change_future'] > 0.005, 'label'] = 2  # BUY
        
        # SELL signal: price decreases by more than 0.5%
        df.loc[df['price_change_future'] < -0.005, 'label'] = 0  # SELL
        
        # Clean up
        df = df.dropna()
        
        return df
    
    def prepare_training_data(self):
        """Prepare training data from historical dataset"""
        
        self.logger.info("PREPARING TRAINING DATA FROM HISTORICAL DATASET")
        
        all_features = []
        all_labels = []
        
        # Combine data from all pairs and timeframes
        for key, df in self.historical_data.items():
            if len(df) < 100:
                continue
            
            # Extract features
            features = df[['price_momentum', 'price_zscore', 'sma_5_ratio', 
                         'sma_20_ratio', 'rsi', 'volatility', 'trend_strength', 
                         'bb_position', 'trend_continuation_score', 
                         'trend_reversal_score']].values
            
            # Remove NaN values
            valid_indices = ~np.isnan(features).any(axis=1)
            features = features[valid_indices]
            labels = df['label'].values[valid_indices]
            
            all_features.extend(features)
            all_labels.extend(labels)
        
        # Convert to numpy arrays
        self.training_features = np.array(all_features)
        self.training_labels = np.array(all_labels)
        
        # Remove any remaining NaN values
        valid_mask = ~np.isnan(self.training_features).any(axis=1)
        self.training_features = self.training_features[valid_mask]
        self.training_labels = self.training_labels[valid_mask]
        
        self.logger.info(f"TRAINING DATA PREPARED:")
        self.logger.info(f"  Features shape: {self.training_features.shape}")
        self.logger.info(f"  Labels shape: {self.training_labels.shape}")
        self.logger.info(f"  BUY signals: {np.sum(self.training_labels == 2)}")
        self.logger.info(f"  SELL signals: {np.sum(self.training_labels == 0)}")
        self.logger.info(f"  HOLD signals: {np.sum(self.training_labels == 1)}")
        
        return len(self.training_features) > 1000
    
    def train_enhanced_model(self):
        """Train enhanced neural network on historical data"""
        
        self.logger.info("TRAINING ENHANCED NEURAL NETWORK")
        
        if len(self.training_features) == 0:
            self.logger.error("No training data available")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_features, self.training_labels, 
            test_size=self.validation_split, random_state=42, stratify=self.training_labels
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = HistoricalTradingData(X_train_scaled, y_train)
        test_dataset = HistoricalTradingData(X_test_scaled, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        self.model = EnhancedNeuralNetwork(
            input_size=X_train_scaled.shape[1],
            hidden_sizes=self.hidden_layers,
            output_size=self.output_classes
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_accuracy = 0
        training_history = {'train_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels.long())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels.long()).sum().item()
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    outputs = self.model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels.long()).sum().item()
            
            # Calculate metrics
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Store history
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_accuracy'].append(train_accuracy)
            training_history['val_accuracy'].append(val_accuracy)
            
            # Update learning rate
            scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler': self.scaler,
                    'training_history': training_history,
                    'model_config': {
                        'input_size': X_train_scaled.shape[1],
                        'hidden_layers': self.hidden_layers,
                        'output_size': self.output_classes,
                        'training_date': datetime.now().isoformat(),
                        'data_points': len(self.training_features),
                        'best_accuracy': best_accuracy
                    }
                }, 'enhanced_neural_model.pth')
            
            # Log progress
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}/{self.epochs}: "
                               f"Loss: {avg_train_loss:.4f}, "
                               f"Train Acc: {train_accuracy:.2f}%, "
                               f"Val Acc: {val_accuracy:.2f}%")
        
        self.logger.info(f"TRAINING COMPLETED - Best Validation Accuracy: {best_accuracy:.2f}%")
        
        return best_accuracy > 60  # Require at least 60% accuracy
    
    def save_training_report(self):
        """Save comprehensive training report"""
        
        report = {
            'training_completed': datetime.now().isoformat(),
            'data_summary': {
                'total_data_points': len(self.training_features),
                'training_pairs': list(self.historical_data.keys()),
                'time_periods': ['M5', 'M15', 'M30', 'H1', 'H4'],
                'label_distribution': {
                    'BUY': int(np.sum(self.training_labels == 2)),
                    'SELL': int(np.sum(self.training_labels == 0)),
                    'HOLD': int(np.sum(self.training_labels == 1))
                }
            },
            'model_config': {
                'architecture': 'Enhanced Neural Network',
                'hidden_layers': self.hidden_layers,
                'input_features': self.input_features,
                'output_classes': self.output_classes,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            },
            'training_goals': {
                'frequent_trading': 'Model trained on extensive data for frequent profitable trades',
                'pattern_recognition': 'Learned from 3 years of historical market data',
                'profitability': 'Optimized for consistent profitability over high frequency'
            }
        }
        
        with open('enhanced_training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info("Training report saved to enhanced_training_report.json")

def main():
    """Run enhanced neural network training"""
    
    print("ENHANCED NEURAL NETWORK TRAINING")
    print("Training on extensive historical data for frequent profitable trading")
    print("=" * 80)
    
    trainer = EnhancedNeuralTrainer()
    
    # Step 1: Collect extensive historical data
    if not trainer.collect_extensive_historical_data():
        print("Failed to collect sufficient historical data")
        return
    
    # Step 2: Prepare training data
    if not trainer.prepare_training_data():
        print("Failed to prepare training data")
        return
    
    # Step 3: Train enhanced model
    success = trainer.train_enhanced_model()
    
    if success:
        trainer.save_training_report()
        print("\nENHANCED NEURAL TRAINING COMPLETED SUCCESSFULLY")
        print("Model trained on extensive historical data for frequent profitable trading")
    else:
        print("\nENHANCED NEURAL TRAINING FAILED")
        print("Model accuracy below required threshold")

if __name__ == "__main__":
    main()
