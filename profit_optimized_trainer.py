#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

class ProfitOptimizedNetwork(nn.Module):
    def __init__(self, input_size=80, hidden_size=256, num_classes=3):
        super(ProfitOptimizedNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_classes)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.direction_head(features), self.confidence_head(features)

class ProfitFeatureEngine:
    def __init__(self):
        self.feature_names = []
    
    def create_features(self, data):
        features = pd.DataFrame(index=data.index)
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = data['close'].rolling(period).mean()
            features[f'price_vs_ma_{period}'] = (data['close'] - features[f'ma_{period}']) / features[f'ma_{period}']
        features['ma_20_slope'] = features['ma_20'].diff(5) / features['ma_20'].shift(5)
        features['ma_50_slope'] = features['ma_50'].diff(5) / features['ma_50'].shift(5)
        features['rsi_7'] = self._rsi(data['close'], 7)
        features['rsi_14'] = self._rsi(data['close'], 14)
        features['rsi_21'] = self._rsi(data['close'], 21)
        features['rsi_divergence'] = features['rsi_14'].diff(3) - (data['close'].pct_change(3) * 100)
        for period in [1, 3, 5, 10, 20]:
            features[f'momentum_{period}'] = data['close'].pct_change(period)
        features['momentum_accel'] = features['momentum_5'].diff(3)
        features['tr'] = np.maximum(data['high']-data['low'], np.maximum(abs(data['high']-data['close'].shift(1)), abs(data['low']-data['close'].shift(1))))
        features['atr_14'] = features['tr'].rolling(14).mean()
        features['atr_ratio'] = features['atr_14'] / features['atr_14'].rolling(50).mean()
        ma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        features['bb_upper'] = ma_20 + (std_20 * 2)
        features['bb_lower'] = ma_20 - (std_20 * 2)
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / ma_20
        features['body_size'] = abs(data['close']-data['open']) / (data['high']-data['low']+0.00001)
        features['upper_wick'] = (data['high']-np.maximum(data['open'],data['close'])) / (data['high']-data['low']+0.00001)
        features['lower_wick'] = (np.minimum(data['open'],data['close'])-data['low']) / (data['high']-data['low']+0.00001)
        features['is_bullish'] = (data['close'] > data['open']).astype(float)
        features['high_10'] = data['high'].rolling(10).max()
        features['low_10'] = data['low'].rolling(10).min()
        features['range_position'] = (data['close']-features['low_10']) / (features['high_10']-features['low_10']+0.00001)
        if 'tick_volume' in data.columns:
            features['volume_ma'] = data['tick_volume'].rolling(20).mean()
            features['volume_ratio'] = data['tick_volume'] / features['volume_ma']
            features['volume_trend'] = features['volume_ratio'].rolling(5).mean() - 1
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        features['macd_hist_slope'] = features['macd_hist'].diff(2)
        low_14 = data['low'].rolling(14).min()
        high_14 = data['high'].rolling(14).max()
        features['stoch_k'] = 100 * (data['close']-low_14) / (high_14-low_14+0.00001)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        features['stoch_crossover'] = features['stoch_k'] - features['stoch_d']
        features['trend_alignment'] = ((features['price_vs_ma_5']>0).astype(float) + (features['price_vs_ma_10']>0).astype(float) + (features['price_vs_ma_20']>0).astype(float) + (features['price_vs_ma_50']>0).astype(float)) / 4
        features['mean_reversion_score'] = -features['bb_position'] + 0.5
        features['vol_adj_momentum'] = features['momentum_5'] / (features['atr_14']/data['close']+0.0001)
        features = features.fillna(0).replace([np.inf,-np.inf], 0)
        current_cols = len(features.columns)
        if current_cols < 80:
            for i in range(80-current_cols):
                features[f'padding_{i}'] = 0
        features = features.iloc[:, :80]
        self.feature_names = features.columns.tolist()
        return features
    
    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta>0,0)).rolling(window=period).mean()
        loss = (-delta.where(delta<0,0)).rolling(window=period).mean()
        rs = gain / (loss+0.0001)
        return 100 - (100 / (1+rs))

def generate_profit_labels(data, features, min_profit_atr=1.5, max_loss_atr=1.0, lookahead=10):
    labels, profit_amounts = [], []
    atr = features['atr_14'].values if 'atr_14' in features.columns else np.full(len(data), 0.001)
    for i in range(len(data)):
        if i + lookahead >= len(data):
            labels.append(2); profit_amounts.append(0); continue
        entry_price = data['close'].iloc[i]
        current_atr = max(atr[i], 0.0001)
        tp_distance, sl_distance = current_atr * min_profit_atr, current_atr * max_loss_atr
        future = data.iloc[i+1:i+lookahead+1]
        buy_result = _simulate_trade(future, entry_price+tp_distance, entry_price-sl_distance, 'BUY')
        sell_result = _simulate_trade(future, entry_price-tp_distance, entry_price+sl_distance, 'SELL')
        if buy_result > 0 and buy_result >= sell_result:
            labels.append(0); profit_amounts.append(buy_result)
        elif sell_result > 0 and sell_result > buy_result:
            labels.append(1); profit_amounts.append(sell_result)
        else:
            labels.append(2); profit_amounts.append(0)
    return np.array(labels), np.array(profit_amounts)

def _simulate_trade(future, tp, sl, direction):
    for _, bar in future.iterrows():
        if direction == 'BUY':
            if bar['low'] <= sl: return -1.0
            if bar['high'] >= tp: return 1.5
        else:
            if bar['high'] >= sl: return -1.0
            if bar['low'] <= tp: return 1.5
    final_price = future['close'].iloc[-1]
    entry_price = (tp + sl) / 2
    if direction == 'BUY':
        return (final_price - entry_price) / abs(tp - entry_price)
    else:
        return (entry_price - final_price) / abs(entry_price - tp)

class ProfitFocusedLoss(nn.Module):
    def __init__(self, loss_weight=2.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, predictions, targets, profit_amounts):
        base_loss = self.ce(predictions, targets)
        weights = torch.ones_like(base_loss)
        weights[profit_amounts > 0] = self.loss_weight
        return (base_loss * weights).mean()

def main():
    print('='*60)
    print('PROFIT-OPTIMIZED NEURAL MODEL TRAINER')
    print('='*60)
    
    if not mt5.initialize():
        print('MT5 initialization failed!')
        return None
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'BTCUSD']
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)
    
    all_data = {}
    print(f'\\nLoading 6 months of data for {len(symbols)} symbols...')
    for symbol in symbols:
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_time, end_time)
            if rates is not None and len(rates) > 100:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                all_data[symbol] = df
                print(f'  {symbol}: {len(df)} bars loaded')
        except Exception as e:
            print(f'  {symbol}: Error - {e}')
    
    if not all_data:
        print('No data loaded!')
        return None
    
    print('\\n' + '='*60)
    print('TRAINING MODEL')
    print('='*60)
    
    feature_engine = ProfitFeatureEngine()
    all_features, all_labels, all_profits = [], [], []
    
    for symbol, data in all_data.items():
        print(f'\\nProcessing {symbol}...')
        features = feature_engine.create_features(data)
        labels, profits = generate_profit_labels(data, features)
        valid_idx = 50
        features = features.iloc[valid_idx:]
        labels = labels[valid_idx:]
        profits = profits[valid_idx:]
        all_features.append(features)
        all_labels.extend(labels)
        all_profits.extend(profits)
        unique, counts = np.unique(labels, return_counts=True)
        print(f'  BUY={counts[0] if 0 in unique else 0}, SELL={counts[1] if 1 in unique else 0}, HOLD={counts[2] if 2 in unique else 0}')
    
    X = pd.concat(all_features, ignore_index=True)
    y = np.array(all_labels)
    profits = np.array(all_profits)
    print(f'\\nTotal samples: {len(X)}, Features: {X.shape[1]}')
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    profits_train = profits[:split_idx]
    
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.LongTensor(y_train)
    profits_train_t = torch.FloatTensor(profits_train)
    X_val_t = torch.FloatTensor(X_val.values)
    y_val_t = torch.LongTensor(y_val)
    
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_dataset = TensorDataset(X_train_t, y_train_t, profits_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    
    model = ProfitOptimizedNetwork(input_size=X.shape[1], hidden_size=256, num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = ProfitFocusedLoss(loss_weight=2.0)
    
    print('\\nTraining...')
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(100):
        model.train()
        train_loss = 0
        for batch_X, batch_y, batch_profits in train_loader:
            optimizer.zero_grad()
            direction, confidence = model(batch_X)
            loss = criterion(direction, batch_y, batch_profits)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_direction, _ = model(X_val_t)
            val_preds = torch.argmax(val_direction, dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
        
        scheduler.step(train_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/100: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}')
    
    model.load_state_dict(best_model_state)
    model.eval()
    
    complete_model = {
        'neural_network': model,
        'feature_engine': feature_engine,
        'training_date': pd.Timestamp.now(),
        'input_size': 80,
        'num_classes': 3,
        'model_class': 'ProfitOptimizedNetwork',
        'feature_names': feature_engine.feature_names,
        'training_symbols': list(all_data.keys()),
        'model_version': '2.0-profit-optimized'
    }
    
    with open('ultimate_neural_model.pkl', 'wb') as f:
        pickle.dump(complete_model, f)
    
    print('\\n' + '='*60)
    print('PROFIT-OPTIMIZED MODEL SAVED!')
    print('='*60)
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Trained on {len(all_data)} symbols with 6 months of data')
    print('File: ultimate_neural_model.pkl')
    
    return complete_model

if __name__ == '__main__':
    main()
