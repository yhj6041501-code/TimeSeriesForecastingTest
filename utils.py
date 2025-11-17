"""
비트코인 변화 방향 예측을 위한 유틸리티 함수들
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# 영문 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_bitcoin_data(start_date='2020-01-01', end_date=None):
    """
    yfinance를 사용하여 비트코인 데이터를 다운로드합니다.
    
    Parameters:
    -----------
    start_date : str
        시작 날짜 (YYYY-MM-DD 형식)
    end_date : str
        종료 날짜 (YYYY-MM-DD 형식), None이면 오늘 날짜
        
    Returns:
    --------
    pd.DataFrame
        비트코인 가격 데이터
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"비트코인 데이터 다운로드 중: {start_date} ~ {end_date}")
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
    print(f"다운로드 완료: {len(btc_data)} 행")
    
    return btc_data


def create_features(df, lookback_days=7):
    """
    가격 데이터로부터 특성을 생성합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        가격 데이터
    lookback_days : int
        과거 몇 일치 데이터를 사용할지
        
    Returns:
    --------
    pd.DataFrame
        특성이 추가된 데이터프레임
    """
    data = df.copy()
    
    # MultiIndex 컬럼을 단순화 (yfinance가 MultiIndex를 반환하는 경우)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # 각 컬럼을 Series로 명시적으로 변환 (squeeze 사용)
    close = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()
    volume = data['Volume'].squeeze()
    
    # 1. 기본 가격 변화율
    data['Returns'] = close.pct_change()
    
    # 2. 이동평균
    for window in [5, 10, 20, 50]:
        ma = close.rolling(window=window).mean()
        data[f'MA_{window}'] = ma
        data[f'MA_{window}_ratio'] = close / ma
    
    # 3. 변동성 (표준편차)
    returns = data['Returns'].squeeze()
    for window in [5, 10, 20]:
        data[f'Volatility_{window}'] = returns.rolling(window=window).std()
    
    # 4. 거래량 변화
    data['Volume_Change'] = volume.pct_change()
    volume_ma_5 = volume.rolling(window=5).mean()
    data['Volume_MA_5'] = volume_ma_5
    data['Volume_ratio'] = volume / volume_ma_5
    
    # 5. 고가-저가 범위
    data['High_Low_Range'] = (high - low) / close
    
    # 6. RSI (Relative Strength Index)
    data['RSI_14'] = calculate_rsi(close, period=14)
    
    # 7. MACD
    macd, macd_signal = calculate_macd(close)
    data['MACD'] = macd
    data['MACD_Signal'] = macd_signal
    
    # 8. 과거 수익률 (lag features)
    for lag in range(1, lookback_days + 1):
        data[f'Returns_Lag_{lag}'] = returns.shift(lag)
    
    # 9. 타겟 변수: 다음 날 가격이 오를지 (1) 내릴지 (0)
    data['Target'] = (close.shift(-1) > close).astype(int)
    
    return data


def calculate_rsi(prices, period=14):
    """
    RSI (Relative Strength Index)를 계산합니다.
    
    Parameters:
    -----------
    prices : pd.Series
        가격 시계열
    period : int
        RSI 계산 기간
        
    Returns:
    --------
    pd.Series
        RSI 값
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence)를 계산합니다.
    
    Parameters:
    -----------
    prices : pd.Series
        가격 시계열
    fast : int
        빠른 EMA 기간
    slow : int
        느린 EMA 기간
    signal : int
        시그널 라인 기간
        
    Returns:
    --------
    tuple
        (MACD, Signal Line)
    """
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    return macd, signal_line


def prepare_data(data, test_size=0.2, validation_size=0.1):
    """
    학습, 검증, 테스트 데이터로 분할합니다.
    
    Parameters:
    -----------
    data : pd.DataFrame
        특성이 포함된 전체 데이터
    test_size : float
        테스트 데이터 비율
    validation_size : float
        검증 데이터 비율
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # NaN 제거
    data_clean = data.dropna()
    
    # 특성과 타겟 분리
    feature_columns = [col for col in data_clean.columns 
                      if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    X = data_clean[feature_columns]
    y = data_clean['Target']
    
    # 시계열 데이터이므로 순차적으로 분할
    train_size = int(len(X) * (1 - test_size - validation_size))
    val_size = int(len(X) * validation_size)
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]
    
    print(f"학습 데이터: {len(X_train)} 샘플")
    print(f"검증 데이터: {len(X_val)} 샘플")
    print(f"테스트 데이터: {len(X_test)} 샘플")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    모델 성능을 평가합니다.
    
    Parameters:
    -----------
    y_true : array-like
        실제 값
    y_pred : array-like
        예측 값
    model_name : str
        모델 이름
        
    Returns:
    --------
    dict
        평가 메트릭
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"{model_name} 성능 평가")
    print(f"{'='*50}")
    print(f"정확도 (Accuracy):  {accuracy:.4f}")
    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall):    {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"{'='*50}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """
    혼동 행렬을 시각화합니다.
    
    Parameters:
    -----------
    y_true : array-like
        실제 값
    y_pred : array-like
        예측 값
    model_name : str
        모델 이름
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], 
                yticklabels=['Down', 'Up'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def compare_models(results_dict):
    """
    여러 모델의 성능을 비교합니다.
    
    Parameters:
    -----------
    results_dict : dict
        {모델명: 평가메트릭} 형태의 딕셔너리
    """
    df_results = pd.DataFrame(results_dict).T
    
    print("\n" + "="*70)
    print("모델 성능 비교")
    print("="*70)
    print(df_results.to_string())
    print("="*70 + "\n")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        df_results[metric].plot(kind='bar', ax=ax, color=color, alpha=0.7)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Model')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
    plt.tight_layout()
    plt.show()


def plot_price_and_predictions(dates, actual_prices, predictions, model_name='Model'):
    """
    실제 가격과 예측 결과를 시각화합니다.
    
    Parameters:
    -----------
    dates : array-like
        날짜
    actual_prices : array-like
        실제 가격
    predictions : array-like
        예측 값 (0 또는 1)
    model_name : str
        모델 이름
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 가격 차트
    ax1.plot(dates, actual_prices, label='BTC Price', color='blue', alpha=0.7)
    ax1.set_ylabel('Price (USD)')
    ax1.set_title(f'{model_name} - Bitcoin Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 예측 결과
    colors = ['red' if p == 0 else 'green' for p in predictions]
    ax2.scatter(dates, predictions, c=colors, alpha=0.5, s=10)
    ax2.set_ylabel('Prediction')
    ax2.set_xlabel('Date')
    ax2.set_title(f'{model_name} - Predictions (Green: Up, Red: Down)')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Down', 'Up'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_trading_profit(y_true, y_pred, initial_capital=10000):
    """
    예측 기반 거래 전략의 수익을 계산합니다.
    
    Parameters:
    -----------
    y_true : array-like
        실제 방향 (0: 하락, 1: 상승)
    y_pred : array-like
        예측 방향 (0: 하락, 1: 상승)
    initial_capital : float
        초기 자본
        
    Returns:
    --------
    dict
        거래 결과
    """
    capital = initial_capital
    positions = []
    
    for true, pred in zip(y_true, y_pred):
        if pred == 1:  # 상승 예측 시 매수
            if true == 1:  # 실제로 상승
                capital *= 1.01  # 1% 수익 (단순화)
            else:  # 실제로 하락
                capital *= 0.99  # 1% 손실
        # pred == 0이면 거래하지 않음
        
        positions.append(capital)
    
    total_return = (capital - initial_capital) / initial_capital * 100
    
    result = {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'positions': positions
    }
    
    print(f"\n{'='*50}")
    print(f"거래 전략 수익률")
    print(f"{'='*50}")
    print(f"초기 자본: ${initial_capital:,.2f}")
    print(f"최종 자본: ${capital:,.2f}")
    print(f"수익률: {total_return:.2f}%")
    print(f"{'='*50}\n")
    
    return result


def simulate_trading_strategy(predictions, actual_prices, dates, initial_capital=10000, 
                             transaction_fee=0.001, strategy_type='simple'):
    """
    실제 가격 데이터를 사용한 트레이딩 시뮬레이션
    
    Parameters:
    -----------
    predictions : array-like
        모델의 예측 (0: 하락 예측 -> 매도/관망, 1: 상승 예측 -> 매수)
    actual_prices : array-like
        실제 가격 데이터
    dates : array-like
        날짜 데이터
    initial_capital : float
        초기 자본 ($)
    transaction_fee : float
        거래 수수료 비율 (기본 0.1%)
    strategy_type : str
        'simple': 단순 매수/관망
        'long_short': 롱/숏 전략
        
    Returns:
    --------
    dict
        트레이딩 결과 상세 정보
    """
    cash = initial_capital
    btc_holdings = 0  # 보유 BTC 수량
    portfolio_values = []
    trade_log = []
    
    for i in range(len(predictions)):
        current_price = actual_prices[i]
        pred = predictions[i]
        
        # 포트폴리오 가치 = 현금 + BTC 보유량 * 현재가
        portfolio_value = cash + btc_holdings * current_price
        portfolio_values.append(portfolio_value)
        
        # 마지막 날은 거래하지 않음
        if i == len(predictions) - 1:
            # 보유 BTC 전량 매도
            if btc_holdings > 0:
                sell_value = btc_holdings * current_price * (1 - transaction_fee)
                cash += sell_value
                trade_log.append({
                    'date': dates[i],
                    'action': 'SELL_ALL',
                    'price': current_price,
                    'amount': btc_holdings,
                    'value': sell_value,
                    'cash': cash,
                    'portfolio_value': cash
                })
                btc_holdings = 0
            continue
        
        if strategy_type == 'simple':
            # 단순 전략: 상승 예측 시 매수, 하락 예측 시 보유 BTC 매도
            if pred == 1:  # 상승 예측
                if cash > 0:
                    # 전체 현금으로 BTC 매수
                    buy_amount = (cash * (1 - transaction_fee)) / current_price
                    btc_holdings += buy_amount
                    trade_log.append({
                        'date': dates[i],
                        'action': 'BUY',
                        'price': current_price,
                        'amount': buy_amount,
                        'value': cash,
                        'cash': 0,
                        'portfolio_value': portfolio_value
                    })
                    cash = 0
            else:  # 하락 예측
                if btc_holdings > 0:
                    # 보유 BTC 전량 매도
                    sell_value = btc_holdings * current_price * (1 - transaction_fee)
                    cash += sell_value
                    trade_log.append({
                        'date': dates[i],
                        'action': 'SELL',
                        'price': current_price,
                        'amount': btc_holdings,
                        'value': sell_value,
                        'cash': cash,
                        'portfolio_value': portfolio_value
                    })
                    btc_holdings = 0
    
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # 벤치마크 계산 (Buy and Hold)
    btc_buy_and_hold = (initial_capital * (1 - transaction_fee)) / actual_prices[0]
    buy_and_hold_value = btc_buy_and_hold * actual_prices[-1] * (1 - transaction_fee)
    buy_and_hold_return = (buy_and_hold_value - initial_capital) / initial_capital * 100
    
    # 총 거래 금액 계산 (수수료 부과 기준)
    total_trade_volume = sum(trade['value'] for trade in trade_log if trade['action'] in ['BUY', 'SELL'])
    total_fees_paid = total_trade_volume * transaction_fee

    result = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'buy_and_hold_return': buy_and_hold_return,
        'excess_return': total_return - buy_and_hold_return,
        'portfolio_values': portfolio_values,
        'trade_log': trade_log,
        'num_trades': len(trade_log),
        'total_trade_volume': total_trade_volume,
        'total_fees_paid': total_fees_paid,
        'dates': dates
    }
    
    return result


def calculate_buy_and_hold_return(prices, initial_capital=10000, transaction_fee=0.001):
    """
    Buy and Hold 전략의 수익률 계산
    
    Parameters:
    -----------
    prices : array-like
        가격 데이터
    initial_capital : float
        초기 자본
    transaction_fee : float
        거래 수수료
        
    Returns:
    --------
    dict
        수익률 정보
    """
    # 처음에 전액 매수
    btc_amount = (initial_capital * (1 - transaction_fee)) / prices[0]
    
    # 마지막에 전액 매도
    final_value = btc_amount * prices[-1] * (1 - transaction_fee)
    
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'strategy': 'Buy and Hold'
    }


def compare_trading_strategies(results_dict):
    """
    여러 트레이딩 전략의 수익률을 비교합니다.

    Parameters:
    -----------
    results_dict : dict
        {전략명: 결과} 형태의 딕셔너리
    """
    comparison_data = []

    for strategy_name, result in results_dict.items():
        comparison_data.append({
            'Strategy': strategy_name,
            'Initial Capital': f"${result['initial_capital']:,.2f}",
            'Final Value': f"${result['final_value']:,.2f}",
            'Total Return (%)': f"{result['total_return']:.2f}",
            'Num Trades': result.get('num_trades', 'N/A'),
            'Total Fees': f"${result.get('total_fees_paid', 0):,.2f}" if 'total_fees_paid' in result else 'N/A'
        })

    df = pd.DataFrame(comparison_data)

    print("\n" + "="*100)
    print("트레이딩 전략 수익률 비교")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")

    return df


def plot_trading_results(results_dict):
    """
    트레이딩 결과를 시각화합니다.
    
    Parameters:
    -----------
    results_dict : dict
        {전략명: 결과} 형태의 딕셔너리
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. 포트폴리오 가치 변화
    for strategy_name, result in results_dict.items():
        if 'portfolio_values' in result:
            dates = result.get('dates', range(len(result['portfolio_values'])))
            axes[0].plot(dates, result['portfolio_values'], 
                        label=f"{strategy_name} (Return: {result['total_return']:.2f}%)",
                        linewidth=2)
    
    axes[0].axhline(y=results_dict[list(results_dict.keys())[0]]['initial_capital'], 
                   color='black', linestyle='--', linewidth=1, label='Initial Capital')
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 수익률 비교 바 차트
    strategies = list(results_dict.keys())
    returns = [results_dict[s]['total_return'] for s in strategies]
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    axes[1].bar(strategies, returns, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title('Total Return Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Return (%)')
    axes[1].set_xlabel('Strategy')
    axes[1].grid(axis='y', alpha=0.3)
    
    # 값 표시
    for i, (strategy, ret) in enumerate(zip(strategies, returns)):
        axes[1].text(i, ret, f'{ret:.2f}%', ha='center', 
                    va='bottom' if ret > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def print_trade_log(trade_log, max_rows=10):
    """
    거래 로그를 출력합니다.
    
    Parameters:
    -----------
    trade_log : list
        거래 기록 리스트
    max_rows : int
        출력할 최대 행 수
    """
    if len(trade_log) == 0:
        print("거래 기록이 없습니다.")
        return
    
    df = pd.DataFrame(trade_log)
    
    print(f"\n{'='*80}")
    print(f"거래 로그 (총 {len(trade_log)}건)")
    print(f"{'='*80}")
    
    if len(df) <= max_rows:
        print(df.to_string(index=False))
    else:
        print(f"\n처음 {max_rows//2}건:")
        print(df.head(max_rows//2).to_string(index=False))
        print(f"\n...")
        print(f"\n마지막 {max_rows//2}건:")
        print(df.tail(max_rows//2).to_string(index=False))
    
    print(f"{'='*80}\n")


# PyTorch 모델 클래스들
class LSTMModel(nn.Module):
    """
    LSTM 기반 이진 분류 모델
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, 
                            batch_first=True, dropout=0)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, num_layers=1,
                            batch_first=True, dropout=0)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        
        self.fc1 = nn.Linear(hidden_size//2, 16)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)
        # BatchNorm: (batch, features, seq_len)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.bn1(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out[:, -1, :])  # Take last output
        lstm_out = self.bn2(lstm_out)
        
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class GRUModel(nn.Module):
    """
    GRU 기반 이진 분류 모델
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers=1,
                          batch_first=True, dropout=0)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, num_layers=1,
                          batch_first=True, dropout=0)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        
        self.fc1 = nn.Linear(hidden_size//2, 16)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        gru_out, _ = self.gru1(x)
        gru_out = self.dropout1(gru_out)
        # BatchNorm: (batch, features, seq_len)
        gru_out = gru_out.permute(0, 2, 1)
        gru_out = self.bn1(gru_out)
        gru_out = gru_out.permute(0, 2, 1)
        
        gru_out, _ = self.gru2(gru_out)
        gru_out = self.dropout2(gru_out[:, -1, :])  # Take last output
        gru_out = self.bn2(gru_out)
        
        out = self.fc1(gru_out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


def train_pytorch_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """
    PyTorch 모델 학습
    
    Parameters:
    -----------
    model : nn.Module
        학습할 모델
    train_loader : DataLoader
        학습 데이터 로더
    val_loader : DataLoader
        검증 데이터 로더
    epochs : int
        에폭 수
    lr : float
        학습률
    patience : int
        Early stopping patience
        
    Returns:
    --------
    dict
        학습 히스토리
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted.squeeze() == batch_y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted.squeeze() == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def predict_pytorch_model(model, data_loader):
    """
    PyTorch 모델로 예측
    
    Parameters:
    -----------
    model : nn.Module
        학습된 모델
    data_loader : DataLoader
        데이터 로더
        
    Returns:
    --------
    tuple
        (예측 확률, 이진 예측)
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    binary_predictions = (predictions > 0.5).astype(int).flatten()
    
    return predictions.flatten(), binary_predictions

