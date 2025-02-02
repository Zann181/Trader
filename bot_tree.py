import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Configuración inicial
trade_history = []  # Inicializamos el historial de operaciones vacío
current_balance = 50.0 + sum(op.get('Cambio Balance', 0) for op in trade_history)
open_position = None
open_price = None
max_price = None
open_price_time = None
trade_size = 0.5
max_loss = 15
tendency = None

# Función para obtener datos en tiempo real
def get_real_time_data(symbol, timeframe, num_candles=1400):
    if not mt5.initialize():
        print("Error al conectar con MetaTrader 5, crack.")
        return pd.DataFrame()
    
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
    }
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, num_candles)
    if rates is None or len(rates) == 0:
        print(f"Error obteniendo datos para {symbol} en {timeframe}, crack.")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Procesar datos
def process_data(data):
    if data.empty:
        print("No hay datos para procesar, crack.")
        return data
    data['smoothed_close'] = savgol_filter(data['close'], window_length=21, polyorder=3)
    data['derivative'] = np.gradient(data['smoothed_close'])
    data['signal'] = (data['derivative'] > 0).astype(int)
    data['signal_change'] = data['signal'].diff().fillna(0)
    data['change_points'] = np.where(data['signal_change'] != 0, data['smoothed_close'], np.nan)
    return data

# Calcular volatilidad con desviación estándar del rango (High - Low)
def calculate_volatility(data, window=20):
    data['range'] = data['high'] - data['low']
    data['volatility'] = data['range'].rolling(window=window).std()
    return data

# Construcción del árbol de decisión
def build_decision_tree(data):
    data = data.dropna()
    features = ['smoothed_close', 'derivative', 'volatility']
    target = 'signal_change'
    
    X = data[features]
    y = data[target]
    
    tree = DecisionTreeClassifier(max_depth=4)
    tree.fit(X, y)
    
    return tree

# Integración de todo el flujo
def main(symbol='Crash 500 Index', timeframe='M5'):
    # Obtener datos
    data = get_real_time_data(symbol, timeframe)
    if data.empty:
        return
    
    # Procesar datos
    data = process_data(data)
    data = calculate_volatility(data)
    
    # Construir árbol de decisión
    tree = build_decision_tree(data)
    
    # Visualización de resultados
    plt.figure(figsize=(12, 6))
    plt.plot(data['time'], data['smoothed_close'], label='Precio Filtrado')
    plt.scatter(data['time'], data['change_points'], color='red', label='Cambio de Tendencia')
    plt.legend()
    plt.show()
    
    return tree

# Ejecutar
if __name__ == "__main__":
    decision_tree = main()
