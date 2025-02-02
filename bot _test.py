import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from datetime import datetime
import plotly.graph_objects as go

# Variables globales
current_balance = 50.0
operations = []  # Lista para almacenar las operaciones
open_position = None
open_price = None
open_time = None
max_price = None
trade_size = 0.5
max_loss = 15
tendency = None
historical_file = "Crash_500_Index_M5.csv"

# Función para cargar datos históricos
def load_historical_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['time'])
        df['smoothed_close'] = savgol_filter(df['close'], window_length=7, polyorder=3)
        df['derivative'] = np.gradient(df['smoothed_close'])
        df['signal'] = (df['derivative'] > 0).astype(int)
        df['signal_change'] = df['signal'].diff().fillna(0)
        return df
    except FileNotFoundError:
        print("Archivo histórico no encontrado.")
        return pd.DataFrame()

# Función para abrir una operación
def open_trade(position, price, time):
    global open_position, open_price, open_time, max_price, current_balance
    open_position = position
    open_price = price
    open_time = time
    max_price = price
    current_balance -= trade_size * price
    print(f"Operación abierta: {position} a {price} en {time}")

# Función para cerrar una operación
def close_trade(close_price, close_time, reason):
    global open_position, open_price, open_time, max_price, current_balance, operations
    if open_position == 'buy':
        loss_or_gain = (close_price - open_price) * trade_size
    elif open_position == 'sell':
        loss_or_gain = (open_price - close_price) * trade_size
    else:
        loss_or_gain = 0

    max_loss_value = max(0, abs(close_price - open_price) * trade_size)
    result = 1 if loss_or_gain > 0 else 0
    current_balance += loss_or_gain + (trade_size * open_price)

    operations.append({
        'Hora Apertura': open_time,
        'Hora Cierre': close_time,
        'Tipo': f'{open_position.capitalize()} ({reason})',
        'Precio Apertura': open_price,
        'Precio Cierre': close_price,
        'Precio Máximo': max_price,
        'Cambio Balance': loss_or_gain,
        'Pérdida Máxima': max_loss_value,
        'Balance Final': current_balance,
        'Resultado': result
    })

    print(f"Operación cerrada: {open_position} a {close_price} en {close_time}. Razón: {reason}. Resultado: {'Ganada' if result == 1 else 'Perdida'}")

    open_position = None
    open_price = None
    open_time = None
    max_price = None

# Función para simular el trading
def simulate_trading(df):
    global tendency, open_position, open_price, open_time, max_price, current_balance

    for i, row in df.iterrows():
        current_price = row['close']
        current_derivative = row['derivative']
        current_time = row['time']

        new_tendency = 'buy' if current_derivative > 0 else 'sell'

        if new_tendency != tendency and open_position is not None:
            close_trade(current_price, current_time, reason="Cambio de tendencia")

        if open_position is None:
            open_trade(new_tendency, current_price, current_time)

        tendency = new_tendency

        if open_position == 'buy' and current_price < open_price - max_loss:
            close_trade(current_price, current_time, reason="Stop loss")
        elif open_position == 'sell' and current_price > open_price + max_loss:
            close_trade(current_price, current_time, reason="Stop loss")

        if open_position is not None:
            max_price = max(max_price, current_price) if open_position == 'buy' else min(max_price, current_price)

# Función para graficar resultados
def plot_operations(df, operations):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], mode='lines', name='Precio de Cierre'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['smoothed_close'], mode='lines', name='Filtro Suavizado', line=dict(dash='dot')))

    close_times = [op['Hora Cierre'] for op in operations]
    close_prices = [op['Precio Cierre'] for op in operations]
    results = [op['Resultado'] for op in operations]

    ganadas_times = [close_times[i] for i in range(len(results)) if results[i] == 1]
    ganadas_prices = [close_prices[i] for i in range(len(results)) if results[i] == 1]
    perdidas_times = [close_times[i] for i in range(len(results)) if results[i] == 0]
    perdidas_prices = [close_prices[i] for i in range(len(results)) if results[i] == 0]

    fig.add_trace(go.Scatter(x=ganadas_times, y=ganadas_prices, mode='markers', name='Ganadas', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=perdidas_times, y=perdidas_prices, mode='markers', name='Perdidas', marker=dict(color='red')))

    fig.update_layout(title='Resultados de Operaciones', xaxis_title='Tiempo', yaxis_title='Precio de Cierre')
    fig.show()

# Cargar datos históricos
df = load_historical_data(historical_file)

# Simular el trading en tiempo real
simulate_trading(df)

# Mostrar resultados de las operaciones
plot_operations(df, operations)

# Mostrar el balance final
print(f"Balance final: {current_balance}")
