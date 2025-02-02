import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import os
import time

# Configuración de archivos y timeframes
file_paths = {
    'D1': 'Volatility_100_Index_D1.csv',
    'H1': 'Volatility_100_Index_H1.csv',
    'M5': 'Volatility_100_Index_M5.csv'
}

# Inicializar conexión con MetaTrader 5
if not mt5.initialize():
    print("MetaTrader5 no pudo inicializarse.")
    quit()

# Variable global para almacenar transacciones
trade_history = []

# Procesar los datos de cada timeframe
def process_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
    else:
        return pd.DataFrame()  # Devuelve un DataFrame vacío si el archivo no existe
    
    data['time'] = pd.to_datetime(data['time'])
    # Suavizado con Savitzky-Golay
    data['smoothed_close'] = savgol_filter(data['close'], window_length=15, polyorder=3)
    # Gradiente y señal
    data['derivative'] = np.gradient(data['smoothed_close'])
    data['signal'] = (data['derivative'] > 0).astype(int)
    # Detectar cambios e incorporar la distancia
    data = detect_signal_changes(data)
    
    # Calcular la distancia de la señal
    data['signal_distance'] = np.nan  # Inicializar la columna
    change_indices = data.index[data['signal_change'] != 0].tolist()
    
    for idx in change_indices:
        if idx < data.index[-1]:
            next_idx = idx + 1
            current_close = data.loc[idx, 'close']
            next_close = data.loc[next_idx, 'close']
            
            if data.loc[idx, 'signal_change'] == 1:
                distance = next_close - current_close
            else:
                distance = current_close - next_close

            data.loc[idx, 'signal_distance'] = distance
    
    return data

# Detectar cambios en la señal
def detect_signal_changes(data):
    data['signal_change'] = data['signal'].diff().fillna(0)
    data['change_points'] = np.where(data['signal_change'] != 0, data['smoothed_close'], np.nan)
    data['after_change_block'] = (data['signal_change'] != 0).cumsum()
    
    # Identificar la primera fila de cada bloque (punto de cambio)
    data['is_first_in_block'] = data.groupby('after_change_block').cumcount() == 0
    
    # Calcular señales post-cambio excluyendo el primer elemento del bloque
    buy_signals = data[~data['is_first_in_block']].groupby('after_change_block')['signal'].apply(lambda x: (x == 1).sum())
    sell_signals = data[~data['is_first_in_block']].groupby('after_change_block')['signal'].apply(lambda x: (x == 0).sum())
    
    data['post_change_buy'] = data['after_change_block'].map(buy_signals)
    data['post_change_sell'] = data['after_change_block'].map(sell_signals)
    
    return data

# Actualizar datos de MetaTrader 5
def update_data(symbol, timeframe, file_path):
    timeframe_map = {
        'D1': mt5.TIMEFRAME_D1,
        'H1': mt5.TIMEFRAME_H1,
        'M5': mt5.TIMEFRAME_M5
    }
    rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, 500)
    if rates is None:
        print(f"Error obteniendo datos para {symbol} en {timeframe}.")
        return
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.to_csv(file_path, index=False)

# Simular el bot de trading
def simulate_trading(data, initial_balance=50, position_size=0.5):
    global trade_history
    balance = initial_balance
    position = 0

    for i, row in data.iterrows():
        if row['is_first_in_block']:
            continue  # Ignorar puntos de cambio
        
        # Lógica de compra
        if row['signal'] == 1:
            units = (balance * position_size) / row['close']
            balance -= units * row['close']
            trade_history.append({
                'time': row['time'],
                'action': 'BUY',
                'price': row['close'],
                'units': units,
                'balance': balance
            })
            position += units
        
        # Lógica de venta
        elif row['signal'] == 0 and position > 0:
            balance += position * row['close']
            trade_history.append({
                'time': row['time'],
                'action': 'SELL',
                'price': row['close'],
                'units': position,
                'balance': balance
            })
            position = 0

    # Cerrar posición al final si queda abierta
    if position > 0:
        balance += position * data.iloc[-1]['close']
        trade_history.append({
            'time': data.iloc[-1]['time'],
            'action': 'SELL',
            'price': data.iloc[-1]['close'],
            'units': position,
            'balance': balance
        })
    
    pd.DataFrame(trade_history).to_csv('realtime_trades.csv', index=False)

# Generar gráficos interactivos
def generate_figure(data, timeframe):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['time'], 
        y=data['close'],
        mode='lines', 
        name=f'Original ({timeframe})',
        line=dict(color='yellow'),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=data['time'], 
        y=data['smoothed_close'],
        mode='lines', 
        name=f'Suavizado ({timeframe})',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=data['time'][data['signal'] == 1], 
        y=data['smoothed_close'][data['signal'] == 1],
        mode='markers', 
        name='Compra',
        marker=dict(color='green', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=data['time'][data['signal'] == 0], 
        y=data['smoothed_close'][data['signal'] == 0],
        mode='markers', 
        name='Venta',
        marker=dict(color='red', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=data['time'][~data['change_points'].isna()], 
        y=data['change_points'].dropna(),
        mode='markers', 
        name='Cambios de Señal',
        marker=dict(color='orange', size=8, symbol='x')
    ))
    fig.update_layout(
        title=f'Gráfico {timeframe}',
        xaxis_title='Tiempo',
        yaxis_title='Precio de Cierre',
        template='plotly_dark',
        hovermode='x'
    )
    return fig

# Crear aplicación Dash
app = Dash(__name__)

app.layout = html.Div([
    # Intervalo de actualización cada 60 segundos
    dcc.Interval(id='update-interval', interval=60 * 1000),

    # Contenedor de gráficos en una sola fila
    html.Div(
        id='graphs-container',
        style={'display': 'flex', 'flex-direction': 'row'}  # Los 3 gráficos se muestran en horizontal
    )
])

# Callback para actualizar los 3 gráficos
@app.callback(
    Output('graphs-container', 'children'),
    Input('update-interval', 'n_intervals')
)
def update_graphs(n_intervals):
    # Símbolo de tu activo
    symbol = 'Volatility 100 Index'
    
    # Lista para ir almacenando cada "panel" (gráfico + texto)
    graph_list = []
    
    for timeframe in file_paths.keys():
        # 1. Actualizar datos
        update_data(symbol, timeframe, file_paths[timeframe])
        
        # 2. Procesar datos
        data = process_data(file_paths[timeframe])
        
        if data.empty:
            graph_list.append(
                html.Div(
                    f"No se pudieron cargar los datos para {timeframe}",
                    style={'color': 'red', 'margin': '10px', 'width': '33%'}
                )
            )
        else:
            # 3. Simular trading
            simulate_trading(data)
            
            # 4. Generar figura
            fig = generate_figure(data, timeframe)

            # 5. Calcular la métrica de distancias
            distances = data['signal_distance'].dropna()
            if not distances.empty:
                avg_distance = distances.mean()
            else:
                avg_distance = 0
            
            # 6. Crear un contenedor con el gráfico y el texto informativo
            graph_container = html.Div([
                dcc.Graph(figure=fig),
                html.Div(
                    f"Distancia promedio tras el cambio de señal: {avg_distance:.4f}",
                    style={'textAlign': 'center', 'marginTop': '10px'}
                )
            ], style={'width': '33%', 'margin': '0 5px'})
            
            graph_list.append(graph_container)
    
    return graph_list

# Ejecutar aplicación
if __name__ == '__main__':
    app.run_server(debug=True)