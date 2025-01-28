import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, dash_table
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
    data['smoothed_close'] = savgol_filter(data['close'], window_length=15, polyorder=3)
    data['derivative'] = np.gradient(data['smoothed_close'])
    data['signal'] = (data['derivative'] > 0).astype(int)
    data = detect_signal_changes(data)
    return data

# Detectar cambios en la señal (transiciones de 1 a 0 o de 0 a 1)
def detect_signal_changes(data):
    data['signal_change'] = data['signal'].diff().fillna(0)
    data['change_points'] = np.where(data['signal_change'] != 0, data['smoothed_close'], np.nan)
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
    position = 0  # Representa la cantidad de activos comprados

    for i, row in data.iterrows():
        if row['signal_change'] == 1:  # Cambio de venta (0) a compra (1)
            units = (balance * position_size) / row['close']
            balance -= units * row['close']
            position += units
            trade_history.append({
                'time': row['time'], 'action': 'BUY', 'price': row['close'],
                'units': units, 'balance': balance
            })
        elif row['signal_change'] == -1 and position > 0:  # Cambio de compra (1) a venta (0)
            balance += position * row['close']
            trade_history.append({
                'time': row['time'], 'action': 'SELL', 'price': row['close'],
                'units': position, 'balance': balance
            })
            position = 0

    # Guardar resultados en un archivo CSV
    pd.DataFrame(trade_history).to_csv('realtime_trades.csv', index=False)

# Generar gráficos interactivos
def generate_figure(data, timeframe):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['close'],
        mode='lines', name=f'Original ({timeframe})',
        line=dict(color='yellow'), opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['smoothed_close'],
        mode='lines', name=f'Suavizado ({timeframe})',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=data['time'][data['signal'] == 1], y=data['smoothed_close'][data['signal'] == 1],
        mode='markers', name='Compra',
        marker=dict(color='green', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=data['time'][data['signal'] == 0], y=data['smoothed_close'][data['signal'] == 0],
        mode='markers', name='Venta',
        marker=dict(color='red', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=data['time'][~data['change_points'].isna()], y=data['change_points'].dropna(),
        mode='markers', name='Cambios de Señal',
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
    dcc.Interval(id='update-interval', interval=60*1000),  # Actualizar cada 60 segundos
    dcc.Dropdown(
        id='timeframe-selector',
        options=[{'label': tf, 'value': tf} for tf in file_paths.keys()],
        value='D1',
        clearable=False
    ),
    html.Div(id='graphs-and-table')
])

# Callback para actualizar gráficos y tablas
@app.callback(
    Output('graphs-and-table', 'children'),
    Input('timeframe-selector', 'value'),
    Input('update-interval', 'n_intervals')
)
def update_content(selected_timeframe, n_intervals):
    symbol = 'Volatility 100 Index'
    update_data(symbol, selected_timeframe, file_paths[selected_timeframe])
    data = process_data(file_paths[selected_timeframe])
    if data.empty:
        return html.Div("No se pudieron cargar los datos.")
    simulate_trading(data)
    figure = generate_figure(data, selected_timeframe)

    return html.Div([
        dcc.Graph(figure=figure),
        dash_table.DataTable(
            id=f'trade-table-{selected_timeframe}',
            columns=[
                {'name': 'Time', 'id': 'time'},
                {'name': 'Action', 'id': 'action'},
                {'name': 'Price', 'id': 'price'},
                {'name': 'Units', 'id': 'units'},
                {'name': 'Balance', 'id': 'balance'}
            ],
            data=trade_history,
            row_selectable='multi',
            style_table={'height': '300px', 'overflowY': 'auto'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
        )
    ])

# Ejecutar aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
