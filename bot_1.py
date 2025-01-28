import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Output, Input
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# Inicializar MetaTrader 5
if not mt5.initialize():
    print("Error al inicializar MetaTrader 5")
    mt5.shutdown()
    exit()

# Configuración de símbolo y marco temporal
symbol = "Volatility 100 Index"  # Cambia el símbolo si es necesario
timeframe = mt5.TIMEFRAME_M1  # Temporalidad 1 minuto
max_loss = 25  # Límite de pérdida máxima por operación

# Variables globales para operaciones
open_position = None
open_price = None
cumulative_profit = 0  # Beneficio acumulado
current_balance = 100.0  # Saldo inicial
operation_history = []
last_signal = None  # Última señal de la derivada (1 o 0)

# Función para obtener datos en tiempo real
def fetch_data(symbol, timeframe):
    utc_to = datetime.datetime.now()
    utc_from = utc_to - datetime.timedelta(minutes=300)  # Últimas 5 horas de datos
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

# Función para convertir datos de 1 minuto a 5 minutos
def convert_to_m5(data):
    data = data.copy()
    data.set_index('time', inplace=True)
    data = data.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'tick_volume': 'sum'
    }).dropna().reset_index()
    return data

# Función para procesar datos y generar señales
def process_data(data):
    if data.empty:
        return data
    
    # Suavizar los precios de cierre
    data['smoothed_close'] = savgol_filter(data['close'], window_length=7, polyorder=2)
    
    # Calcular la derivada
    data['derivative'] = np.gradient(data['smoothed_close'])
    
    # Generar señales basadas en la derivada
    data['signal'] = (data['derivative'] > 0).astype(int)
    
    # Detectar cambios en la señal (de 1 a 0 o de 0 a 1)
    data['signal_change'] = data['signal'].diff().fillna(0)
    
    return data

# Función para generar una gráfica
def generate_figure(data):
    fig = go.Figure()
    
    # Precios originales
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['close'],
        mode='lines', name='Precio Actual',
        line=dict(color='yellow')
    ))
    
    # Derivada diferencial
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['derivative'],
        mode='lines', name='Derivada Diferencial',
        line=dict(color='blue', width=1)
    ))
    
    # Señal filtrada
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['smoothed_close'],
        mode='lines', name='Señal Filtrada',
        line=dict(color='green', width=1)
    ))
    
    # Marcadores para operaciones de compra/venta
    for operation in operation_history:
        if 'Precio Apertura' in operation and 'Hora Apertura' in operation:
            # Marcador de apertura
            fig.add_trace(go.Scatter(
                x=[operation['Hora Apertura']],
                y=[operation['Precio Apertura']],
                mode='markers',
                name=f"Apertura {operation['Tipo']}",
                marker=dict(
                    color='green' if operation['Tipo'] == 'Buy' else 'red',
                    size=10,
                    symbol='triangle-up' if operation['Tipo'] == 'Buy' else 'triangle-down'
                )
            ))
        
        if 'Precio Cierre' in operation and 'Hora Cierre' in operation:
            # Marcador de cierre
            fig.add_trace(go.Scatter(
                x=[operation['Hora Cierre']],
                y=[operation['Precio Cierre']],
                mode='markers',
                name=f"Cierre {operation['Tipo']}",
                marker=dict(
                    color='blue',
                    size=10,
                    symbol='x'
                )
            ))
    
    # Línea de operación activa (si hay una operación abierta)
    if open_position:
        fig.add_trace(go.Scatter(
            x=[data['time'].iloc[-1], data['time'].iloc[-1]],
            y=[open_price, data['close'].iloc[-1]],
            mode='lines',
            name=f'Operación Activa: {open_position}',
            line=dict(color='blue', width=2, dash='dash')
        ))
    
    # Configuración del layout
    fig.update_layout(
        title=f"Precio en Tiempo Real (M5) - Saldo: ${current_balance:.2f}",
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_dark'
    )
    
    return fig

# Crear aplicación Dash
app = Dash(__name__)
app.layout = html.Div([
    dcc.Interval(id='update-interval', interval=20*1000),  # Actualizar cada 20 segundos
    html.H3("Bot de Trading en Tiempo Real - Volatility 100 Index"),
    html.Div(id='balance-display', style={'fontSize': '20px', 'margin': '10px'}),
    html.Div(id='graphs-and-table')
])

@app.callback(
    [Output('graphs-and-table', 'children'),
     Output('balance-display', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def update_content(n):
    global open_position, open_price, cumulative_profit, current_balance, last_signal
    
    # Obtener datos en tiempo real y convertir a M5
    data = fetch_data(symbol, timeframe)
    if data.empty:
        return html.Div(), f"Balance Actual: ${current_balance:.2f} | Operación Actual: {open_position or 'Ninguna'}"
    
    data = convert_to_m5(data)
    data = process_data(data)
    current_price = data['close'].iloc[-1]
    
    # Detectar cambios en la señal de la derivada
    current_signal = data['signal'].iloc[-1]
    signal_change = data['signal_change'].iloc[-1]
    
    # Actualizar saldo con el precio actual
    if open_position == "Buy":
        current_balance = cumulative_profit + (current_price - open_price)
    elif open_position == "Sell":
        current_balance = cumulative_profit + (open_price - current_price)
    else:
        current_balance = cumulative_profit
    
    # Simulación de operaciones basadas en el cambio de la derivada
    if signal_change != 0:  # Detectar cambio en la señal
        if open_position is None:  # Abrir posición
            if current_signal == 1:  # Derivada positiva (tendencia alcista)
                open_position = "Buy"
                open_price = current_price
                operation_history.append({
                    'Tipo': 'Buy',
                    'Precio Apertura': open_price,
                    'Hora Apertura': data['time'].iloc[-1]
                })
            elif current_signal == 0:  # Derivada negativa (tendencia bajista)
                open_position = "Sell"
                open_price = current_price
                operation_history.append({
                    'Tipo': 'Sell',
                    'Precio Apertura': open_price,
                    'Hora Apertura': data['time'].iloc[-1]
                })
        else:  # Cerrar posición existente
            if open_position == "Buy":
                profit = current_price - open_price
                cumulative_profit += profit
                current_balance = cumulative_profit
                operation_history[-1].update({
                    'Precio Cierre': current_price,
                    'Beneficio': profit,
                    'Hora Cierre': data['time'].iloc[-1]
                })
                open_position = None
                open_price = None
            elif open_position == "Sell":
                profit = open_price - current_price
                cumulative_profit += profit
                current_balance = cumulative_profit
                operation_history[-1].update({
                    'Precio Cierre': current_price,
                    'Beneficio': profit,
                    'Hora Cierre': data['time'].iloc[-1]
                })
                open_position = None
                open_price = None
    
    # Actualizar la última señal
    last_signal = current_signal
    
    # Generar gráfica
    figure = generate_figure(data)
    
    # Crear tabla de operaciones
    table = dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in operation_history[0].keys()] if operation_history else [],
        data=operation_history[-10:],
        style_table={'height': '300px', 'overflowY': 'auto'},
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
        style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
    )
    
    # Mostrar balance y operación actual
    balance_display = f"Balance Actual: ${current_balance:.2f} | Operación Actual: {open_position or 'Ninguna'}"
    
    return html.Div([dcc.Graph(figure=figure), table]), balance_display

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)