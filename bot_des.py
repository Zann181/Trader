import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, dash_table, State
import plotly.graph_objects as go
from scipy.signal import savgol_filter

# Inicializar conexión con MetaTrader 5
if not mt5.initialize():
    print("MetaTrader5 no pudo inicializarse.")
    quit()

# Variables globales
try:
    existing_df = pd.read_csv("real_crash_trades_M5.csv", parse_dates=['Hora Apertura', 'Hora Cierre'])
    trade_history = existing_df.to_dict('records')
except (FileNotFoundError, pd.errors.EmptyDataError):
    trade_history = []

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
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
    }
    rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, num_candles)
    if rates is None:
        print(f"Error obteniendo datos para {symbol} en {timeframe}.")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Procesar datos
def process_data(data):
    if data.empty:
        return data
    data['smoothed_close'] = savgol_filter(data['close'], window_length=7, polyorder=2)
    data['derivative'] = np.gradient(data['smoothed_close'])
    data['signal'] = (data['derivative'] > 0).astype(int)
    data['signal_change'] = data['signal'].diff().fillna(0)
    data['change_points'] = np.where(data['signal_change'] != 0, data['smoothed_close'], np.nan)
    return data

# Lógica de trading
def execute_trading_logic(data):
    global open_position, open_price, max_price, current_balance, trade_history, open_price_time, tendency

    if data.empty:
        return

    latest_row = data.iloc[-1]
    current_price = latest_row['close']
    current_time = latest_row['time']

    # Determinar tendencia
    tendency = 'Alcista' if latest_row['signal'] == 1 else 'Bajista'

    # Cerrar operación por pérdida máxima
    if open_position:
        potential_loss = (open_price - current_price) * trade_size if open_position == 'Buy' else (current_price - open_price) * trade_size

        if potential_loss >= max_loss:
            close_trade(current_price, current_time, reason='Pérdida')
            return

    # Detectar señales usando change_points
    if not open_position:
        if not np.isnan(latest_row['change_points']):
            if latest_row['signal'] == 1:
                open_trade('Buy', current_price, current_time)
            elif latest_row['signal'] == 0:
                open_trade('Sell', current_price, current_time)

    # Actualizar precio máximo/mínimo
    if open_position:
        if open_position == 'Buy':
            max_price = max(max_price, current_price)
            if not np.isnan(latest_row['change_points']) and latest_row['signal'] == 0:
                close_trade(current_price, current_time)
        else:
            max_price = min(max_price, current_price)
            if not np.isnan(latest_row['change_points']) and latest_row['signal'] == 1:
                close_trade(current_price, current_time)

# Abrir operación
def open_trade(position, price, time):
    global open_position, open_price, max_price, open_price_time
    open_position = position
    open_price = price
    max_price = price
    open_price_time = time

# Cerrar operación
def close_trade(close_price, close_time, reason='Señal'):
    global open_position, open_price, max_price, current_balance, trade_history, open_price_time

    if open_position is None:
        return

    change = (close_price - open_price) * trade_size if open_position == 'Buy' else (open_price - close_price) * trade_size

    operation = {
        'Hora Apertura': open_price_time,
        'Hora Cierre': close_time,
        'Tipo': f'{open_position} ({reason})',
        'Precio Apertura': open_price,
        'Precio Cierre': close_price,
        'Precio Máximo': max_price,
        'Cambio Balance': change,
        'Pérdida Máxima': abs((open_price - close_price) * trade_size),
        'Balance Final': current_balance + change
    }

    current_balance += change
    trade_history.append(operation)
    open_position = None
    open_price = None
    max_price = None
    open_price_time = None

    # Guardar en CSV
    pd.DataFrame(trade_history).to_csv('real_crash_trades_M5.csv', index=False)

# Generar gráfico
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
        x=data['time'][~data['change_points'].isna()],
        y=data['change_points'].dropna(),
        mode='markers', name='Cambio de Señal',
        marker=dict(color='orange', size=8, symbol='x')
    ))
    fig.update_layout(
        title=f'Gráfico {timeframe} - Balance: ${current_balance:.2f}',
        xaxis_title='Tiempo',
        yaxis_title='Precio de Cierre',
        template='plotly_dark',
        hovermode='x'
    )
    return fig

# Crear aplicación Dash
app = Dash(__name__)
app.layout = html.Div([
    dcc.Interval(id='update-interval', interval=20*1000),
    html.H3("Bot de Trading en Tiempo Real - Crash 500 Index"),
    html.Div(id='balance-display', style={'fontSize': '20px', 'margin': '10px'}),
    html.Div([
        html.Button('Abrir Operación Manual', id='manual-trade-btn', n_clicks=0),
        html.Div(id='manual-trade-feedback', style={'marginTop': '10px'})
    ]),
    html.Div(id='graphs-and-table')
])

@app.callback(
    [Output('graphs-and-table', 'children'),
     Output('balance-display', 'children'),
     Output('manual-trade-feedback', 'children')],
    [Input('update-interval', 'n_intervals'),
     Input('manual-trade-btn', 'n_clicks')]
)
def update_content(n, manual_clicks):
    global open_price_time

    symbol = 'Crash 500 Index'
    timeframe = 'M5'
    data = get_real_time_data(symbol, timeframe)

    if not data.empty:
        data = process_data(data)
        execute_trading_logic(data)
        if open_position and not open_price_time:
            open_price_time = data.iloc[-1]['time']

    figure = generate_figure(data, timeframe) if not data.empty else go.Figure()

    table = dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in trade_history[0].keys()] if trade_history else [],
        data=trade_history[-10:],
        style_table={'height': '300px', 'overflowY': 'auto'},
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
        style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
    )

    balance_display = f"Balance Actual: ${current_balance:.2f} | Operación Actual: {open_position or 'Ninguna'}"

    feedback = ""
    if manual_clicks > 0:
        if not open_position:
            open_trade('Buy' if tendency == 'Alcista' else 'Sell', data.iloc[-1]['close'], data.iloc[-1]['time'])
            feedback = f"Operación manual abierta: {'Buy' if tendency == 'Alcista' else 'Sell'}"
        else:
            feedback = "Ya hay una operación abierta."

    return html.Div([dcc.Graph(figure=figure), table]), balance_display, feedback

if __name__ == '__main__':
    app.run_server(debug=True)