import MetaTrader5 as mt5
import pandas as pd
import datetime
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Output, Input
import plotly.graph_objects as go

# Inicializar MetaTrader 5
if not mt5.initialize():
    print("Error al inicializar MetaTrader 5")
    mt5.shutdown()
    exit()

# Configuración de símbolo y marcos temporales
symbol = "Volatility 100 Index"
timeframes = {
    'H1': mt5.TIMEFRAME_H1,
    'M1': mt5.TIMEFRAME_M1
}

# Cargar historial existente desde CSV
try:
    existing_df = pd.read_csv("operation_history.csv", parse_dates=['Hora Apertura', 'Hora Cierre'])
    operation_history = existing_df.to_dict('records')
except (FileNotFoundError, pd.errors.EmptyDataError):
    operation_history = []

# Variables globales inicializadas desde CSV
cumulative_profit = sum(op.get('Beneficio', 0) for op in operation_history)
current_balance = 100.0 + cumulative_profit
open_position = None
open_price = None
last_trend = None

def fetch_data(symbol, timeframe):
    utc_to = datetime.datetime.now()
    utc_from = utc_to - datetime.timedelta(minutes=300)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

def calculate_mean_value(h1_data, m1_data):
    combined_data = pd.merge_asof(
        m1_data.sort_values('time'),
        h1_data.sort_values('time'),
        on='time',
        suffixes=('_m1', '_h1')
    )
    combined_data['mean_close'] = (combined_data['close_m1'] + combined_data['close_h1']) / 2
    return combined_data

def identify_local_extrema(data):
    data['local_max'] = data['mean_close'].rolling(window=3, center=True).max() == data['mean_close']
    data['local_min'] = data['mean_close'].rolling(window=3, center=True).min() == data['mean_close']
    return data

def generate_figure(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['mean_close'],
        mode='lines', name='Valor Medio',
        line=dict(color='yellow')
    ))
    fig.add_trace(go.Scatter(
        x=data['time'][data['local_max']], y=data['mean_close'][data['local_max']],
        mode='markers', name='Máximos', marker=dict(color='red', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=data['time'][data['local_min']], y=data['mean_close'][data['local_min']],
        mode='markers', name='Mínimos', marker=dict(color='green', size=8)
    ))
    if open_position:
        fig.add_trace(go.Scatter(
            x=[data['time'].iloc[-1]], y=[open_price],
            mode='markers', name=f'Operación: {open_position}',
            marker=dict(color='blue', size=10)
        ))
    fig.update_layout(
        title=f"Precio en Tiempo Real - Saldo: ${current_balance:.2f}",
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_dark'
    )
    return fig

app = Dash(__name__)
app.layout = html.Div([
    html.H3("Bot de Trading Automático"),
    dcc.Graph(id='live-graph', style={'height': '500px'}),
    html.Div(id="balance-display", style={"font-size": "20px", "margin-top": "10px"}),
    html.Div(id="operation-history", style={"margin-top": "20px"}),
    dcc.Interval(id='update-interval', interval=20*1000, n_intervals=0)
])

@app.callback(
    [Output('live-graph', 'figure'),
     Output('balance-display', 'children'),
     Output('operation-history', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def update_graph_and_balance(n):
    global open_position, open_price, cumulative_profit, current_balance, last_trend, operation_history

    # Obtener y procesar datos
    h1_data = fetch_data(symbol, timeframes['H1'])
    m1_data = fetch_data(symbol, timeframes['M1'])
    if h1_data.empty or m1_data.empty:
        return go.Figure(), f"Saldo: ${current_balance:.2f}", "Cargando datos..."
    
    mean_data = calculate_mean_value(h1_data, m1_data)
    mean_data = identify_local_extrema(mean_data)
    current_price = mean_data['mean_close'].iloc[-1]

    # Determinar tendencia
    current_trend = "Lateral"
    if len(mean_data) >= 3:
        closes = mean_data['mean_close'].iloc[-3:]
        if closes.is_monotonic_increasing: current_trend = "Alcista"
        elif closes.is_monotonic_decreasing: current_trend = "Bajista"

    # Actualizar saldo en tiempo real
    if open_position == "Buy":
        current_balance = cumulative_profit + (current_price - open_price)
    elif open_position == "Sell":
        current_balance = cumulative_profit + (open_price - current_price)

    # Lógica de trading
    if current_trend != last_trend:
        if open_position is None:
            if current_trend in ["Alcista", "Bajista"]:
                open_position = "Buy" if current_trend == "Alcista" else "Sell"
                open_price = current_price
                operation_history.append({
                    'Tipo': open_position,
                    'Precio Apertura': open_price,
                    'Hora Apertura': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        else:
            profit = (current_price - open_price) if open_position == "Buy" else (open_price - current_price)
            cumulative_profit += profit
            current_balance = cumulative_profit
            operation_history[-1].update({
                'Precio Cierre': current_price,
                'Beneficio': profit,
                'Hora Cierre': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            open_position = None

    last_trend = current_trend

    # Guardar y mostrar datos
    pd.DataFrame(operation_history).to_csv("operation_history.csv", index=False)
    fig = generate_figure(mean_data)
    
    history_df = pd.DataFrame(operation_history)
    history_table = dash_table.DataTable(
        data=history_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in history_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'padding': '10px'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    ) if not history_df.empty else "No hay operaciones registradas"

    return fig, f"Saldo Actual: ${current_balance:.2f}", history_table

if __name__ == '__main__':
    app.run_server(debug=False)