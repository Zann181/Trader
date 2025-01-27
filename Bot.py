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
last_trend = None  # Última tendencia identificada ("Alcista" o "Bajista")

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

# Función para guardar historial en un archivo CSV
def save_to_csv():
    df = pd.DataFrame(operation_history)
    if not df.empty:
        df.to_csv("operation_history.csv", index=False)

# Función para generar una gráfica
def generate_figure(data):
    fig = go.Figure()

    # Precios originales
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['close'],
        mode='lines', name='Precio Actual',
        line=dict(color='yellow')
    ))

    # Máximos locales
    data['local_max'] = (data['high'] == data['high'].rolling(window=3, center=True).max())
    data['local_min'] = (data['low'] == data['low'].rolling(window=3, center=True).min())

    fig.add_trace(go.Scatter(
        x=data['time'][data['local_max']],
        y=data['high'][data['local_max']],
        mode='markers', name='Máximos Locales',
        marker=dict(color='red', size=8)
    ))

    fig.add_trace(go.Scatter(
        x=data['time'][data['local_min']],
        y=data['low'][data['local_min']],
        mode='markers', name='Mínimos Locales',
        marker=dict(color='green', size=8)
    ))

    # Punto de operación activa
    if open_position:
        fig.add_trace(go.Scatter(
            x=[data['time'].iloc[-1]],
            y=[open_price],
            mode='markers',
            name=f'Operación: {open_position}',
            marker=dict(color='blue', size=10)
        ))

    # Configuración del layout
    fig.update_layout(
        title=f"Precio en Tiempo Real (M5) - Saldo: ${current_balance:.2f}",
        xaxis=dict(rangeslider=dict(visible=True)),
        template='plotly_dark'
    )
    return fig

# Crear la aplicación Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H3("Bot de Trading Automático (M5)"),
    dcc.Graph(id='live-graph', style={'height': '500px'}),
    html.Div(id="balance-display", style={"font-size": "20px", "margin-top": "10px"}),
    html.Div(id="operation-history", style={"font-size": "16px", "margin-top": "20px"}),
    dcc.Interval(id='update-interval', interval=20*1000, n_intervals=0)  # Actualización cada 20 segundos
])

# Callback para actualizar la gráfica y mostrar historial
@app.callback(
    [Output('live-graph', 'figure'),
     Output('balance-display', 'children'),
     Output('operation-history', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def update_graph_and_balance(n_intervals):
    global open_position, open_price, cumulative_profit, current_balance, last_trend

    # Obtener datos en tiempo real y convertir a M5
    data = fetch_data(symbol, timeframe)
    if data.empty:
        return go.Figure(), f"Saldo Actual: ${current_balance:.2f}", "No hay operaciones registradas"

    data = convert_to_m5(data)
    current_price = data['close'].iloc[-1]

    # Determinar tendencia actual
    if len(data) >= 3:
        if data['close'].iloc[-1] > data['close'].iloc[-2] > data['close'].iloc[-3]:
            current_trend = "Alcista"
        elif data['close'].iloc[-1] < data['close'].iloc[-2] < data['close'].iloc[-3]:
            current_trend = "Bajista"
        else:
            current_trend = "Lateral"
    else:
        current_trend = None

    # Actualizar saldo con el precio actual
    if open_position == "Buy":
        current_balance = cumulative_profit + (current_price - open_price)
    elif open_position == "Sell":
        current_balance = cumulative_profit + (open_price - current_price)
    else:
        current_balance = cumulative_profit

    # Simulación de operaciones basadas en el cambio de tendencia
    if current_trend != last_trend:  # Detectar cambio de tendencia
        if open_position is None:  # Abrir posición
            if current_trend == "Alcista":
                open_position = "Buy"
                open_price = current_price
                operation_history.append({
                    'Tipo': 'Buy',
                    'Precio Apertura': open_price,
                    'Hora Apertura': data['time'].iloc[-1]
                })
            elif current_trend == "Bajista":
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

    # Actualizar la última tendencia
    last_trend = current_trend

    # Generar gráfica
    fig = generate_figure(data)

    # Guardar historial en CSV
    save_to_csv()

    # Mostrar historial de operaciones
    history = dash_table.DataTable(
        data=pd.DataFrame(operation_history).to_dict('records') if operation_history else [],
        columns=[
            {'name': 'Tipo', 'id': 'Tipo'},
            {'name': 'Precio Apertura', 'id': 'Precio Apertura'},
            {'name': 'Precio Cierre', 'id': 'Precio Cierre'},
            {'name': 'Beneficio', 'id': 'Beneficio'},
            {'name': 'Hora Apertura', 'id': 'Hora Apertura'},
            {'name': 'Hora Cierre', 'id': 'Hora Cierre'}
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    )

    return fig, f"Saldo Actual: ${current_balance:.2f}", history

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=False)
