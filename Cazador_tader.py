import MetaTrader5 as mt5
import pandas as pd
import datetime
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objects as go
import math
from scipy.stats import norm

# Inicializar MetaTrader 5
if not mt5.initialize():
    print("Error al inicializar MetaTrader 5")
    mt5.shutdown()
    exit()

# Configuración de símbolo y marcos temporales
symbol = "Volatility 100 Index"  # Reemplaza con el símbolo deseado
timeframes = {
    'D1': mt5.TIMEFRAME_D1,  # Marco temporal de 1 día
    'H1': mt5.TIMEFRAME_H1,  # Marco temporal de 1 hora
    'M1': mt5.TIMEFRAME_M1   # Usaremos M1 para generar M5
}


# Configuración de símbolo y marco temporal
symbol = "Volatility 100 Index"  # Cambia el símbolo si es necesario
timeframe = mt5.TIMEFRAME_M5  # Temporalidad 5 minutos
max_loss = 25  # Límite de pérdida máxima por operación

# Variables globales para operaciones
open_position = None
open_price = None
cumulative_profit = 0  # Beneficio acumulado
operation_history = []



# Función para obtener datos en tiempo real
def fetch_data(symbol, timeframe):
    utc_to = datetime.datetime.now()
    utc_from = utc_to - datetime.timedelta(days=2)  # Últimos 2 días de datos
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data

# Función para simular compras y ventas
def simulate_trades(data):
    global open_position, open_price, cumulative_profit, operation_history

    # Calcular máximos y mínimos locales
    data['local_max'] = data['close'][(data['close'].shift(1) < data['close']) & (data['close'].shift(-1) < data['close'])]
    data['local_min'] = data['close'][(data['close'].shift(1) > data['close']) & (data['close'].shift(-1) > data['close'])]

    for i in range(len(data)):
        current_price = data['close'].iloc[i]
        time_now = data['time'].iloc[i]

        if open_position is None:  # Abrir posición
            if not pd.isna(data['local_min'].iloc[i]):  # Señal de compra
                open_position = "Buy"
                open_price = current_price
                operation_history.append({
                    'Tipo': 'Buy',
                    'Precio': open_price,
                    'Hora': time_now
                })
            elif not pd.isna(data['local_max'].iloc[i]):  # Señal de venta
                open_position = "Sell"
                open_price = current_price
                operation_history.append({
                    'Tipo': 'Sell',
                    'Precio': open_price,
                    'Hora': time_now
                })
        else:  # Cerrar posición
            if open_position == "Buy" and (not pd.isna(data['local_max'].iloc[i]) or current_price - open_price <= -max_loss):
                profit = current_price - open_price
                cumulative_profit += profit
                operation_history[-1].update({'Cierre': current_price, 'Beneficio': profit, 'Hora Cierre': time_now})
                open_position = None
                open_price = None
            elif open_position == "Sell" and (not pd.isna(data['local_min'].iloc[i]) or open_price - current_price <= -max_loss):
                profit = open_price - current_price
                cumulative_profit += profit
                operation_history[-1].update({'Cierre': current_price, 'Beneficio': profit, 'Hora Cierre': time_now})
                open_position = None
                open_price = None

    return data







# Función para convertir datos de M1 a M5
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

# Función para obtener y procesar los datos
def fetch_and_update_data(timeframe_name, timeframe):
    csv_file = f"{symbol.replace(' ', '_')}_{timeframe_name}.csv"
    try:
        existing_data = pd.read_csv(csv_file)
        existing_data['time'] = pd.to_datetime(existing_data['time'])
        last_time = existing_data['time'].max()
    except FileNotFoundError:
        existing_data = pd.DataFrame()
        last_time = datetime.datetime.now() - datetime.timedelta(days=60)
    
    utc_from = last_time + datetime.timedelta(seconds=1)
    utc_to = datetime.datetime.now()
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        return existing_data
    
    new_data = pd.DataFrame(rates)
    new_data['time'] = pd.to_datetime(new_data['time'], unit='s')
    if timeframe_name == 'M5':
        combined_data = convert_to_m5(pd.concat([existing_data, new_data]))
    else:
        combined_data = pd.concat([existing_data, new_data])

    combined_data = combined_data.drop_duplicates(subset=['time']).reset_index(drop=True)
    combined_data.to_csv(csv_file, index=False)
    return combined_data

# Función para generar una gráfica
def generate_figure(data, timeframe_name):
    # Ajustar suavizado: Menos agresivo para mantener más detalles
    sigma_value = 2 if timeframe_name == 'D1' else 10
    data['smoothed_close'] = gaussian_filter1d(data['close'], sigma=sigma_value, mode='nearest')
    data['moving_average_70'] = data['close'].rolling(window=70).mean()

    # Calcular máximos y mínimos locales
    smoothed_close_prices = data['smoothed_close'].values
    local_maxima = argrelextrema(smoothed_close_prices, np.greater)[0]
    local_minima = argrelextrema(smoothed_close_prices, np.less)[0]

    # Asignar máximos y mínimos a las columnas correspondientes
    data['local_max'] = np.nan
    data['local_min'] = np.nan
    data.loc[data.index[local_maxima], 'local_max'] = data['smoothed_close'].iloc[local_maxima]
    data.loc[data.index[local_minima], 'local_min'] = data['smoothed_close'].iloc[local_minima]

    # Crear la figura
    fig = go.Figure()

    # Precios originales
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['close'],
        mode='lines', name=f'Precios Originales ({timeframe_name})',
        line=dict(color='yellow'), opacity=0.5
    ))

    # Precios suavizados
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['smoothed_close'],
        mode='lines', name=f'Precios Suavizados ({timeframe_name})',
        line=dict(color='blue')
    ))

    # Promedio móvil
    fig.add_trace(go.Scatter(
        x=data['time'], y=data['moving_average_70'],
        mode='lines', name=f'Promedio Móvil (70)',
        line=dict(color='purple')
    ))

    # Máximos locales
    fig.add_trace(go.Scatter(
        x=data['time'][~data['local_max'].isna()], y=data['local_max'].dropna(),
        mode='markers', name='Máximos Locales',
        marker=dict(color='red', size=8)
    ))

    # Mínimos locales
    fig.add_trace(go.Scatter(
        x=data['time'][~data['local_min'].isna()], y=data['local_min'].dropna(),
        mode='markers', name='Mínimos Locales',
        marker=dict(color='green', size=8)
    ))

    # Configuración del layout con ajuste automático de ejes X e Y
    last_day = data['time'].max() - datetime.timedelta(days=1)
    fig.update_layout(
        title=f'Precios con Máximos y Mínimos Locales ({timeframe_name})',
        xaxis=dict(
            range=[last_day, data['time'].max()],  # Ajustar rango al último día
            rangeslider=dict(visible=True),  # Deslizador para eje X
            type="date"
        ),
        yaxis=dict(
            autorange=True,  # Permitir ajuste automático del rango Y
            fixedrange=False  # Hacer que el eje Y sea interactivo
        ),
        template='plotly_dark'
    )
    return fig
def ultimos_puntos(data):
    """
    Determina si el último punto es alcista, bajista o neutro.
    """
    if data.empty or data['local_max'].isna().all() or data['local_min'].isna().all():
        print("El último punto es indeterminado")
        return "Indeterminado"

    # Obtén los índices de los últimos valores máximo y mínimo
    max_time = data['time'][~data['local_max'].isna()].iloc[-1] if not data['local_max'].isna().all() else None
    min_time = data['time'][~data['local_min'].isna()].iloc[-1] if not data['local_min'].isna().all() else None

    if max_time and (not min_time or max_time > min_time):
        print("El último punto es bajista")
        return "Bajista"
    elif min_time and (not max_time or min_time > max_time):
        print("El último punto es alcista")
        return "Alcista"
    else:
        print("El último punto es neutro")
        return "Neutro"

# Clase Black-Scholes
class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        d1 = (math.log(self.current_price / self.strike) +
              (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / \
             (self.volatility * math.sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * math.sqrt(self.time_to_maturity)

        call_price = (self.current_price * norm.cdf(d1) -
                      self.strike * math.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2))
        put_price = (self.strike * math.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2) -
                     self.current_price * norm.cdf(-d1))
        return call_price, put_price



# Crear la aplicación Dash
app = Dash(__name__)
app.layout = html.Div([

    # Controles de entrada para Black-Scholes
    html.Div([
        html.H3("Calculadora de Opciones (Black-Scholes)"),
        html.Label("Precio Actual del Activo:"),
        dcc.Input(id='current-price', type='number', value=100.0, step=0.1),
        html.Label("Precio de Ejercicio:"),
        dcc.Input(id='strike-price', type='number', value=100.0, step=0.1),
        html.Label("Volatilidad (%):"),
        dcc.Input(id='volatility', type='number', value=20.0, step=0.1),
        html.Label("Tasa de Interés Libre de Riesgo (%):"),
        dcc.Input(id='interest-rate', type='number', value=5.0, step=0.1),
        html.Label("Tiempo hasta el Vencimiento (años):"),
        dcc.Input(id='time-to-maturity', type='number', value=1.0, step=0.1),
        html.Button('Calcular Opciones', id='calculate-options', n_clicks=0)
    ], style={'margin-bottom': '20px'}),
    # Resultados de las opciones
    html.Div(id='options-output', style={'margin-top': '20px', 'text-align': 'center'}),



    # Controles para ajustar el eje Y
    html.Div([
        html.H3("Ajustar eje Y:", style={'text-align': 'center'}),
        html.Label("Mínimo del eje Y:", style={'display': 'block', 'text-align': 'center'}),
        dcc.Input(id='y-axis-min', type='number', value=None, placeholder="Ingrese valor mínimo",
                  style={'margin': '0 auto', 'display': 'block'}),
        html.Label("Máximo del eje Y:", style={'display': 'block', 'text-align': 'center'}),
        dcc.Input(id='y-axis-max', type='number', value=None, placeholder="Ingrese valor máximo",
                  style={'margin': '0 auto', 'display': 'block'}),
        html.Button('Actualizar', id='update-button', style={'margin': '0 auto', 'display': 'block'}),
    ], style={'margin-bottom': '20px', 'text-align': 'center'}),
    
    # Fila de gráficos
    html.Div([
        html.Div([
            dcc.Graph(id='live-graph-D1', style={'height': '500px'}),
            #html.H4("Gráfico D1", style={'text-align': 'center'})
            
        ], style={
            'flex': '1',
            'margin': '5 px',
            'background-color': '#f5f5f5',
            'padding': '20px',
            'text-align': 'center',
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
        }),
         #html.Div(id='tendencia-D1', style={'text-align': 'center', 'margin-top': '10px'})
        html.Div([
            dcc.Graph(id='live-graph-H1', style={'height': '500px'}),
            #html.H4("Gráfico H1", style={'text-align': 'center'})
        ], style={
            'flex': '1',
            'margin': '1px',
            'background-color': '#f5f5f5',
            'padding': '20px',
            'text-align': 'center',
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
        }),

        html.Div([
            dcc.Graph(id='live-graph-M5', style={'height': '500px'}),
            #html.H4("Gráfico M5", style={'text-align': 'center'})
        ], style={
            'flex': '1',
            'margin': '1px',
            'background-color': '#f5f5f5',
            'padding': '20px',
            'text-align': 'center',
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
        })
    ], style={
        'display': 'flex',
        'justify-content': 'space-evenly',
        'align-items': 'center'
    }),

    # Fila de tendencias
    html.Div([
        html.Div(id='tendencia-D1', style={
            'flex': '1',
            'margin': '10px',
            'background-color': '#e8e8e8',
            'padding': '10px',
            'text-align': 'center',
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
        }),
        html.Div(id='tendencia-H1', style={
            'flex': '1',
            'margin': '10px',
            'background-color': '#e8e8e8',
            'padding': '10px',
            'text-align': 'center',
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
        }),
        html.Div(id='tendencia-M5', style={
            'flex': '1',
            'margin': '10px',
            'background-color': '#e8e8e8',
            'padding': '10px',
            'text-align': 'center',
            'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
        })
    ], style={
        'display': 'flex',
        'justify-content': 'space-evenly',
        'align-items': 'center'
    }),

    dcc.Interval(id='update-interval', interval=90*1000, n_intervals=0)  # Actualización cada 1.5 minutos
])

# Callback para actualizar las gráficas
# Callback para actualizar las gráficas y las tendencias
@app.callback(
    [
        Output('options-output', 'children'),
        Output('live-graph-H1', 'figure'),
        Output('live-graph-D1', 'figure'),
        Output('live-graph-M5', 'figure'),
        Output('tendencia-D1', 'children'),
        Output('tendencia-H1', 'children'),
        Output('tendencia-M5', 'children')
    ],
    [
        Input('calculate-options', 'n_clicks'),
        Input('update-interval', 'n_intervals'),
        Input('y-axis-min', 'value'),
        Input('y-axis-max', 'value')
    ],
    [
        Input('current-price', 'value'),
        Input('strike-price', 'value'),
        Input('volatility', 'value'),
        Input('interest-rate', 'value'),
        Input('time-to-maturity', 'value')
    ]
)
def update_graphs_and_trends(n_clicks, n_intervals, y_min, y_max, current_price, strike, volatility, interest_rate, time_to_maturity):
    # Inicializamos las variables para los resultados
    options_output = "Sin cálculos de opciones"
    figures = []
    trends = []
    
    # 1. Calcular precios de opciones si el botón fue presionado
    if n_clicks > 0:
        volatility /= 100  # Convertir a porcentaje
        interest_rate /= 100
        bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
        call_price, put_price = bs.calculate_prices()
        options_output = html.Div([
            html.H4(f"Precio CALL: ${call_price:.2f}"),
            html.H4(f"Precio PUT: ${put_price:.2f}")
        ])

    # 2. Actualizar gráficas y tendencias
    for timeframe_name, timeframe in timeframes.items():
        data = fetch_and_update_data(timeframe_name, timeframe)
        if not data.empty:
            # Generar la figura con ajuste dinámico del eje Y
            figure = generate_figure_with_y_adjustment(data, timeframe_name, y_min, y_max)
            trend = ultimos_puntos(data)
        else:
            figure = go.Figure()  # Figura vacía si no hay datos
            trend = "Sin datos"
        
        figures.append(figure)
        trends.append(f"Tendencia: {trend}")

    # 3. Devolver resultados
    return (
        options_output,
        figures[1],  # Gráfico H1
        figures[0],  # Gráfico D1
        figures[2],  # Gráfico M5
        trends[0],   # Tendencia D1
        trends[1],   # Tendencia H1
        trends[2]    # Tendencia M5
    )
# Función para generar una gráfica con ajuste dinámico del eje Y
def generate_figure_with_y_adjustment(data, timeframe_name, y_min, y_max):
    fig = generate_figure(data, timeframe_name)  # Reutiliza la función original
    # Ajustar el eje Y si se proporcionan valores
    fig.update_layout(
        yaxis=dict(
            range=[y_min, y_max] if y_min is not None and y_max is not None else None,
            autorange=True if y_min is None or y_max is None else False
        )
    )
    return fig
def calculate_options(n_clicks, current_price, strike, volatility, interest_rate, time_to_maturity):
    if n_clicks > 0:
        # Convertir volatilidad y tasa de interés a proporciones
        volatility /= 100
        interest_rate /= 100

        # Crear objeto Black-Scholes
        bs = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
        call_price, put_price = bs.calculate_prices()

        return html.Div([
            html.H4(f"Precio CALL: ${call_price:.2f}"),
            html.H4(f"Precio PUT: ${put_price:.2f}")
        ])
    return "Ingresa los parámetros y haz clic en 'Calcular Opciones'"



# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=False)
