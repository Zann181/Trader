import MetaTrader5 as mt5
import pandas as pd
import datetime
import numpy as np
from scipy.signal import argrelextrema, savgol_filter
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objects as go

# Inicializar MetaTrader 5
if not mt5.initialize():
    print("Error al inicializar MetaTrader 5")
    mt5.shutdown()
    exit()

# Configuración optimizada
SYMBOLS = [
    "Volatility 100 Index",
    "Crash 300 Index",
    "Crash 500 Index",
    "Boom 300 Index",
    "Boom 500 Index"
]

TIMEFRAMES = {
    'D1': mt5.TIMEFRAME_H4,
    'H1': mt5.TIMEFRAME_H1,
    'M5': mt5.TIMEFRAME_M5
}

COLORS = {
    'background': '#1e1e1e',
    'text': '#ffffff',
    'card': '#2d2d2d'
}

# Función optimizada para obtener datos
def fetch_data(symbol, timeframe):
    utc_to = datetime.datetime.now()
    utc_from = utc_to - datetime.timedelta(days=60)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data.drop_duplicates('time')

# Función para procesar datos y detectar extremos
def process_data(data):
    if data.empty or len(data) < 15:
        data['smoothed'] = np.nan
        data['maxima'] = np.nan
        data['minima'] = np.nan
        return data
    
    data['smoothed'] = savgol_filter(data['close'], window_length=min(15, len(data)), polyorder=3)
    data['ma_50'] = data['close'].rolling(50).mean()
    
    maxima = argrelextrema(data['smoothed'].values, np.greater_equal, order=5)[0]
    minima = argrelextrema(data['smoothed'].values, np.less_equal, order=5)[0]
    
    data['maxima'] = np.nan
    data['minima'] = np.nan
    data.iloc[maxima, data.columns.get_loc('maxima')] = data['smoothed'].iloc[maxima]
    data.iloc[minima, data.columns.get_loc('minima')] = data['smoothed'].iloc[minima]
    
    return data

# Función para determinar la tendencia
def get_trend(data):
    if data.empty or data['maxima'].isna().all() or data['minima'].isna().all():
        return "Neutro"
    
    last_max = data['maxima'].last_valid_index()
    last_min = data['minima'].last_valid_index()
    
    if last_max and last_min:
        return "Alcista" if last_max > last_min else "Bajista"
    return "Neutro"

# Crear aplicación Dash optimizada
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Análisis de Mercado en Tiempo Real", style={
        'textAlign': 'center',
        'color': COLORS['text'],
        'padding': '20px'
    }),
    
    *[html.Div([
        html.H2(symbol, style={
            'color': COLORS['text'],
            'margin': '20px',
            'textAlign': 'center'
        }),
        html.Div([
            *[html.Div([
                dcc.Graph(id=f'graph-{symbol}-{tf}', config={'displayModeBar': False}),
                html.Div(id=f'trend-{symbol}-{tf}', style={
                    'color': COLORS['text'],
                    'padding': '10px',
                    'fontSize': '18px',
                    'textAlign': 'center'
                })
            ], style={
                'width': '32%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'backgroundColor': COLORS['card'],
                'margin': '5px',
                'borderRadius': '10px'
            }) for tf in TIMEFRAMES]
        ])
    ]) for symbol in SYMBOLS],
    
    dcc.Interval(
        id='interval-component',
        interval=90*1000,
        n_intervals=0
    )
], style={
    'backgroundColor': COLORS['background'],
    'padding': '20px'
})

# Callback optimizado para actualizar todas las gráficas
@app.callback(
    [Output(f'graph-{symbol}-{tf}', 'figure') for symbol in SYMBOLS for tf in TIMEFRAMES] +
    [Output(f'trend-{symbol}-{tf}', 'children') for symbol in SYMBOLS for tf in TIMEFRAMES],
    [Input('interval-component', 'n_intervals')]
)
def update_all_graphs(n):
    figures = []
    trends = []
    
    for symbol in SYMBOLS:
        for tf_name, tf_value in TIMEFRAMES.items():
            data = fetch_data(symbol, tf_value)
            processed_data = process_data(data)
            
            fig = go.Figure()
            
            if 'smoothed' in processed_data.columns and not processed_data.empty:
                fig.add_trace(go.Scatter(x=processed_data['time'], y=processed_data['close'],
                                         line=dict(color='#00b4d8', width=1), name='Precio Real'))
                fig.add_trace(go.Scatter(x=processed_data['time'], y=processed_data['smoothed'],
                                         line=dict(color='#ff758c', width=2), name='Tendencia'))
                
                trend = get_trend(processed_data)
                trend_color = '#70e000' if trend == 'Alcista' else '#ff006e' if trend == 'Bajista' else '#cccccc'
                trend_text = f"Tendencia {tf_name}: {trend}"
            else:
                fig = go.Figure()
                trend_text = "Sin datos disponibles"
                trend_color = COLORS['text']
            
            figures.append(fig)
            trends.append(html.Span(trend_text, style={'color': trend_color}))
    
    return figures + trends

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_ui=False)
