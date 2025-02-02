import logging
import MetaTrader5 as mt5
import pandas as pd
import datetime
import numpy as np
import os
from scipy.signal import argrelextrema, savgol_filter
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Output, Input, State
import plotly.graph_objects as go

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) %(levelname)s %(message)s",
    datefmt="%m/%d/%y - %H:%M:%S %Z"
)
logger = logging.getLogger("market_bot")

# Inicializar MetaTrader 5
if not mt5.initialize():
    logger.error("Error al inicializar MetaTrader 5")
    mt5.shutdown()
    exit()

# Configuración de colores y símbolos
COLORS = {
    'background': '#1e1e1e',
    'text': '#ffffff',
    'card': '#2d2d2d',
    'positive': '#4CAF50',
    'negative': '#FF5252'
}

SYMBOLS = [
    "Volatility 100 Index",
    "Crash 300 Index",
    "Crash 500 Index",
    "Boom 300 Index",
    "Boom 500 Index"
]

TIMEFRAMES = {
    'D1': mt5.TIMEFRAME_D1,
    'H1': mt5.TIMEFRAME_H1,
    'M5': mt5.TIMEFRAME_M5
}

class DataFetcher:
    @staticmethod
    def fetch_data(symbol, timeframe):
        """
        Descarga datos históricos para un símbolo y timeframe específico.
        """
        utc_to = datetime.datetime.now()
        utc_from = utc_to - datetime.timedelta(days=160)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        
        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        return data.drop_duplicates('time')

class DataProcessor:
    @staticmethod
    def process_data(data):
        """
        Aplica un filtro de suavizado y determina máximos y mínimos locales.
        También calcula Bandas de Bollinger con EMA.
        """
        if data.empty or len(data) < 20:
            # Inicializamos columnas si no hay datos suficientes
            data['smoothed'] = np.nan
            data['maxima'] = np.nan
            data['minima'] = np.nan
            data['ema_20'] = np.nan  # <-- Cambio
            data['upper_band'] = np.nan
            data['lower_band'] = np.nan
            data['touch_upper'] = np.nan  # <-- Añadido
            data['touch_lower'] = np.nan  # <-- Añadido            
            return data
        
        # 1. Suavizado con Savitzky-Golay (o puedes usar directamente la EMA como filtro principal)
        data['smoothed'] = savgol_filter(
            data['close'],
            window_length=7,
            polyorder=2
        )

        # 2. EMA (ejemplo de 50 periodos) # <-- Cambio
        data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()

        # 3. Determinación de máximos y mínimos locales en la serie suavizada
        maxima = argrelextrema(data['smoothed'].values, np.greater_equal, order=7)[0]
        minima = argrelextrema(data['smoothed'].values, np.less_equal, order=7)[0]

        data['maxima'] = np.nan
        data['minima'] = np.nan
        data.iloc[maxima, data.columns.get_loc('maxima')] = data['smoothed'].iloc[maxima]
        data.iloc[minima, data.columns.get_loc('minima')] = data['smoothed'].iloc[minima]

        # 4. Bandas de Bollinger usando EMA de 20 periodos como línea central # <-- Cambio
        window = 5
        data['ema_20'] = data['smoothed'].ewm(span=window, adjust=False).mean()
        
        # Puedes usar también ewm para la desviación estándar, pero muchas veces se deja rolling:
        data['std_dev'] = data['smoothed'].rolling(window).std()
        
        data['upper_band'] = data['ema_20'] + 2 * data['std_dev']
        data['lower_band'] = data['ema_20'] - 2 * data['std_dev']
        
        return data

    @staticmethod
    def get_trend(data):
        """
        Determina si la tendencia más reciente es alcista, bajista o neutra,
        basándose en la posición relativa de los últimos máximos y mínimos.
        """
        if data.empty or data['maxima'].isna().all() or data['minima'].isna().all():
            return "Neutro"
        
        last_max = data['maxima'].last_valid_index()
        last_min = data['minima'].last_valid_index()
        
        if last_max and last_min:
            return "Alcista" if last_max < last_min else "Bajista"
        return "Neutro"

class TradingApp:
    def __init__(self):
        self.app = Dash(__name__)
        self.app.layout = self.create_layout()
        self.register_callbacks()

    def create_layout(self):
        return html.Div([
            html.H1("Sistema de Trading Automatizado", style={
                'textAlign': 'center',
                'color': COLORS['text'],
                'padding': '20px',
                'backgroundColor': COLORS['card']
            }),
            
            dash_table.DataTable(
                id='operaciones-table',
                columns=[
                    {'name': 'Símbolo', 'id': 'simbolo'},
                    {'name': 'Hora Apertura', 'id': 'hora_apertura'},
                    {'name': 'Hora Cierre', 'id': 'hora_cierre'},
                    {'name': 'Precio Cierre', 'id': 'valor_cierre'},
                    {'name': 'Beneficio', 'id': 'beneficio'},
                    {'name': 'Clasificación', 'id': 'clasificacion'},
                    {'name': 'Beneficio Total', 'id': 'total_beneficio'}
                ],
                style_table={'overflowX': 'auto', 'margin': '20px'},
                style_header={
                    'backgroundColor': COLORS['card'],
                    'color': COLORS['text'],
                    'fontWeight': 'bold'
                },
                style_cell={
                    'backgroundColor': COLORS['background'],
                    'color': COLORS['text'],
                    'minWidth': '120px'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'clasificacion', 'filter_query': '{clasificacion} eq "ganada"'},
                        'backgroundColor': COLORS['positive'],
                        'color': 'white'
                    },
                    {
                        'if': {'column_id': 'clasificacion', 'filter_query': '{clasificacion} eq "perdida"'},
                        'backgroundColor': COLORS['negative'],
                        'color': 'white'
                    }
                ]
            ),
            
            # Sección de gráficos
            html.Div([
                html.Div([
                    html.Div([
                        html.H2(symbol, style={
                            'color': COLORS['text'],
                            'textAlign': 'center',
                            'marginBottom': '20px'
                        }),
                        html.Div([
                            html.Div([
                                dcc.Graph(
                                    id=f'graph-{symbol}-{tf}',
                                    config={'displayModeBar': False},
                                    style={'height': '400px'}
                                ),
                                html.Div(
                                    id=f'trend-{symbol}-{tf}',
                                    style={
                                        'color': COLORS['text'],
                                        'padding': '10px',
                                        'fontSize': '16px',
                                        'textAlign': 'center'
                                    }
                                )
                            ], style={
                                'width': '32%',
                                'display': 'inline-block',
                                'verticalAlign': 'top',
                                'backgroundColor': COLORS['card'],
                                'margin': '5px',
                                'borderRadius': '10px'
                            }) for tf in TIMEFRAMES
                        ])
                    ], style={
                        'backgroundColor': COLORS['background'],
                        'padding': '20px',
                        'margin': '10px',
                        'borderRadius': '15px'
                    })
                ]) for symbol in SYMBOLS
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=90*1000,  # Actualiza cada 90s
                n_intervals=0
            ),
            
            dcc.ConfirmDialog(
                id='alert',
                message="",
                displayed=False
            )
        ], style={
            'backgroundColor': COLORS['background'],
            'padding': '20px'
        })

    def register_callbacks(self):
        @self.app.callback(
            Output('operaciones-table', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_table(n):
            """
            Cada vez que se activa el Interval, se lee el CSV de operaciones
            y se actualiza la tabla.
            """
            try:
                if os.path.exists('operaciones.csv'):
                    df = pd.read_csv(
                        'operaciones.csv',
                        low_memory=False,
                        parse_dates=['hora_apertura', 'hora_cierre'],
                        infer_datetime_format=True,
                        dayfirst=False
                    )
                    df['hora_apertura'] = df['hora_apertura'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    df['hora_cierre'] = df['hora_cierre'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    return df.to_dict('records')
                return []
            except Exception as e:
                logger.error(f"Error al cargar datos: {str(e)}")
                return []

        @self.app.callback(
            [Output(f'graph-{symbol}-{tf}', 'figure') for symbol in SYMBOLS for tf in TIMEFRAMES] +
            [Output(f'trend-{symbol}-{tf}', 'children') for symbol in SYMBOLS for tf in TIMEFRAMES] +
            [Output('alert', 'message'), Output('alert', 'displayed')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_all(n):
            """
            Actualiza todos los gráficos, la etiqueta de tendencia y el popup de alerta.
            """
            figures = []
            trends = []
            alert_messages = []

            for symbol in SYMBOLS:
                for tf_name, tf_value in TIMEFRAMES.items():
                    data = DataFetcher.fetch_data(symbol, tf_value)
                    processed_data = DataProcessor.process_data(data)

                    fig = go.Figure()

                    if not processed_data.empty:
                        # Precio principal
                        fig.add_trace(go.Scatter(
                            x=processed_data['time'],
                            y=processed_data['close'],
                            line=dict(color='#00b4d8', width=1),
                            name='Precio'
                        ))
                        
                        # Línea suavizada (Savitzky-Golay)
                        fig.add_trace(go.Scatter(
                            x=processed_data['time'],
                            y=processed_data['smoothed'],
                            line=dict(color='#ff758c', width=2),
                            name='Tendencia'
                        ))
                        
                        # Máximos y mínimos
                        fig.add_trace(go.Scatter(
                            x=processed_data['time'],
                            y=processed_data['maxima'],
                            mode='markers',
                            marker=dict(color='red', size=8),
                            name='Máximos'
                        ))
                        fig.add_trace(go.Scatter(
                            x=processed_data['time'],
                            y=processed_data['minima'],
                            mode='markers',
                            marker=dict(color='green', size=8),
                            name='Mínimos'
                        ))

                        # Bandas de Bollinger (EMA de 20)
                        fig.add_trace(go.Scatter(
                            x=processed_data['time'],
                            y=processed_data['upper_band'],
                            line=dict(color='gray', width=1),
                            name='Bollinger Superior'
                        ))
                        fig.add_trace(go.Scatter(
                            x=processed_data['time'],
                            y=processed_data['lower_band'],
                            line=dict(color='gray', width=1),
                            fill='tonexty',  # Relleno entre banda superior e inferior
                            fillcolor='rgba(128,128,128,0.2)',
                            name='Bollinger Inferior'
                        ))



                        

                        # Determinar tendencia
                        trend = DataProcessor.get_trend(processed_data)
                        trend_color = '#70e000' if trend == 'Alcista' else '#ff006e' if trend == 'Bajista' else '#cccccc'
                        trend_text = f"{tf_name}: {trend}"

                        if trend in ("Alcista", "Bajista"):
                            alert_messages.append(f"Alerta en {symbol} ({tf_name}): {trend}")
                    else:
                        trend_text = "Sin datos"
                        trend_color = COLORS['text']
                        trend = "Neutro"

                    fig.update_layout(
                        plot_bgcolor=COLORS['card'],
                        paper_bgcolor=COLORS['background'],
                        font=dict(color=COLORS['text']),
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis=dict(gridcolor='#444'),
                        yaxis=dict(gridcolor='#444'),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    figures.append(fig)
                    trends.append(html.Span(trend_text, style={'color': trend_color}))

            # Construir el mensaje de alerta si hay tendencias marcadas
            alert_message = "\n".join(alert_messages) if alert_messages else ""
            show_alert = bool(alert_messages)

            return figures + trends + [alert_message, show_alert]

    def run(self):
        self.app.run_server(debug=False, port=8050)

if __name__ == '__main__':
    app = TradingApp()
    app.run()
