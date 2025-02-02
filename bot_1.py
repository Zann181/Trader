import logging
import MetaTrader5 as mt5
import pandas as pd
import datetime
import numpy as np
from scipy.signal import argrelextrema
from dash import Dash, dcc, html
import plotly.graph_objects as go
from statsmodels.tsa.api import SimpleExpSmoothing

# Configuración global
logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) %(levelname)s %(message)s",
    datefmt="%m/%d/%y - %H:%M:%S %Z"
)
logger = logging.getLogger("market_bot")

COLORS = {
    'background': '#000000',
    'text': '#FFFFFF',
    'card': '#1e1e1e',
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

class MetaTraderConnector:
    @staticmethod
    def initialize():
        if not mt5.initialize():
            logger.error("Error al inicializar MetaTrader 5")
            mt5.shutdown()
            exit()

    @staticmethod
    def fetch_data(symbol, timeframe, days=160):
        utc_to = datetime.datetime.now()
        utc_from = utc_to - datetime.timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        return data.drop_duplicates('time')

class DataProcessor:
    @staticmethod
    def process_data(data):
        if data.empty or len(data) < 15:
            data['smoothed'] = np.nan
            data['maxima'] = np.nan
            data['minima'] = np.nan
            return data

        model = SimpleExpSmoothing(data['close']).fit(smoothing_level=0.2, optimized=False)
        data['smoothed'] = model.fittedvalues

        data['ma_50'] = data['close'].rolling(50).mean()
        data['ma_200'] = data['close'].rolling(200).mean()

        maxima = argrelextrema(data['smoothed'].values, np.greater_equal, order=5)[0]
        minima = argrelextrema(data['smoothed'].values, np.less_equal, order=5)[0]

        data['maxima'] = np.nan
        data['minima'] = np.nan
        data.iloc[maxima, data.columns.get_loc('maxima')] = data['smoothed'].iloc[maxima]
        data.iloc[minima, data.columns.get_loc('minima')] = data['smoothed'].iloc[minima]

        return data

class MarketAnalyzer:
    def __init__(self, data):
        self.data = data
        self.levels = []
        self.trend = None

    def analyze(self):
        self._determine_trend()
        self._identify_levels()

    def _determine_trend(self):
        if len(self.data) < 200:
            self.trend = "Indeterminado"
            return

        ema50 = self.data['ma_50']
        ema200 = self.data['ma_200']

        self.trend = "Alcista" if ema50.iloc[-1] > ema200.iloc[-1] else "Bajista"

    def _identify_levels(self):
        minima = self.data[self.data['minima'].notna()]
        maxima = self.data[self.data['maxima'].notna()]

        if not minima.empty:
            self.levels.append(('Soporte', minima['minima'].min(), '#006400'))

        if not maxima.empty:
            self.levels.append(('Resistencia', maxima['maxima'].max(), '#FF0000'))

class TradingVisualizer:
    @staticmethod
    def create_figure(data, levels, trend):
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=data['time'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Precio'
        ))

        for level in levels:
            fig.add_shape(type='line',
                          x0=data['time'].iloc[0],
                          x1=data['time'].iloc[-1],
                          y0=level[1],
                          y1=level[1],
                          line=dict(color=level[2], width=2, dash='dot'),
                          name=level[0])

        fig.update_layout(
            title=f'Análisis de Mercado - Tendencia {trend}',
            xaxis_title='Fecha',
            yaxis_title='Precio',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            font_color=COLORS['text'],
            xaxis_rangeslider_visible=False
        )

        return fig

class TradingApp:
    def __init__(self):
        self.app = Dash(__name__)
        self._setup_layout()

    def _setup_layout(self):
        self.app.layout = html.Div(style={'backgroundColor': COLORS['background']}, children=[
            html.H1("Sistema de Trading Avanzado", style={'color': COLORS['text'], 'textAlign': 'center'}),
            *[
                html.Div([
                    html.H2(f"{symbol}", style={'color': COLORS['text'], 'marginTop': '20px'}),
                    html.Div([
                        dcc.Graph(figure=TradingVisualizer.create_figure(
                            DataProcessor.process_data(
                                MetaTraderConnector.fetch_data(symbol, TIMEFRAMES[tf])
                            ),
                            [],
                            "Indeterminado"
                        )) for tf in TIMEFRAMES
                    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-around'})
                ]) for symbol in SYMBOLS
            ]
        ])

    def run(self):
        self.app.run_server(debug=True)

if __name__ == "__main__":
    MetaTraderConnector.initialize()
    app = TradingApp()
    app.run()
    mt5.shutdown()
