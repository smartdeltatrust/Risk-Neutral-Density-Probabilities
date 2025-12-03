import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()
API_KEY_FINVIZ = os.getenv("FINVIZ_KEY")

# Carga variables del archivo .env
load_dotenv()


@dataclass
class Settings:
    # Base de Finviz
    FINVIZ_BASE_URL: str = "https://elite.finviz.com"

    # Token de Finviz (defínelo en .env como FINVIZ_AUTH_TOKEN=...)
    FINVIZ_AUTH_TOKEN: str = os.getenv("FINVIZ_AUTH_TOKEN", "")

    # Parámetros por defecto
    DEFAULT_RANGE: str = "ytd"   # d1, d5, m1, m3, m6, ytd, y1, y2, y5, max
    DEFAULT_RATE: float = 0.04   # tasa libre de riesgo anual aprox.
    HIST_SIGMA_REL: float = 0.01 # 1% del precio como anchura de gaussiana histórica

    # Grid numérico
    N_STRIKE_POINTS: int = 200
    N_PRICE_POINTS: int = 300
    PRICE_PADDING: float = 0.10  # 10% de margen alrededor del rango de precios


settings = Settings()
