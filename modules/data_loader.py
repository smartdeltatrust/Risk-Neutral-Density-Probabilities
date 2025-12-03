import yfinance as yf
import pandas as pd
import io
from urllib.parse import urlencode
from typing import Literal
import io
from typing import Literal
from assets.config.settings import settings
import time
import re
import requests
from assets.config.settings import settings



def obtener_info_ticker(ticker: str) -> dict:
    """
    Descarga y retorna información detallada del ticker usando yfinance.
    Parámetros:
        ticker (str): Símbolo del ticker (ej. 'AAPL', 'MSFT', 'GOOGL')
    Retorna:
        dict: Diccionario con la siguiente información:
            - info_general: Información básica del activo
            - historial: DataFrame con precios históricos
            - dividendos: Serie de dividendos
            - splits: Serie de splits
            - calendario: Fechas clave (earnings, etc.)
            - recomendaciones: DataFrame con recomendaciones de analistas
    """
    try:
        accion = yf.Ticker(ticker)

        return {
            "info_general": accion.info,  # puede estar vacío en tickers no válidos
            "historial": accion.history(period="max"),
            "dividendos": accion.dividends,
            "splits": accion.splits,
            "calendario": accion.calendar,
            "recomendaciones": accion.recommendations
        }
    except Exception as e:
        print(f"Error al obtener datos del ticker '{ticker}': {e}")
        return {}


def _get_with_retries(
    url: str,
    n_retries: int = 3,
    timeout: int = 20,  # subimos de 10s a 20s
) -> requests.Response:
    """
    Llama a requests.get con reintentos básicos en caso de Timeout.
    Lanza la excepción si tras n_retries sigue fallando.
    """
    last_err = None
    for attempt in range(n_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout as e:
            last_err = e
            if attempt < n_retries - 1:
                time.sleep(1.0)  # pequeño backoff
            else:
                raise
        except requests.exceptions.RequestException as e:
            # Para otros errores (DNS, 403, etc.) no tiene sentido reintentar mucho
            raise

    # por si acaso
    if last_err is not None:
        raise last_err







#################### COmienza el Cálculo de las Densidades #################
# modules/data_loader.py



RangeCode = Literal["d1", "d5", "m1", "m3", "m6", "ytd", "y1", "y2", "y5", "max"]


def _read_csv_from_response(resp: requests.Response) -> pd.DataFrame:
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def fetch_quote_history(
    ticker: str,
    range_code: RangeCode | str | None = None,
    auth_token: str | None = None,
) -> pd.DataFrame:
    if range_code is None:
        range_code = settings.DEFAULT_RANGE

    if auth_token is None:
        auth_token = settings.FINVIZ_AUTH_TOKEN

    params = {
        "t": ticker,
        "p": "d",
        "r": range_code,
        "auth": auth_token,
    }

    url = f"{settings.FINVIZ_BASE_URL}/quote_export.ashx?" + urlencode(params)
    try:
        resp = _get_with_retries(url, n_retries=3, timeout=20)
    except requests.exceptions.Timeout as e:
        raise RuntimeError("Timeout al descargar histórico desde Finviz") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error al descargar histórico desde Finviz: {e}") from e
    
    df = _read_csv_from_response(resp)

    # ⬇️ Aquí el cambio importante: formato MM/DD/YYYY
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    df = df.sort_values("Date").reset_index(drop=True)
    return df




def fetch_available_expiries(
    ticker: str,
    auth_token: str | None = None,
) -> list[str]:
    """
    Devuelve la lista de fechas de vencimiento disponibles para un ticker,
    usando yfinance (solo para obtener la lista de vencimientos).

    Ignora auth_token a propósito: Finviz se sigue usando para precios,
    y yfinance solo nos da las fechas tipo '2025-12-19'.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return []

    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options or []
    except Exception as e:
        raise RuntimeError(f"No se pudieron obtener los vencimientos desde yfinance: {e}")

    # yfinance ya da strings tipo '2025-12-19'
    # los ordenamos por fecha
    try:
        expiries_sorted = sorted(expiries, key=pd.to_datetime)
    except Exception:
        expiries_sorted = sorted(expiries)

    return expiries_sorted




def fetch_options_chain(
    ticker: str,
    expiry: str,  # "YYYY-MM-DD"
    auth_token: str | None = None,
) -> pd.DataFrame:
    """
    Descarga la cadena de opciones para un ticker y vencimiento concreto.
    """
    if auth_token is None:
        auth_token = settings.FINVIZ_AUTH_TOKEN

    params = {
        "t": ticker,
        "ty": "oc",
        "e": expiry,
        "auth": auth_token,
    }
    url = f"{settings.FINVIZ_BASE_URL}/export/options?" + urlencode(params)
    try:
        resp = _get_with_retries(url, n_retries=3, timeout=20)
    except requests.exceptions.Timeout as e:
        raise RuntimeError("Timeout al descargar cadena de opciones desde Finviz") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error al descargar opciones desde Finviz: {e}") from e

    df = _read_csv_from_response(resp)
    return df


def clean_options_chain(
    df_raw: pd.DataFrame,
    min_open_interest: int = 50,
    max_rel_spread: float = 0.5,
) -> pd.DataFrame:
    """
    Limpia el dataframe de opciones de Finviz:
    - renombra columnas
    - convierte tipos numéricos
    - calcula mid_price y price usable
    - aplica filtro por OI y spread relativo
    """
    import numpy as np

    col_map = {
        "Contract Name": "contract",
        "Last Trade": "last_trade",
        "Strike": "strike",
        "Last Close": "last_close",
        "Bid": "bid",
        "Ask": "ask",
        "Change $": "change_dollar",
        "Change %": "change_pct",
        "Volume": "volume",
        "Open Int.": "open_interest",
        "Type": "option_type",
        "IV": "iv",
        "Delta": "delta",
        "Gamma": "gamma",
        "Theta": "theta",
        "Vega": "vega",
        "Rho": "rho",
    }

    df = df_raw.rename(columns=col_map).copy()

    df["option_type"] = df["option_type"].str.lower()
    df["last_trade"] = pd.to_datetime(df["last_trade"], errors="coerce")

    # limpiar porcentaje
    df["change_pct"] = (
        df["change_pct"].astype(str).str.replace("%", "", regex=False)
    )

    num_cols = [
        "strike", "last_close", "bid", "ask", "change_dollar",
        "change_pct", "volume", "open_interest", "iv",
        "delta", "gamma", "theta", "vega", "rho",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # mid price
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["price"] = np.where(
        df["mid_price"].notna() & (df["mid_price"] > 0),
        df["mid_price"],
        df["last_close"],
    )

    # filtros de liquidez
    df = df[df["open_interest"] >= min_open_interest]
    df = df[df["mid_price"] > 0]

    df["spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = df["spread"] / df["mid_price"]
    df = df[(df["rel_spread"] >= 0) & (df["rel_spread"] <= max_rel_spread)]

    df = df[df["price"].notna()]
    return df.reset_index(drop=True)
