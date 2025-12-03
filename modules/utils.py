# modules/utils.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from assets.config.settings import settings


# -------------------------------------------------
# 1. Utilidades básicas
# -------------------------------------------------
def gaussian_density(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    coeff = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    return coeff * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# -------------------------------------------------
# 2. Construcción de ejes y matriz de densidad
# -------------------------------------------------
def build_price_axis(
    quotes_df: pd.DataFrame,
    rnd_list: list[Tuple[np.ndarray, np.ndarray]],
    price_padding: float | None = None,
    n_price_points: int | None = None,
) -> np.ndarray:
    """
    Construye un eje de precios común a partir del histórico (Close)
    y de los grids de precios de las distintas RNDs.

    En lugar de usar [min(K), max(K)] de la RND, usamos aproximadamente
    el rango entre los cuantiles 1% y 99% para evitar colas extremas con
    probabilidad casi nula.
    """
    if price_padding is None:
        price_padding = settings.PRICE_PADDING
    if n_price_points is None:
        n_price_points = settings.N_PRICE_POINTS

    prices_hist = quotes_df["Close"].values
    all_prices_list = [prices_hist]

    for K_grid, pdf in rnd_list:
        # Si no tenemos pdf o tiene tamaño raro, usamos todo K_grid
        if pdf is None or len(pdf) != len(K_grid):
            all_prices_list.append(K_grid)
            continue

        pdf = np.clip(pdf, 0, None)
        if pdf.sum() <= 0:
            all_prices_list.append(K_grid)
            continue

        # CDF para calcular cuantiles aproximados
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]

        try:
            low = np.interp(0.01, cdf, K_grid)
            high = np.interp(0.99, cdf, K_grid)
        except Exception:
            # fallback: todo el soporte si algo sale mal
            low, high = K_grid.min(), K_grid.max()

        all_prices_list.append(np.array([low, high]))

    all_prices = np.concatenate(all_prices_list)
    p_min, p_max = np.nanmin(all_prices), np.nanmax(all_prices)
    span = p_max - p_min
    p_min -= price_padding * span
    p_max += price_padding * span

    return np.linspace(p_min, p_max, n_price_points)



def build_time_price_density(
    quotes_df: pd.DataFrame,
    rnd_by_date: Dict[pd.Timestamp, Tuple[np.ndarray, np.ndarray]],
    hist_sigma_rel: float | None = None,
    interpolate_future: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye la matriz de densidad conjunta (precio x tiempo):

    - quotes_df: histórico con columnas ['Date', 'Close']
    - rnd_by_date: dict {expiry_date: (K_grid, pdf_K)} para una o varias fechas

    Si interpolate_future=True, interpola densidades entre vencimientos para
    rellenar todas las fechas futuras.

    NUEVO:
    - Entre la última fecha histórica (last_hist) y el primer vencimiento futuro,
      la densidad pasa de una gaussiana estrecha (histórica) a la RND completa
      del vencimiento de forma progresiva.
    """
    if hist_sigma_rel is None:
        hist_sigma_rel = settings.HIST_SIGMA_REL

    rnd_list = list(rnd_by_date.values())
    price_grid = build_price_axis(quotes_df, rnd_list)

    # Fechas históricas y última fecha disponible
    dates_hist = quotes_df["Date"].values
    last_hist = quotes_df["Date"].max()

    # Sólo expiries >= última fecha histórica
    future_expiries = sorted(
        [pd.Timestamp(d) for d in rnd_by_date.keys() if pd.Timestamp(d) >= last_hist]
    )

    # Eje completo: histórico + futuro hasta el último vencimiento
    if future_expiries:
        dates_future = pd.date_range(
            start=last_hist,
            end=future_expiries[-1],
            freq="B",  # business days
        )
    else:
        dates_future = pd.to_datetime([])

    dates_all = np.unique(
        np.concatenate(
            [
                dates_hist.astype("datetime64[ns]"),
                dates_future.values.astype("datetime64[ns]"),
            ]
        )
    )

    n_p = len(price_grid)
    n_t = len(dates_all)
    density = np.zeros((n_p, n_t), dtype=float)

    # -------------------------------------------------
    # 1) Histórico: gausianas estrechas alrededor del Close
    # -------------------------------------------------
    close_by_date = quotes_df.set_index("Date")["Close"].to_dict()

    for j, d in enumerate(dates_all):
        d_ts = pd.Timestamp(d)
        if d_ts in close_by_date:
            close = close_by_date[d_ts]
            sigma = max(close * hist_sigma_rel, 1e-3)
            pdf = gaussian_density(price_grid, close, sigma)
            pdf = np.clip(pdf, 0, None)
            area = np.trapz(pdf, price_grid)
            if area > 0:
                pdf /= area
            density[:, j] = pdf

    # Si no hay expiries futuros, devolvemos solo histórico
    if not future_expiries:
        return dates_all, price_grid, density

    # -------------------------------------------------
    # 2) Precomputar pdfs de expiries en price_grid
    # -------------------------------------------------
    expiry_pdfs: Dict[pd.Timestamp, np.ndarray] = {}
    for d_exp, (K_grid, pdf_K) in rnd_by_date.items():
        d_exp = pd.Timestamp(d_exp)
        pdf_interp = np.interp(price_grid, K_grid, pdf_K, left=0.0, right=0.0)
        pdf_interp = np.clip(pdf_interp, 0, None)
        area = np.trapz(pdf_interp, price_grid)
        if area > 0:
            pdf_interp /= area
        expiry_pdfs[d_exp] = pdf_interp

    # -------------------------------------------------
    # 2b) Perfil estrecho en la última fecha histórica
    #     (punto de partida del cono futuro)
    # -------------------------------------------------
    S_last = close_by_date[last_hist]
    sigma_last = max(S_last * hist_sigma_rel, 1e-3)

    base_pdf_last = gaussian_density(price_grid, S_last, sigma_last)
    base_pdf_last = np.clip(base_pdf_last, 0, None)
    area_base = np.trapz(base_pdf_last, price_grid)
    if area_base > 0:
        base_pdf_last /= area_base

    # Primer vencimiento futuro = “fin del cono”
    first_expiry = future_expiries[0]
    total_days_first = max((first_expiry - last_hist).days, 1)

    # -------------------------------------------------
    # 3) Densidades futuras (interpoladas en el tiempo)
    # -------------------------------------------------
    for j, d in enumerate(dates_all):
        d_ts = pd.Timestamp(d)
        if d_ts in close_by_date:
            continue  # ya tiene histórico

        # --- NUEVO: entre last_hist y el primer vencimiento ---
        # Mezcla progresiva: base_pdf_last -> RND del primer vencimiento
        if last_hist < d_ts <= first_expiry:
            t_days = max((d_ts - last_hist).days, 0)
            w = t_days / total_days_first  # 0 cerca de hoy, 1 en el vencimiento

            pdf_expiry = expiry_pdfs[first_expiry]
            pdf_mix = (1.0 - w) * base_pdf_last + w * pdf_expiry

            pdf_mix = np.clip(pdf_mix, 0, None)
            area_mix = np.trapz(pdf_mix, price_grid)
            if area_mix > 0:
                pdf_mix /= area_mix

            density[:, j] = pdf_mix
            continue

        # Por seguridad: si hubiera fechas < primer expiry pero también < last_hist
        if d_ts < future_expiries[0]:
            density[:, j] = expiry_pdfs[future_expiries[0]]
            continue

        # Después del último vencimiento: se mantiene la última densidad
        if d_ts > future_expiries[-1]:
            density[:, j] = expiry_pdfs[future_expiries[-1]]
            continue

        # Entre vencimientos futuros: interpolación lineal en el tiempo (como antes)
        for k in range(len(future_expiries) - 1):
            T_left, T_right = future_expiries[k], future_expiries[k + 1]
            if T_left <= d_ts <= T_right:
                if not interpolate_future or T_left == T_right:
                    density[:, j] = expiry_pdfs[T_left]
                else:
                    w = (d_ts - T_left).days / max((T_right - T_left).days, 1)
                    pdf_left = expiry_pdfs[T_left]
                    pdf_right = expiry_pdfs[T_right]
                    pdf_interp_time = (1 - w) * pdf_left + w * pdf_right
                    pdf_interp_time = np.clip(pdf_interp_time, 0, None)
                    area2 = np.trapz(pdf_interp_time, price_grid)
                    if area2 > 0:
                        pdf_interp_time /= area2
                    density[:, j] = pdf_interp_time
                break

    return dates_all, price_grid, density



# -------------------------------------------------
# 3. Limpieza CALL/PUT (paridad) y banda histórica
# -------------------------------------------------
def build_clean_calls_from_chain(
    options_df: pd.DataFrame,
    S0: float,
    valuation_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    r_annual: float,
    q_annual: float,
) -> pd.DataFrame:
    """
    Construye un DataFrame de 'calls limpios' combinando CALL y PUT
    usando paridad put-call.

    Devuelve columnas:
    ['strike', 'call_price_clean']
    """
    T_years = (expiry_date - valuation_date).days / 365.0
    T_years = max(T_years, 1e-6)

    df = options_df.copy()
    calls = df[df["option_type"] == "call"].copy()
    puts = df[df["option_type"] == "put"].copy()

    calls = calls.set_index("strike")
    puts = puts.set_index("strike")

    all_strikes = sorted(set(calls.index) | set(puts.index))
    rows = []

    disc_r = np.exp(-r_annual * T_years)
    disc_q = np.exp(-q_annual * T_years)

    for K in all_strikes:
        C_direct = calls.loc[K, "price"] if K in calls.index else np.nan
        P_direct = puts.loc[K, "price"] if K in puts.index else np.nan

        # Paridad: C_parity = P + S0 e^{-qT} - K e^{-rT}
        C_parity = np.nan
        if np.isfinite(P_direct):
            C_parity = P_direct + S0 * disc_q - K * disc_r

        candidates = []
        if np.isfinite(C_direct) and C_direct > 0:
            candidates.append(C_direct)
        if np.isfinite(C_parity) and C_parity > 0:
            candidates.append(C_parity)

        if not candidates:
            continue

        C_clean = float(np.mean(candidates))
        rows.append({"strike": K, "call_price_clean": C_clean})

    clean_calls = pd.DataFrame(rows)
    clean_calls = clean_calls.sort_values("strike").reset_index(drop=True)
    return clean_calls


def compute_realized_conf_band(
    quotes_df: pd.DataFrame,
    horizon_days: int,
) -> Tuple[float, float, float, float] | None:
    """
    Calcula bandas 95% y 68% históricas para un horizonte en días.
    Devuelve: (low_2p5, low_16, high_84, high_97p5) o None si no hay datos suficientes.
    """
    closes = quotes_df.set_index("Date")["Close"].sort_index()
    S = closes.values

    if len(S) <= horizon_days:
        return None

    r_H = []
    for i in range(len(S) - horizon_days):
        r = np.log(S[i + horizon_days] / S[i])
        r_H.append(r)

    r_H = np.array(r_H)
    if len(r_H) == 0:
        return None

    S0 = S[-1]
    future_prices = S0 * np.exp(r_H)

    low_2p5 = float(np.quantile(future_prices, 0.025))
    low_16 = float(np.quantile(future_prices, 0.16))
    high_84 = float(np.quantile(future_prices, 0.84))
    high_97p5 = float(np.quantile(future_prices, 0.975))

    return low_2p5, low_16, high_84, high_97p5


# -------------------------------------------------
# 4. Construcción de RND desde la cadena de opciones
# -------------------------------------------------
def _col(df: pd.DataFrame, *names):
    """Devuelve el primer nombre de columna que exista en el DataFrame."""
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Ninguna de las columnas {names} existe en el DataFrame.")


def compute_rnd_from_calls(
    options_df: pd.DataFrame,
    spot: float,
    valuation_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    r_annual: float,
    q_annual: float = 0.0,
    oi_min: int = 50,
    moneyness_low: float = 0.5,
    moneyness_high: float = 1.6,
    n_grid: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye la RND a partir de la cadena de opciones de FINVIZ (calls),
    con:
      - limpieza básica de datos
      - interpolación PCHIP
      - segunda derivada numérica
      - ajuste para que E_Q[S_T] = S0 * exp((r-q)*T)

    Devuelve:
      K_grid (soporte de precios) y pdf_K (densidad en ese soporte).
    """
    df = options_df.copy()

    # --- columnas ---
    strike_col = _col(df, "strike", "Strike")
    type_col = _col(df, "option_type", "type", "Type")
    bid_col = _col(df, "bid", "Bid")
    ask_col = _col(df, "ask", "Ask")
    last_col = _col(df, "last_close", "Last Close", "last", "Last")
    oi_col = _col(df, "open_int", "Open Int.", "Open Int", "open_interest")

    # Solo CALLS
    calls = df[df[type_col].str.lower() == "call"].copy()

    # Filtro por OI mínimo
    calls = calls[calls[oi_col] >= oi_min]

    # Mid-price razonable: (bid+ask)/2, luego bid, luego ask, luego last
    bid = calls[bid_col].astype(float)
    ask = calls[ask_col].astype(float)
    last = calls[last_col].astype(float)

    mid = np.where(
        (bid > 0) & (ask > 0),
        0.5 * (bid + ask),
        np.where(
            bid > 0,
            bid,
            np.where(
                ask > 0,
                ask,
                np.where(last > 0, last, np.nan),
            ),
        ),
    )

    calls["mid"] = mid
    calls = calls[np.isfinite(calls["mid"]) & (calls["mid"] > 0)]

    if calls.empty:
        raise ValueError("No hay calls líquidas con precios válidos para construir la RND.")

    # Rango de strikes alrededor del spot
    K = calls[strike_col].astype(float).values
    C = calls["mid"].values

    lo = moneyness_low * spot
    hi = moneyness_high * spot
    mask = (K >= lo) & (K <= hi)
    K = K[mask]
    C = C[mask]

    if K.size < 5:
        raise ValueError("No hay suficientes strikes en el rango alrededor del spot.")

    # Ordenar por strike
    order = np.argsort(K)
    K = K[order]
    C = C[order]

    # Eliminar strikes duplicados (promediar precios)
    uniq_K, idx_inv = np.unique(K, return_inverse=True)
    uniq_C = np.zeros_like(uniq_K, dtype=float)
    counts = np.zeros_like(uniq_K, dtype=int)
    for i, k_idx in enumerate(idx_inv):
        uniq_C[k_idx] += C[i]
        counts[k_idx] += 1
    uniq_C = uniq_C / counts

    K = uniq_K
    C = uniq_C

    # Tiempo a vencimiento en años
    T_days = (pd.Timestamp(expiry_date) - pd.Timestamp(valuation_date)).days
    T = max(T_days / 365.25, 1e-6)

    # Quitamos descuento: C_tilde(K) = C0 * exp(rT) = E[(S_T - K)+]
    C_tilde = C * np.exp(r_annual * T)

    # Interpolación PCHIP (monótona, menos oscilaciones que spline cúbico)
    interp = PchipInterpolator(K, C_tilde, extrapolate=False)

    K_grid = np.linspace(K.min(), K.max(), n_grid)
    C_grid = interp(K_grid)

    # No-negatividad básica
    C_grid = np.maximum(C_grid, 0.0)

    # Segunda derivada numérica: f(K) ~ d2C_tilde/dK^2
    d1 = np.gradient(C_grid, K_grid)
    d2 = np.gradient(d1, K_grid)

    pdf = np.maximum(d2, 0.0)

    # Normalización
    integral = np.trapz(pdf, K_grid)
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("La densidad obtenida es degenerada (integral <= 0).")

    pdf /= integral

    # --- Ajuste al forward teórico ---
    F_theo = spot * np.exp((r_annual - q_annual) * T)
    m_rnd = np.trapz(K_grid * pdf, K_grid)

    if m_rnd > 0 and np.isfinite(m_rnd):
        scale = F_theo / m_rnd
        # Relajamos un poco el clipping para corregir sesgos grandes
        scale = np.clip(scale, 0.2, 3.0)

        K_scaled = K_grid * scale
        pdf_scaled = pdf / scale  # cambio de variable Y = aX

        # Renormalizamos por si acaso
        integral2 = np.trapz(pdf_scaled, K_scaled)
        if integral2 > 0:
            pdf_scaled /= integral2
        else:
            K_scaled, pdf_scaled = K_grid, pdf

        return K_scaled, pdf_scaled

    # Si algo salió raro, devolvemos la versión sin escalar
    return K_grid, pdf



def compute_rnd_from_clean_calls(
    clean_calls_df: pd.DataFrame,
    spot: float,
    valuation_date: pd.Timestamp,
    expiry_date: pd.Timestamp,
    r_annual: float,
    q_annual: float = 0.0,
    moneyness_low: float = 0.5,
    moneyness_high: float = 1.6,
    n_grid: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Igual que compute_rnd_from_calls, pero partiendo de un DataFrame con
    columnas ['strike', 'call_price_clean'] ya construidas a partir de CALL+PUT
    (paridad put-call).

    Se asume que clean_calls_df viene de build_clean_calls_from_chain().
    """
    if clean_calls_df.empty:
        raise ValueError("clean_calls_df está vacío; no se puede construir la RND.")

    df = clean_calls_df.copy()
    K = df["strike"].astype(float).values
    C = df["call_price_clean"].astype(float).values

    # Rango de moneyness alrededor del spot
    lo = moneyness_low * spot
    hi = moneyness_high * spot
    mask = (K >= lo) & (K <= hi)
    K = K[mask]
    C = C[mask]

    if K.size < 5:
        raise ValueError("No hay suficientes strikes limpios en el rango alrededor del spot.")

    # Ordenar por strike
    order = np.argsort(K)
    K = K[order]
    C = C[order]

    # Tiempo a vencimiento en años
    T_days = (pd.Timestamp(expiry_date) - pd.Timestamp(valuation_date)).days
    T = max(T_days / 365.25, 1e-6)

    # Quitamos descuento: C_tilde(K) = C0 * exp(rT) = E[(S_T - K)+]
    C_tilde = C * np.exp(r_annual * T)

    # Interpolación PCHIP
    interp = PchipInterpolator(K, C_tilde, extrapolate=False)

    K_grid = np.linspace(K.min(), K.max(), n_grid)
    C_grid = interp(K_grid)
    C_grid = np.maximum(C_grid, 0.0)

    # Segunda derivada numérica
    d1 = np.gradient(C_grid, K_grid)
    d2 = np.gradient(d1, K_grid)
    pdf = np.maximum(d2, 0.0)

    # Normalización
    integral = np.trapz(pdf, K_grid)
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("La densidad obtenida es degenerada (integral <= 0).")
    pdf /= integral

    # Ajuste al forward teórico
    F_theo = spot * np.exp((r_annual - q_annual) * T)
    m_rnd = np.trapz(K_grid * pdf, K_grid)

    if m_rnd > 0 and np.isfinite(m_rnd):
        scale = F_theo / m_rnd
        scale = np.clip(scale, 0.2, 3.0)  # mismo criterio que en compute_rnd_from_calls

        K_scaled = K_grid * scale
        pdf_scaled = pdf / scale  # cambio de variable Y = aX

        integral2 = np.trapz(pdf_scaled, K_scaled)
        if integral2 > 0:
            pdf_scaled /= integral2
        else:
            K_scaled, pdf_scaled = K_grid, pdf

        return K_scaled, pdf_scaled

    return K_grid, pdf
