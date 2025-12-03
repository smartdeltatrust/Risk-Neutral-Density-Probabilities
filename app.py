# app.py
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import os


from assets.config.settings import settings
from modules.data_loader import (
	fetch_quote_history,
	fetch_options_chain,
	clean_options_chain,
	fetch_available_expiries,   # <-- NUEVO
)
from modules.utils import (
	compute_rnd_from_calls,
	compute_rnd_from_clean_calls,
	build_time_price_density,
	build_clean_calls_from_chain,
)
from modules.plots import plot_main_figure

# Asegurar raíz del proyecto en sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

st.set_page_config(
	page_title="Densidad de probabilidad implícita",
	layout="wide",
)



# === DETECCIÓN DE ENTORNO ===
APP_ENV = os.getenv("APP_ENV", "").strip().lower()
IS_DEV = APP_ENV in ("", "dev", "development")

# DEBUG TEMPORAL: para ver qué está pasando
st.write(f"DEBUG APP_ENV='{APP_ENV}', IS_DEV={IS_DEV}")


# --- Simple stub for "Pro" access validation ---
def es_usuario_pro(api_key: str) -> bool:
    """
    Stub de validación de API key.
    Más adelante deberás sustituir esto por una llamada a tu backend,
    base de datos o servicio de licencias.

    Por ahora, cualquier API key que empiece por 'PRO-' se considera válida.
    """
    if not api_key:
        return False
    return api_key.strip().upper().startswith("PRO-")

# En Stripe Dashboard creas un Payment Link para tu suscripción
STRIPE_PAYMENT_LINK = "https://buy.stripe.com/eVq3cx1Isbd74qLd6BcfK01"


# --------- CACHES ---------
@st.cache_data
def cached_quotes(ticker: str, range_code: str, auth_token: str):
	return fetch_quote_history(ticker, range_code, auth_token)


@st.cache_data
def cached_expiries(ticker: str, auth_token: str):
	"""Lista de vencimientos disponibles en Finviz para ese ticker."""
	return fetch_available_expiries(ticker, auth_token)


@st.cache_data
def cached_options(ticker: str, expiry: str, auth_token: str):
	raw = fetch_options_chain(ticker, expiry, auth_token)
	return clean_options_chain(raw)


IS_DEV = os.getenv("APP_ENV", "dev").lower() == "dev"


def main():
    # -------------------------------------------------
    # TÍTULO
    # -------------------------------------------------
    st.subheader("Risk-Neutral Density Probabilities from Options Prices")

    # =================================================
    # PAYWALL (Stripe) SOLO EN PRODUCCIÓN
    # En modo desarrollo (IS_DEV = True) este bloque se salta.
    # =================================================
    if not IS_DEV:
        with st.container():
            st.markdown("##### Pro access")

            col_key, col_cta = st.columns([2, 1])

            with col_key:
                pro_key = st.text_input(
                    "Pro API key",
                    type="password",
                    help="Enter the key you received after subscribing.",
                )

            with col_cta:
                st.markdown("Don’t have a key?")
                st.markdown(
                    f"[➡ Get Pro access with Stripe]({STRIPE_PAYMENT_LINK})",
                    unsafe_allow_html=True,
                )
                st.caption("You will be redirected to a secure Stripe Checkout page.")

            # Validación simple (stub)
            if not es_usuario_pro(pro_key):
                st.info(
                    "Enter a valid Pro key to unlock the options-implied probability cone "
                    "and heatmap. After you subscribe via Stripe, you can manually issue "
                    "or receive a key that starts with 'PRO-'."
                )
                st.stop()
    # ================== FIN PAYWALL ==================
    # -------------------------------------------------
    # CONTENEDORES EN ORDEN VISUAL
    # -------------------------------------------------
    chart_container = st.container()          # 1) gráfica principal
    below_chart_container = st.container()    # 2) checkbox heatmap
    controls_container = st.container()       # 3) controles

    # -------------------------------------------------
    # Token Finviz (solo desde settings / .env, SIN mostrar en UI)
    # -------------------------------------------------
    auth_token = settings.FINVIZ_AUTH_TOKEN
    if not auth_token:
        st.error(
            "Finviz token is not configured.\n\n"
            "Set FINVIZ_AUTH_TOKEN in your .env or in assets/config/settings.py."
        )
        st.stop()

    # -------------------------------------------------
    # CONTROLES (aparecen visualmente debajo de la gráfica)
    # -------------------------------------------------
    with controls_container:
        col1, col2 = st.columns(2)

        # --- Columna izquierda: ticker y rango histórico ---
        with col1:
            ticker = st.text_input("Ticker", value="MSFT").upper().strip()
            range_code = st.selectbox(
                "Historical range",
                options=["d1", "d5", "m1", "m3", "m6", "ytd", "y1", "y2", "y5", "max"],
                index=["d1", "d5", "m1", "m3", "m6", "ytd", "y1", "y2", "y5", "max"].index(
                    settings.DEFAULT_RANGE
                ),
            )

        # --- Columna derecha: vencimiento, ventanas y tasas ---
        with col2:
            # Vencimientos desde yfinance
            available_expiries: list[str] = []
            if ticker:
                try:
                    available_expiries = fetch_available_expiries(ticker)
                except RuntimeError as e:
                    st.warning(str(e))
                    available_expiries = []

            if available_expiries:
                today = pd.Timestamp.today().normalize()
                expiry_dates: list[pd.Timestamp | None] = []
                for s in available_expiries:
                    try:
                        expiry_dates.append(pd.to_datetime(s))
                    except Exception:
                        expiry_dates.append(None)

                # Vencimiento más cercano dentro de los próximos 30 días
                best_idx = None
                best_days = None
                for idx, dt in enumerate(expiry_dates):
                    if dt is None:
                        continue
                    days = (dt - today).days
                    if days < 0:
                        continue  # ya vencido
                    if days <= 30:
                        if (best_days is None) or (days < best_days):
                            best_days = days
                            best_idx = idx

                # Si no hay ninguno en <= 30 días, usamos el futuro más cercano
                if best_idx is None:
                    for idx, dt in enumerate(expiry_dates):
                        if dt is None:
                            continue
                        days = (dt - today).days
                        if days >= 0:
                            best_idx = idx
                            break

                # Si aún así no encontramos nada, usamos el último
                if best_idx is None:
                    best_idx = len(available_expiries) - 1

                expiry_str = st.selectbox(
                    "Expiry",
                    options=available_expiries,
                    index=best_idx,
                    help="Expiries available from yfinance",
                )
            else:
                expiry_str = st.text_input(
                    "Expiry (YYYY-MM-DD)",
                    value="2025-12-19",
                    help="Type the expiry manually if the list is not available.",
                )

            # Ventanas temporal
            col_hist_win, col_fut_win = st.columns(2)
            with col_hist_win:
                past_days = st.number_input(
                    "Historical window (days)",
                    min_value=30,
                    max_value=2000,
                    value=252,
                    step=10,
                )
            with col_fut_win:
                future_days = 120  # fijo por diseño

            # Tasas
            r_rate = st.number_input(
                "Risk-free rate (r, annual)",
                value=float(settings.DEFAULT_RATE),
                step=0.005,
                format="%.3f",
            )
            q_rate = st.number_input(
                "Dividend yield (q, annual)",
                value=0.0,
                step=0.005,
                format="%.3f",
                help="Approximate dividend yield (0 if not applicable).",
            )

    # Parámetro de anchura histórica (fijo)
    hist_sigma_rel = float(settings.HIST_SIGMA_REL)

    # -------------------------------------------------
    # 1) Datos históricos (quotes)
    # -------------------------------------------------
    try:
        quotes_df = cached_quotes(ticker, range_code, auth_token)
    except RuntimeError as e:
        st.error(f"Could not download historical data from Finviz: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while downloading historical data: {e}")
        st.stop()

    if quotes_df.empty:
        st.error("No historical data for that ticker / range.")
        st.stop()

    valuation_date = quotes_df["Date"].max()
    spot = float(
        quotes_df.loc[quotes_df["Date"] == valuation_date, "Close"].iloc[0]
    )

    # -------------------------------------------------
    # 2) Vencimiento
    # -------------------------------------------------
    try:
        expiry_date = pd.to_datetime(expiry_str)
    except Exception:
        st.error("Invalid expiry date format.")
        st.stop()

    # -------------------------------------------------
    # 3) Cadena de opciones desde Finviz
    # -------------------------------------------------
    try:
        options_df = cached_options(ticker, expiry_str, auth_token)
    except RuntimeError as e:
        st.error(f"Could not download options chain from Finviz: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while downloading options: {e}")
        st.stop()

    if options_df.empty:
        st.error("No options found for that expiry.")
        st.stop()

    # -------------------------------------------------
    # 4) RND neutral al riesgo con CALL+PUT limpios
    # -------------------------------------------------
    try:
        clean_calls_df = build_clean_calls_from_chain(
            options_df,
            S0=spot,
            valuation_date=valuation_date,
            expiry_date=expiry_date,
            r_annual=r_rate,
            q_annual=q_rate,
        )

        if not clean_calls_df.empty:
            K_grid, pdf_K = compute_rnd_from_clean_calls(
                clean_calls_df,
                spot=spot,
                valuation_date=valuation_date,
                expiry_date=expiry_date,
                r_annual=r_rate,
                q_annual=q_rate,
            )
        else:
            K_grid, pdf_K = compute_rnd_from_calls(
                options_df,
                spot=spot,
                valuation_date=valuation_date,
                expiry_date=expiry_date,
                r_annual=r_rate,
                q_annual=q_rate,
            )

    except Exception as e:
        st.error(f"Could not build RND from options chain: {e}")
        st.stop()

    rnd_by_date = {pd.Timestamp(expiry_date): (K_grid, pdf_K)}

    # -------------------------------------------------
    # 5) Matriz tiempo-precio (histórico + futuro)
    # -------------------------------------------------
    dates_all, price_grid, density = build_time_price_density(
        quotes_df,
        rnd_by_date,
        hist_sigma_rel=hist_sigma_rel,
        interpolate_future=True,
    )

    min_date = valuation_date - pd.Timedelta(days=int(past_days))
    max_date = valuation_date + pd.Timedelta(days=int(future_days))

    mask = (dates_all >= np.datetime64(min_date)) & (
        dates_all <= np.datetime64(max_date)
    )
    if mask.sum() == 0:
        st.warning("Selected window does not overlap available data.")
        st.stop()

    dates_win = dates_all[mask]
    density_win = density[:, mask]

    expiry_dates_win = [
        d
        for d in rnd_by_date.keys()
        if (pd.Timestamp(d) >= min_date) and (pd.Timestamp(d) <= max_date)
    ]

    # -------------------------------------------------
    # 6) Checkbox de HEATMAP justo debajo de la gráfica
    # -------------------------------------------------
    with below_chart_container:
        show_heatmap = st.checkbox(
            "Show density heatmap",
            value=False,
            key="chk_density_heatmap",
        )

    # -------------------------------------------------
    # 7) DIBUJAR la gráfica PRINCIPAL
    # -------------------------------------------------
    with chart_container:
        plot_main_figure(
            quotes_df,
            dates_win,
            price_grid,
            density_win,
            expiry_dates=expiry_dates_win,
            valuation_date=valuation_date,
            show_heatmap=show_heatmap,
        )


if __name__ == "__main__":
    main()


st.subheader("Explanation")

st.markdown(
    r"""
This chart shows how the options market assigns probabilities to different price levels over time.
The heatmap translates those probabilities into colors so you can see where probability mass is concentrated.

Each point in the time–price plane in the future has an associated density: if the color is very faint,
the market sees that scenario as unlikely; if the color is more intense, many price paths compatible with
current option prices pass through that region.

Historical candles show the prices that actually occurred in the past, while the cone and heatmap show
which combinations of date and price are consistent with option prices under the risk–neutral measure.

You can think of an entire swarm of possible future price paths: the heatmap highlights in brighter colors
the zones where more trajectories accumulate, according to option prices, and leaves almost black the zones
where almost no simulated path arrives.

For example, if 60 days from now the brightest area is near a price of 420, this means that, under the market’s
risk–neutral view, it is more likely to find the price around 420 than far above or below that value, and the
68% and 95% bands indicate ranges where most of that probability is concentrated.

Working with implied densities instead of a single “target price” lets you evaluate tail risk, asymmetries
and extreme scenarios, which makes this visualization especially useful to design strategies, size positions,
and understand how the market is pricing future uncertainty.
"""
)

st.markdown(
    r"""
### Mathematical summary of the methodology

We start from the option chain and build *clean* call prices.
If $C(K)$ is the call price and $P(K)$ the put price at the same strike $K$,
with spot $S_0$, risk–free rate $r$ and dividend yield $q$, we use:
"""
)

st.latex(
    r"""
C_{\text{clean}}(K) \approx
\begin{cases}
\dfrac{\text{bid} + \text{ask}}{2} & \text{if there is a valid spread} \\
P(K) + S_0 e^{-qT} - K e^{-rT} & \text{(put–call parity)} \\
\end{cases}
"""
)

st.markdown(
    r"""
Then we remove discounting:
"""
)

st.latex(
    r"""
\tilde C(K) = C_{\text{clean}}(K)\, e^{rT}
\approx
\mathbb{E}_Q\big[(S_T - K)^+\big],
"""
)

st.markdown(
    r"""
and we apply the Breeden–Litzenberger formula to obtain the risk–neutral
density in strike space:
"""
)

st.latex(
    r"""
f_Q(K) = \frac{\partial^2 \tilde C(K)}{\partial K^2}.
"""
)

st.markdown(
    r"""
Numerically, we interpolate $\tilde C(K)$ on a grid $K_{\text{grid}}$,
compute second derivatives via finite differences, force $f_Q(K) \ge 0$
and normalize:
"""
)

st.latex(
    r"""
\int f_Q(K)\, dK = 1,
"""
)

st.markdown(
    r"""
also adjusting the first moment to match the theoretical forward:
"""
)

st.latex(
    r"""
\mathbb{E}_Q[S_T]
=
\int K\, f_Q(K)\, dK
\approx
S_0 e^{(r - q)T}.
"""
)

st.markdown(
    r"""
On each historical date $t$ we model intraday uncertainty as a Gaussian
centered at the close $S_t$:
"""
)

st.latex(
    r"""
p_{\text{hist}}(s \mid t)
\propto
\exp\left(
-\frac{1}{2}\,
\frac{(s - S_t)^2}{(\sigma_{\text{hist}} S_t)^2}
\right),
"""
)

st.markdown(
    r"""
with fixed $\sigma_{\text{hist}}$ relative to the price.
For each time $t$ (past or future) we build the density $p_t(s)$ on a price grid
and compute quantiles numerically. The quantile $q_\alpha(t)$ satisfies:
"""
)

st.latex(
    r"""
\int_{-\infty}^{q_\alpha(t)} p_t(s)\, ds = \alpha,
"""
)

st.markdown(
    r"""
and from these we obtain the 68% and 95% confidence bands that define
the probability cone shown in the chart.
"""
)
