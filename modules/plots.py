import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def compute_quantile_bands(price_grid: np.ndarray, density: np.ndarray):
    """
    A partir de la matriz density [n_precios x n_tiempos], calcula
    bandas 68% (16-84) y 95% (2.5-97.5) y la mediana (50%) para cada columna.
    Devuelve:
      q2p5, q16, q50, q84, q97p5  (cada uno array len = n_tiempos)
    """
    n_p, n_t = density.shape
    if n_p < 2 or n_t < 1:
        return None, None, None, None, None

    dx = price_grid[1] - price_grid[0]
    q2p5 = np.full(n_t, np.nan)
    q16 = np.full(n_t, np.nan)
    q50 = np.full(n_t, np.nan)
    q84 = np.full(n_t, np.nan)
    q97p5 = np.full(n_t, np.nan)

    for j in range(n_t):
        pdf = density[:, j]
        if not np.isfinite(pdf).any():
            continue
        if pdf.sum() <= 0:
            continue

        pdf = np.clip(pdf, 0, None)
        cdf = np.cumsum(pdf) * dx
        if cdf[-1] <= 0:
            continue
        cdf /= cdf[-1]

        def q_level(level: float):
            idx = np.searchsorted(cdf, level)
            idx = min(max(idx, 0), len(price_grid) - 1)
            return price_grid[idx]

        q2p5[j] = q_level(0.025)
        q16[j] = q_level(0.16)
        q50[j] = q_level(0.50)
        q84[j] = q_level(0.84)
        q97p5[j] = q_level(0.975)

    return q2p5, q16, q50, q84, q97p5

def plot_main_figure(
    quotes_df: pd.DataFrame,
    dates_all: np.ndarray,
    price_grid: np.ndarray,
    density: np.ndarray,
    expiry_dates: list[pd.Timestamp],
    valuation_date: pd.Timestamp,
    show_heatmap: bool = True,
    show_past_rnd: bool = False,
):
    """
    Figura principal: OHLC + cono RND (68/95%) + mediana +
    opcionalmente heatmap y bandas RND históricas.
    """

    valuation_date = pd.Timestamp(valuation_date)

    # Eje X y máscaras pasado/futuro
    x_vals = pd.to_datetime(dates_all)
    y_vals = price_grid
    mask_future = x_vals >= valuation_date
    mask_past = ~mask_future

    # Limitamos velas a la ventana visible
    mask_hist_win = (quotes_df["Date"] >= x_vals.min()) & (quotes_df["Date"] <= x_vals.max())
    quotes_win = quotes_df.loc[mask_hist_win].copy()
    if quotes_win.empty:
        quotes_win = quotes_df.copy()

    # Bandas de confianza (todo el grid)
    q2p5, q16, q50, q84, q97p5 = compute_quantile_bands(price_grid, density)

    fig = go.Figure()

    # -------------------------------------------------
    # 1) Heatmap (solo si show_heatmap = True)
    # -------------------------------------------------
    if show_heatmap:
        z_plot = density.copy()
        if mask_past.any():
            z_plot[:, mask_past] *= 0.0  # pasado muy tenue

        z_pos = z_plot[np.isfinite(z_plot) & (z_plot > 0)]
        zmax = float(np.percentile(z_pos, 90)) if z_pos.size > 0 else 1.0

        heat = go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=z_plot,
            colorscale=[
                [0.0, "rgba(0,0,0,0.0)"],
                [0.2, "#001f3f"],
                [0.4, "#003f7f"],
                [0.6, "#0074D9"],
                [0.8, "#00E5FF"],
                [0.9, "#FFFF00"],
                [1.0, "#FFFF99"],
            ],
            zmin=0.0,
            zmax=zmax,
            zsmooth="best",
            opacity=0.6,
            colorbar=dict(
                title="Density",
                x=1.05,
                xanchor="left",
                y=0.5,
                len=0.8,
            ),
            showscale=True,
            name="Density",
            showlegend=False,
        )
        fig.add_trace(heat)

    # -------------------------------------------------
    # 2) Velas OHLC
    # -------------------------------------------------
    fig.add_trace(
        go.Candlestick(
            x=quotes_win["Date"],
            open=quotes_win["Open"],
            high=quotes_win["High"],
            low=quotes_win["Low"],
            close=quotes_win["Close"],
            name="Underlying (OHLC)",
            increasing=dict(
                line=dict(color="#00FF7F", width=1.4),
                fillcolor="rgba(0,255,127,0.40)",
            ),
            decreasing=dict(
                line=dict(color="#FF4D4D", width=1.4),
                fillcolor="rgba(255,77,77,0.40)",
            ),
            showlegend=True,
        )
    )

    # -------------------------------------------------
    # 3) Bandas RND: SOLO futuro (cono) + mediana
    # -------------------------------------------------
    if q2p5 is not None:
        # --- futuro: bandas rellenas 95% y 68% ---
        if mask_future.any():
            x_future = x_vals[mask_future]
            q2p5_f = q2p5[mask_future]
            q97p5_f = q97p5[mask_future]
            q16_f = q16[mask_future]
            q84_f = q84[mask_future]

            # 95% lower (línea visible + hover propio)
            trace_95_lower = go.Scatter(
                x=x_future,
                y=q2p5_f,
                mode="lines",
                line=dict(color="rgba(0,90,255,0.9)", width=1),
                showlegend=False,
                hovertemplate="RND 95% low: %{y:.2f}<extra></extra>",
            )
            # 95% upper con relleno
            trace_95_upper = go.Scatter(
                x=x_future,
                y=q97p5_f,
                mode="lines",
                line=dict(color="rgba(0,90,255,0.9)", width=1),
                fill="tonexty",
                fillcolor="rgba(0,90,255,0.18)",
                name="RND 95%",
                hovertemplate="RND 95% high: %{y:.2f}<extra></extra>",
            )

            # 68% lower
            trace_68_lower = go.Scatter(
                x=x_future,
                y=q16_f,
                mode="lines",
                line=dict(color="rgba(0,191,255,0.9)", width=1),
                showlegend=False,
                hovertemplate="RND 68% low: %{y:.2f}<extra></extra>",
            )
            # 68% upper con relleno
            trace_68_upper = go.Scatter(
                x=x_future,
                y=q84_f,
                mode="lines",
                line=dict(color="rgba(0,191,255,0.9)", width=1),
                fill="tonexty",
                fillcolor="rgba(0,191,255,0.25)",
                name="RND 68%",
                hovertemplate="RND 68% high: %{y:.2f}<extra></extra>",
            )

            fig.add_trace(trace_95_lower)
            fig.add_trace(trace_95_upper)
            fig.add_trace(trace_68_lower)
            fig.add_trace(trace_68_upper)

        # Mediana (hist + futuro), tenue para no tapar velas
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=q50,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.4)", width=1.4),
                name="RND Median",
                hovertemplate="RND Median: %{y:.2f}<extra></extra>",
            )
        )

        # Rango Y basado en todo el 95%
        all_low = np.nanmin(q2p5[np.isfinite(q2p5)])
        all_high = np.nanmax(q97p5[np.isfinite(q97p5)])
        if np.isfinite(all_low) and np.isfinite(all_high) and all_high > all_low:
            span = all_high - all_low
            y_min = all_low - 0.05 * span
            y_max = all_high + 0.05 * span
            fig.update_yaxes(range=[y_min, y_max])


    # -------------------------------------------------
    # 4) Líneas verticales de vencimiento
    # -------------------------------------------------
    if len(price_grid) > 0:
        y_min_shapes = float(np.nanmin(price_grid))
        y_max_shapes = float(np.nanmax(price_grid))
    else:
        y_min_shapes, y_max_shapes = 0.0, 1.0

    for d_exp in expiry_dates:
        x_val = pd.Timestamp(d_exp).to_pydatetime()

        fig.add_shape(
            type="line",
            x0=x_val,
            x1=x_val,
            y0=y_min_shapes,
            y1=y_max_shapes,
            xref="x",
            yref="y",
            line=dict(
                color="#FFFF00",
                width=1,
                dash="dash",
            ),
        )

        fig.add_annotation(
            x=x_val,
            y=y_max_shapes,
            text=str(pd.Timestamp(d_exp).date()),
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="#FFFF00", size=10),
        )

    # -------------------------------------------------
    # 5) Estética general
    # -------------------------------------------------
    fig.update_layout(
        template=None,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        xaxis_title="Date",
        yaxis_title="Price / Strike",
        title="Densidad de Probabilidad Implícita a Partir de Precios de Opciones",
        hovermode="x unified",
        font=dict(
            family="Roboto Mono, monospace",
            color="#E0E0E0",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="#444444",
            borderwidth=1,
        ),
        margin=dict(l=60, r=90, t=60, b=90),
    )


    fig.update_xaxes(
        showgrid=True,
        gridcolor="#333333",
        rangeslider_visible=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#333333",
    )

    st.plotly_chart(fig, use_container_width=True)


