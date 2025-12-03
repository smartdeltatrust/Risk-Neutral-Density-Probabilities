# Proyecto: Risk-Neutral Probability Cone (RND Options Analytics)

## 1. Visión general

Este proyecto implementa una aplicación web (Streamlit) que construye y visualiza la **densidad neutral al riesgo (RND)** implícita en los precios de las opciones, y la representa como un **“cono de probabilidad” sobre la serie histórica de precios** de una acción o ETF.

La app muestra en un mismo gráfico:

- Velas OHLC históricas en estilo terminal (fondo negro, velas verde/rojo).
- Bandas de confianza del 68 % y 95 % de la RND a futuro (cono de probabilidad).
- La mediana neutral al riesgo.
- Opcionalmente, un mapa de calor de densidades (heatmap) superpuesto.

Objetivo: ofrecer a traders retail avanzados una visualización tipo “Bloomberg / Tastytrade” sin necesidad de acceder a plataformas institucionales.

---

## 2. Público objetivo e intención del producto

Público objetivo inicial:

- Traders retail avanzados de opciones sobre acciones USA.
- Usuarios familiarizados con IV, delta, skew, etc., pero sin acceso a herramientas pro.
- Interesados en ver la distribución implícita del subyacente y no solo volatilidad o griegas.

Intención de negocio:

- Evolucionar la app a un **SaaS de suscripción**:
  - Plan mensual (ej. 19–29 USD) para acceso ilimitado a tickers y vencimientos.
  - Integración con **Stripe** para cobro recurrente.
- Más adelante:
  - Sustituir yfinance/Finviz por un proveedor de datos con licencia comercial (Polygon, Intrinio, etc.).
  - Añadir paneles adicionales (analítica de opciones, modelos basados en IA, ranking de subyacentes, etc.).

---

## 3. Arquitectura técnica actual

### 3.1 Stack

- Lenguaje: Python.
- Framework UI: Streamlit.
- Gráficos: Plotly (graph_objects).
- Cálculo numérico: NumPy, SciPy (PchipInterpolator), pandas.
- Estilo visual: tema oscuro tipo terminal (fondo negro, colores cian/azul/amarillo).

### 3.2 Estructura de módulos

- `app.py`
  - Punto de entrada de Streamlit.
  - Controla:
    - Inputs de usuario (ticker, rango histórico, vencimiento, ventanas temporal, tasas r y q).
    - Descarga de datos históricos y opciones.
    - Construcción de la RND.
    - Construcción de la matriz tiempo–precio–densidad.
    - Llamado a las funciones de plotting.
  - Integra el checkbox `Show density heatmap` justo debajo del gráfico.
  - Selecciona por defecto el vencimiento más cercano dentro de los próximos 30 días usando la lista de expiries.

- `modules/data_loader.py`
  - `fetch_quote_history(ticker, range_code, auth_token)`
    - Descarga histórico OHLC desde Finviz (en esta versión).
  - `fetch_options_chain(ticker, expiry, auth_token)`
    - Descarga la cadena de opciones (calls y puts) para un vencimiento.
  - `clean_options_chain(raw_df)`
    - Limpia nombres de columnas, tipifica numéricos, etc.
  - `fetch_available_expiries(ticker)`
    - Usa yfinance para obtener lista de vencimientos disponibles (`['2025-06-20', ...]`).

  En producción, estas funciones deben ser reemplazadas por una capa `data_provider` con un proveedor de datos comercial (Polygon, Intrinio…).

- `modules/utils.py`
  - `gaussian_density(x, mu, sigma)`
    - Calcula densidad normal univariante, usada para modelar incertidumbre histórica intradía.
  - `build_price_axis(quotes_df, rnd_list, price_padding, n_price_points)`
    - Construye el eje de precios global.
    - Usa precios históricos y, para la RND, el rango entre los cuantiles ~1 % y 99 % para evitar colas extremas.
  - `build_time_price_density(quotes_df, rnd_by_date, hist_sigma_rel, interpolate_future=True)`
    - Construye la matriz de densidad `density[precio, tiempo]`:
      - Para fechas históricas: una gaussiana alrededor del cierre.
      - Para fechas futuras: interpolación en el tiempo entre densidades de vencimientos (por ahora un vencimiento).
  - `build_clean_calls_from_chain(options_df, S0, valuation_date, expiry_date, r_annual, q_annual)`
    - Combina precios de CALL y PUT utilizando paridad put-call para construir precios de call “limpios” por strike.
  - `compute_realized_conf_band(quotes_df, horizon_days)`
    - Calcula bandas históricas 68/95 % (actualmente no se muestran en el gráfico principal).
  - `compute_rnd_from_calls(options_df, spot, valuation_date, expiry_date, r_annual, q_annual, ...)`
    - Construye la RND usando **sólo calls**, filtrando por OI, rango de moneyness, etc.
  - `compute_rnd_from_clean_calls(clean_calls_df, spot, valuation_date, expiry_date, r_annual, q_annual, ...)`
    - Igual que lo anterior pero a partir de los precios limpios de call generados con CALL+PUT.
    - Implementa interpolación PCHIP sobre \( C_{\tilde{}}(K) \) y derivadas numéricas de segundo orden.

- `modules/plots.py`
  - `compute_quantile_bands(price_grid, density)`
    - Para cada columna temporal de la matriz de densidad, calcula:
      - Cuantiles 2.5 %, 16 %, 50 %, 84 %, 97.5 %.
  - `plot_main_figure(quotes_df, dates_all, price_grid, density, expiry_dates, valuation_date, show_heatmap)`
    - Construye la figura principal:
      - Velas OHLC estilizadas (verde para velas alcistas, rojo para velas bajistas).
      - Bandas RND 68 % y 95 % SOLO en el futuro:
        - Relleno semitransparente.
        - Líneas superior e inferior visibles con hover.
      - Mediana RND (línea blanca tenue) en todo el periodo.
      - Opcionalmente, heatmap de densidades (cuando `show_heatmap=True`).
      - Líneas verticales amarillas en fechas de vencimiento.

- `assets/config/settings.py`
  - Parámetros globales:
    - `FINVIZ_AUTH_TOKEN`
    - `DEFAULT_RANGE`
    - `HIST_SIGMA_REL`
    - `PRICE_PADDING`
    - `N_PRICE_POINTS`, `N_STRIKE_POINTS`
    - `DEFAULT_RATE`, etc.

---

## 4. Metodología matemática (resumen)

1. **Precios de call limpios**

A partir de la cadena de opciones:

- Se seleccionan CALL y PUT por strike \( K \), con bid, ask, last y open interest.
- Se calcula un mid-price:

\[
C_{\text{mid}}(K) \approx \frac{\text{bid} + \text{ask}}{2}
\]

cuando hay spread válido. Si no, se usan bid, ask o last como alternativas.

- Usando la paridad put-call, se construye otro estimador:

\[
C_{\text{parity}}(K) = P(K) + S_0 e^{-qT} - K e^{-rT}
\]

- Se promedian los candidatos válidos para obtener:

\[
C_{\text{clean}}(K)
\]

2. **Breeden–Litzenberger y densidad neutral al riesgo**

- Se quita el descuento temporal:

\[
\tilde C(K) = C_{\text{clean}}(K)\, e^{rT} \approx \mathbb{E}_Q[(S_T - K)^+]
\]

- Se interpola \(\tilde C(K)\) con PCHIP sobre una malla \(K_{\text{grid}}\).
- Se obtiene la segunda derivada numérica:

\[
f_Q(K) \approx \frac{\partial^2 \tilde C(K)}{\partial K^2}
\]

- Se fuerza \(f_Q(K) \ge 0\), se normaliza a 1 y se ajusta el primer momento para respetar el forward teórico:

\[
\mathbb{E}_Q[S_T] = \int K f_Q(K) dK \approx S_0 e^{(r-q)T}.
\]

3. **Construcción del cono de probabilidad**

- En cada fecha histórica \(t\), se modela:

\[
p_{\text{hist}}(s\mid t) \propto \exp\Big(-\frac{(s - S_t)^2}{2(\sigma_{\text{hist}} S_t)^2}\Big)
\]

con \(\sigma_{\text{hist}}\) proporcional al precio.

- Para fechas futuras, se interpola suavemente entre la densidad histórica (último día) y la RND al vencimiento, rellenando todas las fechas hasta expiry.
- Para cada tiempo \(t\), se calculan cuantiles numéricos:

\[
\int_{-\infty}^{q_\alpha(t)} p_t(s)\,ds = \alpha
\]

y de ahí las bandas 68 % y 95 % que definen el cono.

---

## 5. Estado actual de la UI

- Al cargar el app:
  - Se muestra el título.
  - Se calcula por defecto el vencimiento más cercano dentro de los siguientes 30 días (a partir de expiries vía yfinance).
  - Se centra la gráfica en una ventana temporal definida por:
    - `past_days` (por defecto 252).
    - `future_days` fijo a 30.
- El usuario puede:
  - Cambiar ticker.
  - Cambiar rango histórico.
  - Cambiar vencimiento (seleccionando de la lista).
  - Activar o desactivar el heatmap de densidad.
- La gráfica principal es limpia y legible:
  - Fondo negro.
  - Velas de precio claras.
  - Cono RND de 68/95 % con rellenos semitransparentes.
  - Mediana en blanco tenue.
  - Líneas de vencimiento en amarillo.

---

## 6. Próximos pasos hacia producción

1. Sustituir Finviz/yfinance por un proveedor de datos con licencia comercial.
2. Añadir una capa `data_provider` bien definida.
3. Integrar un paywall con Stripe (suscripciones mensuales/anuales).
4. Desplegar la app en un entorno estable (Render, Railway, AWS, etc.).
5. Añadir monitoreo básico de errores y latencias.
6. Iterar sobre feedback de usuarios reales (traders avanzados).
