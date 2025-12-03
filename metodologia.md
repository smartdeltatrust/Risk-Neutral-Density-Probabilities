# Metodología

> Metodología para la construcción de un mapa de calor de precios
> históricos y cono de probabilidad implícita a partir de opciones
> sobre acciones.

---

## 1. Objetivo general

El objetivo de la aplicación es visualizar, en un único panel:

1. La trayectoria histórica del precio de una acción (en formato OHLC).
2. La **distribución neutral al riesgo** del precio al vencimiento de una
   opción (risk–neutral density, RND) inferida a partir de la cadena de
   opciones.
3. Un **cono de probabilidad** hacia el futuro que muestra intervalos de
   confianza del 68 % y 95 % sobre el nivel futuro del subyacente,
   derivados de esa RND.
4. Opcionalmente, un **heatmap** de densidad que permite ver cómo se
   reparte la probabilidad sobre el eje precio–tiempo.

La filosofía es similar a las vistas de terminal profesional (Bloomberg,
Tastytrade, etc.), pero usando una metodología explícita y reproducible
a partir de datos públicos.

---

## 2. Fuentes de datos y preprocesamiento

### 2.1. Datos históricos del subyacente

Los precios históricos se obtienen mediante la API de **Finviz Elite**:

- Se consulta el endpoint de exportación (`quote_export.ashx`) con:
  - Ticker (`t=`), por ejemplo `MSFT`.
  - Rango temporal (`p=`), por ejemplo `ytd`, `y1`, `m6`, etc.
- La respuesta es un CSV con columnas:
  - `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

En la aplicación:

- Se normaliza la columna de fechas a `datetime64[ns]`.
- Se asegura el orden cronológico ascendente.
- Se utiliza la serie de **cierre** (`Close`) como referencia principal
  para la parte histórica.

La **fecha de valoración** \(t_0\) se define como la última fecha
histórica disponible:
\[
t_0 = \max\{\text{Date}\}.
\]
El **spot** o precio actual del subyacente se toma como:
\[
S_0 = \text{Close}(t_0).
\]

### 2.2. Cadena de opciones

La cadena de opciones se obtiene también desde Finviz:

- Endpoint de opciones (`options.ashx`) con:
  - Ticker `t=`.
  - Fecha de vencimiento `e=` (cuando aplica).
- El resultado se normaliza a un `DataFrame` con columnas
  estandarizadas, típicamente:

  - `strike` — precio de ejercicio \(K\).
  - `option_type` — `"call"` o `"put"`.
  - `bid`, `ask`, `last_close`.
  - `open_int` — open interest.

Se realiza una limpieza mínima:

- Conversión a `float` de las columnas numéricas.
- Eliminación de filas con `strike` o precios inválidos (NaN, ≤ 0).

### 2.3. Lista de vencimientos

Para mejorar la experiencia de usuario, la **lista de vencimientos
disponibles** se obtiene vía `yfinance`:

- `yf.Ticker(ticker).options` devuelve una lista de fechas tipo
  `"YYYY-MM-DD"`.
- Se ordena esa lista por fecha creciente y se presenta en un `selectbox`.
- El usuario elige un vencimiento \(T\); la API de Finviz se sigue
  utilizando para los precios de opciones.

Esta combinación de fuentes separa claramente:

- **Finviz** → precios y volúmenes (información de mercado).
- **yfinance** → solo *metadatos* (lista de vencimientos).

---

## 3. Densidad neutral al riesgo a partir de la cadena de opciones

El núcleo matemático del proyecto es la extracción de la **densidad
neutral al riesgo** \( f_Q(S_T) \) del precio del subyacente \(S_T\) en
el vencimiento \(T\), a partir de precios de opciones.

### 3.1. Paridad put–call y “calls limpios”

Dado que en la práctica las cotizaciones de CALL y PUT pueden contener
ruido, se combina la información mediante la **paridad put–call**.

Sea:

- \(C_0(K)\): precio actual de un call europeo con strike \(K\) y vencimiento \(T\).
- \(P_0(K)\): precio actual del put correspondiente.
- \(r\): tasa libre de riesgo anual (constante).
- \(q\): dividend yield anual aproximado (constante).
- \(S_0\): precio spot actual del subyacente (último cierre).
- \(T\): tiempo a vencimiento (en años) desde la fecha de valoración \(t_0\).

La paridad put–call (ajustada por dividendos continuos) da:

\[
C_0(K) - P_0(K) = S_0 e^{-qT} - K e^{-rT}.
\]

Despejando el call teórico:

\[
C_{\text{parity}}(K) = P_0(K) + S_0 e^{-qT} - K e^{-rT}.
\]

**Algoritmo de limpieza:**

1. Se separan `calls` y `puts` por `option_type`.
2. Se indexa cada tabla por `strike`, y se toma la unión de strikes.
3. Para cada strike \(K\):

   - Se define un precio "mid" del call:
     \[
     C_{\text{direct}}(K) =
     \begin{cases}
        \frac{\text{bid} + \text{ask}}{2}, & \text{si bid, ask > 0} \\
        \text{bid}, & \text{si solo bid > 0} \\
        \text{ask}, & \text{si solo ask > 0} \\
        \text{last\_close}, & \text{si solo last\_close > 0} \\
        \text{NaN}, & \text{en otro caso}
     \end{cases}
     \]

   - Si existe un put con ese strike, se calcula
     \(C_{\text{parity}}(K)\) como arriba.

   - Se construye una lista de candidatos positivos:
     \(\{ C_{\text{direct}}, C_{\text{parity}} \} \cap (0,\infty)\).

   - Si hay al menos un candidato, se define el **call limpio**:
     \[
     C_{\text{clean}}(K) = \text{media de los candidatos válidos}.
     \]

4. Se obtiene así un `DataFrame` `clean_calls_df` con columnas:
   `['strike', 'call_price_clean']`.

Esta etapa explota la información redundante CALL/PUT y reduce el ruido
individual de las cotizaciones.

> Si, por alguna razón, no se puede construir `clean_calls_df` (falta
> de puts, datos inconsistentes), se recurre a una versión simplificada
> basada solo en calls (`compute_rnd_from_calls`).

### 3.2. Selección de rangos de moneyness

Para evitar strikes demasiado lejanos con poca liquidez, se restringe el
rango de análisis a un intervalo de **moneyness**:

\[
K \in [m_{\text{low}} S_0, \, m_{\text{high}} S_0],
\]
donde típicamente:
- \(m_{\text{low}} \approx 0.5\),
- \(m_{\text{high}} \approx 1.6\).

Además, se exige un número mínimo de strikes válidos (por ejemplo,
\(\geq 5\)) para considerar la RND confiable.

### 3.3. Interpolación de precios en strike

Sea el conjunto de strikes limpios \(\{K_i\}\) con sus precios
\(\{ C_i \}\). Se define el tiempo a vencimiento:

\[
T = \frac{(T_{\text{fecha}} - t_0)}{365.25} \quad (\text{en años}).
\]

Se elimina el descuento temporal para aproximar la esperanza del payoff:

\[
\tilde{C}(K) = C_0(K)\, e^{rT} \approx \mathbb{E}_Q[(S_T - K)^+].
\]

Se construye una **interpolación PCHIP** (`PchipInterpolator`) sobre los
puntos \((K_i,\tilde{C}_i)\), generando una función suave
\(\tilde{C}(K)\) definida sobre un grid denso de strikes \(K_{\text{grid}}\).

La elección de PCHIP (en lugar de spline cúbico clásico) obedece a:

- Mejor comportamiento monótono.
- Menos oscilaciones extrañas entre puntos.
- Mayor estabilidad numérica para derivadas.

### 3.4. Densidad Breeden–Litzenberger

El resultado clásico de Breeden & Litzenberger (1978) establece que,
bajo ciertos supuestos, la **segunda derivada** de \(\tilde{C}(K)\) con
respecto a \(K\) está relacionada con la densidad neutral al riesgo:

\[
\frac{\partial^2 \tilde{C}(K)}{\partial K^2}
= \frac{\partial^2}{\partial K^2} \mathbb{E}_Q[(S_T - K)^+]
= f_Q(K),
\]
donde \(f_Q(K)\) es la densidad de \(S_T\) evaluada en \(K\).

En la implementación:

1. Se evalúa \(\tilde{C}(K)\) en un grid equiespaciado \(K_{\text{grid}}\).
2. Se calcula numéricamente la primera derivada:
   \[
   d_1(K) = \frac{\partial \tilde{C}(K)}{\partial K}
   \]
   mediante `np.gradient`.
3. Se calcula la segunda derivada:
   \[
   d_2(K) = \frac{\partial^2 \tilde{C}(K)}{\partial K^2}
   \approx \texttt{np.gradient}(d_1(K), K_{\text{grid}}).
   \]
4. Se define la densidad cruda:
   \[
   f_{\text{raw}}(K) = \max\{ d_2(K), 0 \},
   \]
   truncando valores negativos numéricos.
5. Se normaliza:
   \[
   f_Q(K) = \frac{f_{\text{raw}}(K)}{\int f_{\text{raw}}(u)\,du}.
   \]

El resultado es un par \((K_{\text{grid}}, f_Q(K_{\text{grid}}))\) que
representa la **risk–neutral density**.

### 3.5. Ajuste al forward teórico

Por construcción numérica, el primer momento de \(f_Q\) puede no coincidir
exactamente con el forward teórico bajo tasas constantes \(r, q\):

\[
F_{\text{theo}} = S_0 \, e^{(r - q) T}.
\]

Se calcula la media de la RND:

\[
\mu_{\text{RND}} = \int K\, f_Q(K) \, dK.
\]

Se define un factor de escala:

\[
\alpha = \frac{F_{\text{theo}}}{\mu_{\text{RND}}}.
\]

Para evitar correcciones excesivas por ruido, se limita \(\alpha\) a un
rango razonable (por ejemplo \([0.2, 3.0]\)). Finalmente se realiza un
cambio de variable:

- Nuevo soporte:
  \[
  K' = \alpha K.
  \]
- Nueva densidad:
  \[
  f'_Q(K') = \frac{1}{\alpha} f_Q\left(\frac{K'}{\alpha}\right).
  \]

Después se renormaliza
\(\int f'_Q(K') dK' = 1\).

En la práctica se implementa de forma discreta, devolviendo:
\[
(K_{\text{grid}}', \, f_Q'(K_{\text{grid}}')).
\]

---

## 4. Construcción de la densidad conjunta precio–tiempo

El gráfico principal muestra una **matriz de densidad**
\(\rho(p, t)\) sobre un eje de precios \(p\) común y un eje temporal
que incluye pasado y futuro.

### 4.1. Eje de precios común

Sea `rnd_list` la colección de pares \((K_{\text{grid}}, f_Q)\) para
uno o varios vencimientos.

Además de los precios históricos \(\{ S_t \}\) (cierres), se utiliza la
información de las colas de cada RND.

Para cada RND:

1. Se calcula una CDF discreta:
   \[
   F(K_j) = \sum_{i \leq j} f_Q(K_i).
   \]
2. Se aproximan los cuantiles 1 % y 99 %:
   \[
   K_{1\%} \approx F^{-1}(0.01), \quad
   K_{99\%} \approx F^{-1}(0.99).
   \]

En lugar de usar todo el rango \([\min K, \max K]\), se guarda solo
\([K_{1\%}, K_{99\%}]\), evitando colas extremas con probabilidad casi
nula que ensanchan innecesariamente el eje Y.

Se construye un conjunto de precios candidato:

\[
\mathcal{P} = \{\text{Close}_t\}_{t\le t_0}
\cup \{ K_{1\%}^{(m)}, K_{99\%}^{(m)} \}_{m}.
\]

Se define:

\[
p_{\min} = \min \mathcal{P}, \quad p_{\max} = \max \mathcal{P},
\]
y se amplía ligeramente con un factor de padding
\(\delta \in (0,1)\) (por ejemplo 10 %):

\[
p_{\min}' = p_{\min} - \delta(p_{\max} - p_{\min}), \quad
p_{\max}' = p_{\max} + \delta(p_{\max} - p_{\min}).
\]

El **price grid** final es:

\[
\text{price\_grid} = \{ p_1, \dots, p_N \}
\]
discreto y equiespaciado entre \(p_{\min}'\) y \(p_{\max}'\).

### 4.2. Eje temporal

Se define el conjunto de fechas:

- `dates_hist`: fechas históricas de precios OHLC.
- `future_expiries`: fechas de vencimiento \(\{T_m\}\) posteriores o
  iguales a \(t_0\).

Se construye:

- `dates_future`: rango de días hábiles desde \(t_0\) hasta el último
  vencimiento disponible (`pd.date_range(..., freq="B")`).

El eje temporal global:

\[
\text{dates\_all} = \text{unique}(\text{dates\_hist} \cup \text{dates\_future}).
\]

### 4.3. Densidad histórica

Para cada fecha histórica \(t \le t_0\) se aproxima una densidad
condicionada de forma muy simple:

- Sea \(S_t\) el cierre de ese día.
- Se define una desviación proporcional al precio:
  \[
  \sigma_t = \max(S_t \cdot \sigma_{\text{rel}}, \, \epsilon),
  \]
  con \(\sigma_{\text{rel}}\) fijo (por ejemplo 1–2 %) y
  \(\epsilon > 0\) pequeño para evitar problemas numéricos.

- La densidad se toma como Gaussiana en el eje de precios:

  \[
  \rho(p,t) \propto 
  \exp\left(-\frac{1}{2} \left(\frac{p - S_t}{\sigma_t}\right)^2\right).
  \]

- Se normaliza para que:
  \[
  \int \rho(p,t)\,dp = 1.
  \]

En términos de implementación:

```python
pdf = gaussian_density(price_grid, S_t, sigma_t)
pdf /= np.trapz(pdf, price_grid)
density[:, j] = pdf
