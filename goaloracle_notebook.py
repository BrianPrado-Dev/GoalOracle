# %% [markdown]
# # ⚽ GoalOracle — Sistema avanzado de predicción de marcadores de fútbol internacional
#
# Pipeline reproducible (Jupyter / Google Colab) que combina **tres enfoques de modelado**:
#
# 1. **Modelo jerárquico bayesiano (MCMC con PyMC v5)** — estima fuerza de ataque/defensa
#    por selección + ventaja de localía, con verosimilitud de Poisson.
# 2. **Regresión con XGBoost** (`objective="count:poisson"`) sobre features históricas.
# 3. **Ajuste Dixon-Coles** — corrige la dependencia en marcadores bajos (0-0, 1-0, 0-1, 1-1).
#
# Los Goles Esperados (xG) de los modelos 1 y 2 se promedian, se construye una **matriz de
# probabilidades 8x8** y se genera un **dashboard de visualización**.
#
# > **Escenario de prueba final:** `Mexico` (local) vs `Czech Republic` (visitante), cancha **neutral**.

# %% [markdown]
# ## 1. Entorno y librerías
#
# En Google Colab / entorno limpio, descomenta la celda de instalación una sola vez.

# %%
# --- Instalación de dependencias (descomentar en Colab / entorno nuevo) ---
# !pip install -q "pymc>=5.10" "xgboost>=2.0" arviz pandas numpy matplotlib seaborn scipy

# %%
import warnings
warnings.filterwarnings("ignore")  # silencia avisos de convergencia/depreciación no críticos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import poisson
from scipy.optimize import minimize

# Las dos librerías pesadas se importan con manejo de errores explícito,
# para dar un mensaje claro si faltan en el entorno.
try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "Falta 'xgboost'. Instala con: pip install xgboost"
    ) from e

try:
    import pymc as pm
    import arviz as az
except ImportError as e:
    raise ImportError(
        "Falta 'pymc'/'arviz'. Instala con: pip install 'pymc>=5.10' arviz"
    ) from e

# Estilo de gráficos limpio
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

print("Versiones ->",
      f"pandas {pd.__version__} | numpy {np.__version__} |",
      f"pymc {pm.__version__} | xgboost {xgb.__version__}")

# %% [markdown]
# ## 2. Ingesta y preprocesamiento de datos
#
# Fuente: dataset público `martj42/international_results`. Se filtra de **2018-01-01 a hoy**,
# se conservan las columnas relevantes, se eliminan nulos y se fuerzan los goles a enteros.

# %%
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FECHA_INICIO = "2018-01-01"


def cargar_datos(url: str = DATA_URL, fecha_inicio: str = FECHA_INICIO) -> pd.DataFrame:
    """Descarga, limpia y filtra el histórico de partidos internacionales.

    Parameters
    ----------
    url : str
        Ruta CSV cruda del dataset.
    fecha_inicio : str
        Fecha mínima (inclusive) a conservar, formato 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Columnas: date, home_team, away_team, home_score, away_score, neutral.
        Ordenado cronológicamente, sin nulos, goles enteros.
    """
    try:
        df = pd.read_csv(url)
    except Exception as e:  # red, URL caída, parseo, etc.
        raise RuntimeError(f"No se pudo cargar el dataset desde {url}: {e}") from e

    # --- Tipos y filtro temporal ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    hoy = pd.Timestamp.today().normalize()
    mask = (df["date"] >= pd.Timestamp(fecha_inicio)) & (df["date"] <= hoy)
    df = df.loc[mask].copy()

    # --- Selección de variables y limpieza ---
    cols = ["date", "home_team", "away_team", "home_score", "away_score", "neutral"]
    faltantes = [c for c in cols if c not in df.columns]
    if faltantes:
        raise KeyError(f"El dataset no contiene las columnas esperadas: {faltantes}")

    df = df[cols].dropna(subset=cols)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError("El dataset quedó vacío tras el filtrado. Revisa 'fecha_inicio'.")

    print(f"Partidos cargados: {len(df):,} | rango: "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df

# %% [markdown]
# ## 3. Ingeniería de características (features para XGBoost)
#
# Para cada equipo se calculan, **usando únicamente partidos pasados** (evita fuga de datos):
#
# - Promedio de goles anotados / recibidos en los últimos **5** y **10** partidos.
# - Porcentaje de victorias en los últimos **10** partidos.
# - Días de descanso desde el último partido.
#
# Estrategia: se transforma a formato *largo* (una fila por equipo y partido), se calculan
# medias móviles con `shift(1)` (excluyen el partido en curso) y se reincorporan al formato
# de partido como features `home_*` / `away_*`.

# %%
VENTANAS = (5, 10)


def _a_formato_largo(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el DataFrame de partidos a formato largo (una fila por equipo)."""
    df = df.reset_index(drop=True).copy()
    df["match_id"] = df.index

    local = df[["match_id", "date", "home_team", "away_team",
                "home_score", "away_score", "neutral"]].copy()
    local.columns = ["match_id", "date", "team", "opponent",
                     "goals_for", "goals_against", "neutral"]
    local["side"] = "home"

    visit = df[["match_id", "date", "away_team", "home_team",
                "away_score", "home_score", "neutral"]].copy()
    visit.columns = ["match_id", "date", "team", "opponent",
                     "goals_for", "goals_against", "neutral"]
    visit["side"] = "away"

    largo = pd.concat([local, visit], ignore_index=True)
    largo["win"] = (largo["goals_for"] > largo["goals_against"]).astype(int)
    return largo.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def construir_features(df: pd.DataFrame, ventanas=VENTANAS):
    """Enriquece el dataset con features históricas por equipo.

    Returns
    -------
    df_feat : pd.DataFrame
        Una fila por partido con columnas home_* y away_* para cada feature.
    largo : pd.DataFrame
        Formato largo (usado luego para snapshot de stats al predecir).
    feature_cols : list[str]
        Orden canónico de columnas de entrada para XGBoost.
    """
    largo = _a_formato_largo(df)
    g = largo.groupby("team", sort=False)

    # Medias móviles SOBRE EL PASADO: shift(1) descarta el partido en curso.
    for w in ventanas:
        largo[f"gf_avg_{w}"] = g["goals_for"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        largo[f"ga_avg_{w}"] = g["goals_against"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    largo["winrate_10"] = g["win"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    # Días de descanso desde el partido anterior del mismo equipo.
    largo["rest_days"] = g["date"].transform(lambda s: s.diff().dt.days)
    rest_mediana = largo["rest_days"].median()
    largo["rest_days"] = largo["rest_days"].fillna(rest_mediana).clip(upper=365)

    # --- Reincorporar al formato de partido ---
    base_cols = [f"gf_avg_{w}" for w in ventanas] + \
                [f"ga_avg_{w}" for w in ventanas] + ["winrate_10", "rest_days"]

    home_feats = largo[largo.side == "home"].set_index("match_id")[base_cols]
    away_feats = largo[largo.side == "away"].set_index("match_id")[base_cols]

    df_feat = df.reset_index(drop=True).copy()
    df_feat["match_id"] = df_feat.index
    df_feat = df_feat.set_index("match_id")

    feature_cols = []
    for c in base_cols:
        df_feat[f"home_{c}"] = home_feats[c]
        df_feat[f"away_{c}"] = away_feats[c]
        feature_cols.extend([f"home_{c}", f"away_{c}"])

    df_feat["neutral_int"] = df_feat["neutral"].astype(int)
    feature_cols.append("neutral_int")

    df_feat = df_feat.reset_index(drop=True)
    print(f"Features construidas: {len(feature_cols)} columnas predictoras.")
    return df_feat, largo, feature_cols


def construir_snapshot(largo: pd.DataFrame, ventanas=VENTANAS) -> dict:
    """Stats 'al día' por equipo (incluyen el último partido) para predecir un encuentro futuro.

    A diferencia de las features de entrenamiento (que excluyen el partido en curso),
    aquí se incluye todo el historial porque el partido a predecir aún no se ha jugado.
    """
    snapshot = {}
    for team, grp in largo.sort_values("date").groupby("team", sort=False):
        d = {}
        for w in ventanas:
            d[f"gf_avg_{w}"] = grp["goals_for"].tail(w).mean()
            d[f"ga_avg_{w}"] = grp["goals_against"].tail(w).mean()
        d["winrate_10"] = grp["win"].tail(10).mean()
        d["last_date"] = grp["date"].max()
        snapshot[team] = d
    return snapshot


def features_partido(snapshot: dict, home: str, away: str, neutral: bool,
                     feature_cols, ventanas=VENTANAS, hoy=None) -> pd.DataFrame:
    """Construye la fila de features (1xN) para un partido a predecir con XGBoost."""
    if home not in snapshot:
        raise ValueError(f"Sin historial para el equipo local '{home}'.")
    if away not in snapshot:
        raise ValueError(f"Sin historial para el equipo visitante '{away}'.")

    hoy = pd.Timestamp.today().normalize() if hoy is None else pd.Timestamp(hoy)
    h, a = snapshot[home], snapshot[away]

    fila = {}
    for w in ventanas:
        fila[f"home_gf_avg_{w}"] = h[f"gf_avg_{w}"]
        fila[f"away_gf_avg_{w}"] = a[f"gf_avg_{w}"]
        fila[f"home_ga_avg_{w}"] = h[f"ga_avg_{w}"]
        fila[f"away_ga_avg_{w}"] = a[f"ga_avg_{w}"]
    fila["home_winrate_10"] = h["winrate_10"]
    fila["away_winrate_10"] = a["winrate_10"]
    fila["home_rest_days"] = min((hoy - h["last_date"]).days, 365)
    fila["away_rest_days"] = min((hoy - a["last_date"]).days, 365)
    fila["neutral_int"] = int(neutral)

    # Reordenar EXACTAMENTE como en el entrenamiento.
    return pd.DataFrame([fila])[feature_cols]

# %% [markdown]
# ## 4. Modelo 1 — MCMC bayesiano jerárquico (PyMC)
#
# Modelo log-lineal de Poisson:
#
# $$\log \lambda_{home} = \mu + \gamma\,(1-\text{neutral}) + \text{att}_{home} - \text{def}_{away}$$
# $$\log \lambda_{away} = \mu + \text{att}_{away} - \text{def}_{home}$$
#
# - $\mu$ = intercepto (intercept), $\gamma$ = ventaja de localía (home advantage).
# - `att` / `def` = fuerza de ataque/defensa por selección, con priors Normales jerárquicos.
# - Se centran att/def (suma cero) para garantizar identificabilidad frente al intercepto.

# %%
def construir_indices(df: pd.DataFrame):
    """Mapa equipo -> índice entero (incluye locales y visitantes)."""
    equipos = sorted(set(df["home_team"]) | set(df["away_team"]))
    return equipos, {t: i for i, t in enumerate(equipos)}


def modelo_bayesiano(df: pd.DataFrame, draws: int = 1000, tune: int = 1000,
                     chains: int = 2, target_accept: float = 0.9, seed: int = 42):
    """Ajusta el modelo jerárquico de Poisson vía MCMC (NUTS).

    Returns
    -------
    modelo : pm.Model
    trace : arviz.InferenceData
    idx : dict  (equipo -> índice)
    """
    equipos, idx = construir_indices(df)
    n_eq = len(equipos)

    home_idx = df["home_team"].map(idx).to_numpy()
    away_idx = df["away_team"].map(idx).to_numpy()
    neutral = df["neutral"].astype(int).to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()

    with pm.Model() as modelo:
        # --- Priors ---
        intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
        home_adv = pm.Normal("home_adv", mu=0.0, sigma=1.0)

        sd_att = pm.HalfNormal("sd_att", sigma=1.0)   # hiperprior de dispersión
        sd_def = pm.HalfNormal("sd_def", sigma=1.0)
        attack = pm.Normal("attack", mu=0.0, sigma=sd_att, shape=n_eq)
        defense = pm.Normal("defense", mu=0.0, sigma=sd_def, shape=n_eq)

        # Restricción suma-cero (identificabilidad) como Deterministic reutilizable.
        attack_c = pm.Deterministic("attack_c", attack - pm.math.mean(attack))
        defense_c = pm.Deterministic("defense_c", defense - pm.math.mean(defense))

        # --- Tasas esperadas (lambda) ---
        log_lh = intercept + home_adv * (1 - neutral) + attack_c[home_idx] - defense_c[away_idx]
        log_la = intercept + attack_c[away_idx] - defense_c[home_idx]

        # --- Verosimilitud Poisson ---
        pm.Poisson("home_goals", mu=pm.math.exp(log_lh), observed=hs)
        pm.Poisson("away_goals", mu=pm.math.exp(log_la), observed=as_)

        # --- Inferencia ---
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=True)

    return modelo, trace, idx


def _params_posteriores(trace):
    """Extrae medias posteriores de los parámetros (centrados)."""
    post = trace.posterior
    att = post["attack_c"].mean(dim=["chain", "draw"]).to_numpy()
    deff = post["defense_c"].mean(dim=["chain", "draw"]).to_numpy()
    intercept = float(post["intercept"].mean())
    home_adv = float(post["home_adv"].mean())
    return att, deff, intercept, home_adv


def xg_bayesiano(trace, idx, home: str, away: str, neutral: bool = False):
    """xG (local, visitante) a partir de las medias posteriores."""
    if home not in idx or away not in idx:
        raise ValueError("Equipo sin parámetros en la traza (no aparece en el histórico).")
    att, deff, intercept, home_adv = _params_posteriores(trace)
    hi, ai = idx[home], idx[away]
    log_lh = intercept + home_adv * (0 if neutral else 1) + att[hi] - deff[ai]
    log_la = intercept + att[ai] - deff[hi]
    return float(np.exp(log_lh)), float(np.exp(log_la))


def lambdas_historicos(trace, idx, df: pd.DataFrame):
    """Tasas ajustadas (lambda_home, lambda_away) para CADA partido del histórico.

    Útil para estimar el parámetro rho de Dixon-Coles por máxima verosimilitud.
    """
    att, deff, intercept, home_adv = _params_posteriores(trace)
    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    neutral = df["neutral"].astype(int).to_numpy()
    lh = np.exp(intercept + home_adv * (1 - neutral) + att[hi] - deff[ai])
    la = np.exp(intercept + att[ai] - deff[hi])
    return lh, la

# %% [markdown]
# ## 5. Modelo 2 — Regresión con XGBoost (`count:poisson`)
#
# Dos regresores independientes (goles local / goles visitante) con regularización para
# evitar overfitting (`max_depth=4`, `learning_rate=0.05`, submuestreo).

# %%
def entrenar_xgboost(df_feat: pd.DataFrame, feature_cols, seed: int = 42):
    """Entrena dos XGBRegressor (local y visitante) con objetivo Poisson.

    Returns
    -------
    modelo_home, modelo_away : xgb.XGBRegressor
    """
    df_t = df_feat.dropna(subset=feature_cols).copy()
    if df_t.empty:
        raise ValueError("No hay filas válidas para entrenar XGBoost tras eliminar nulos.")

    X = df_t[feature_cols]
    y_home = df_t["home_score"]
    y_away = df_t["away_score"]

    params = dict(
        objective="count:poisson",   # predicción de conteos (goles)
        n_estimators=400,
        max_depth=4,                 # árboles poco profundos -> menos overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )

    modelo_home = xgb.XGBRegressor(**params).fit(X, y_home)
    modelo_away = xgb.XGBRegressor(**params).fit(X, y_away)
    print(f"XGBoost entrenado sobre {len(df_t):,} partidos.")
    return modelo_home, modelo_away


def xg_xgboost(modelo_home, modelo_away, snapshot, home, away, neutral, feature_cols):
    """xG (local, visitante) predichos por los árboles para un partido concreto."""
    X = features_partido(snapshot, home, away, neutral, feature_cols)
    lh = float(modelo_home.predict(X)[0])
    la = float(modelo_away.predict(X)[0])
    return lh, la

# %% [markdown]
# ## 6. Modelo 3 — Ajuste Dixon-Coles (complementario)
#
# La asunción de independencia estricta de Poisson subestima los marcadores bajos. El factor
# $\tau$ de Dixon-Coles corrige (0-0), (1-0), (0-1) y (1-1):
#
# $$P(x,y) = \tau_{\lambda,\mu}(x,y)\cdot \text{Poisson}(x;\lambda)\cdot \text{Poisson}(y;\mu)$$

# %%
def tau_dixon_coles(x, y, lh, la, rho):
    """Factor de corrección de dependencia para marcadores bajos."""
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    elif x == 0 and y == 1:
        return 1.0 + lh * rho
    elif x == 1 and y == 0:
        return 1.0 + la * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _neg_loglik_rho(rho, lh, la, x, y):
    """Log-verosimilitud negativa (vectorizada) del ajuste DC en función de rho."""
    tau = np.ones_like(lh, dtype=float)
    m00 = (x == 0) & (y == 0); tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    m01 = (x == 0) & (y == 1); tau[m01] = 1.0 + lh[m01] * rho
    m10 = (x == 1) & (y == 0); tau[m10] = 1.0 + la[m10] * rho
    m11 = (x == 1) & (y == 1); tau[m11] = 1.0 - rho

    tau = np.clip(tau, 1e-9, None)  # evita log de valores no positivos
    p = tau * poisson.pmf(x, lh) * poisson.pmf(y, la)
    p = np.clip(p, 1e-12, None)
    return -np.sum(np.log(p))


def estimar_rho(lh, la, x, y, rho0: float = -0.05):
    """Estima rho por máxima verosimilitud sobre el histórico (scipy.optimize.minimize)."""
    try:
        res = minimize(lambda r: _neg_loglik_rho(r[0], lh, la, x, y),
                       x0=[rho0], bounds=[(-0.2, 0.2)], method="L-BFGS-B")
        rho = float(res.x[0]) if res.success else rho0
    except Exception as e:
        print(f"[aviso] Falló la estimación de rho ({e}); se usa {rho0}.")
        rho = rho0
    print(f"Rho (Dixon-Coles) estimado: {rho:+.4f}")
    return rho

# %% [markdown]
# ## 7. Matriz de probabilidades 8x8
#
# Combina los xG finales y aplica la corrección Dixon-Coles. Filas = goles del **local**
# (0–7), columnas = goles del **visitante** (0–7). Se normaliza para que sume ≈ 1.0.

# %%
def matriz_probabilidades(lh: float, la: float, rho: float = 0.0, max_goles: int = 7):
    """Matriz NumPy (max_goles+1) x (max_goles+1) de probabilidades por marcador exacto."""
    if lh <= 0 or la <= 0:
        raise ValueError("Los xG deben ser positivos.")
    size = max_goles + 1
    M = np.zeros((size, size))
    px = poisson.pmf(np.arange(size), lh)  # P(goles local)
    py = poisson.pmf(np.arange(size), la)  # P(goles visitante)
    for x in range(size):
        for y in range(size):
            M[x, y] = tau_dixon_coles(x, y, lh, la, rho) * px[x] * py[y]
    M = M / M.sum()  # normaliza (la truncación a 7 y tau alteran la masa total)
    assert abs(M.sum() - 1.0) < 1e-9, "La matriz no suma 1.0"
    return M


def resumen_resultados(M):
    """Probabilidades agregadas: gana local / empate / gana visitante."""
    p_local = np.tril(M, -1).sum()   # x > y  (triángulo inferior)
    p_empate = np.trace(M)           # x == y (diagonal)
    p_visit = np.triu(M, 1).sum()    # x < y  (triángulo superior)
    return p_local, p_empate, p_visit

# %% [markdown]
# ## 8. Visualizaciones avanzadas (dashboard)
#
# Tres paneles: (1) heatmap de la matriz, (2) probabilidad de resultado, (3) top-10 marcadores.

# %%
def _abrev(nombre: str) -> str:
    """Abreviatura de 3 letras en mayúsculas (p. ej. 'Mexico' -> 'MEX')."""
    return nombre[:3].upper()


def top_marcadores(M, home, away, n: int = 10):
    """Lista descendente de los n marcadores exactos más probables, ya formateados."""
    ha, aa = _abrev(home), _abrev(away)
    items = [(f"{ha} {x}-{y} {aa}", M[x, y])
             for x in range(M.shape[0]) for y in range(M.shape[1])]
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:n]


def dashboard(M, home: str, away: str, lh: float, la: float):
    """Genera el panel de 3 gráficos con matplotlib/seaborn."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Heatmap de marcadores (%) ---
    sns.heatmap(M * 100, annot=True, fmt=".1f", cmap="YlGnBu",
                cbar_kws={"label": "Probabilidad (%)"}, ax=axes[0],
                linewidths=.5, linecolor="white")
    axes[0].set_title(f"Matriz de marcadores\n{home} (xG {lh:.2f}) vs {away} (xG {la:.2f})",
                      fontweight="bold")
    axes[0].set_xlabel(f"Goles {away} (visitante)")
    axes[0].set_ylabel(f"Goles {home} (local)")

    # --- Panel 2: Probabilidad de resultado ---
    p_local, p_empate, p_visit = resumen_resultados(M)
    etiquetas = [f"Gana {_abrev(home)}", "Empate", f"Gana {_abrev(away)}"]
    valores = np.array([p_local, p_empate, p_visit]) * 100
    colores = ["#2a9d8f", "#e9c46a", "#e76f51"]
    barras = axes[1].bar(etiquetas, valores, color=colores, edgecolor="black")
    axes[1].set_title("Probabilidad de resultado (1X2)", fontweight="bold")
    axes[1].set_ylabel("Probabilidad (%)")
    axes[1].set_ylim(0, max(valores) * 1.2)
    for b, v in zip(barras, valores):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%",
                     ha="center", va="bottom", fontweight="bold")

    # --- Panel 3: Top-10 marcadores exactos ---
    top = top_marcadores(M, home, away, n=10)
    labels = [t[0] for t in top][::-1]          # invertido: mayor arriba
    probs = [t[1] * 100 for t in top][::-1]
    axes[2].barh(labels, probs, color="#264653", edgecolor="black")
    axes[2].set_title("Top 10 marcadores más probables", fontweight="bold")
    axes[2].set_xlabel("Probabilidad (%)")
    for i, v in enumerate(probs):
        axes[2].text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)

    fig.suptitle("⚽ GoalOracle — Dashboard de predicción", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig

# %% [markdown]
# ## 9. Pipeline de ejecución y prueba
#
# Orquesta todo el flujo: datos → features → modelos → combinación de xG → matriz → dashboard.

# %%
def ejecutar_pipeline(home: str = "Mexico", away: str = "Czech Republic", neutral: bool = True,
                      draws: int = 1000, chains: int = 2):
    """Ejecuta el pipeline completo y devuelve un diccionario con todos los artefactos."""
    print("=" * 70)
    print(f"GoalOracle | {home} vs {away} | cancha neutral = {neutral}")
    print("=" * 70)

    # 1-2) Datos
    df = cargar_datos()

    # 3) Features
    df_feat, largo, feature_cols = construir_features(df)
    snapshot = construir_snapshot(largo)

    # 4) Modelo bayesiano
    print("\n[1/3] Entrenando modelo bayesiano (MCMC)...")
    _, trace, idx = modelo_bayesiano(df, draws=draws, chains=chains)
    lh_bayes, la_bayes = xg_bayesiano(trace, idx, home, away, neutral)
    print(f"   xG Bayesiano  -> {home}: {lh_bayes:.3f} | {away}: {la_bayes:.3f}")

    # 5) Modelo XGBoost
    print("\n[2/3] Entrenando XGBoost...")
    m_home, m_away = entrenar_xgboost(df_feat, feature_cols)
    lh_xgb, la_xgb = xg_xgboost(m_home, m_away, snapshot, home, away, neutral, feature_cols)
    print(f"   xG XGBoost    -> {home}: {lh_xgb:.3f} | {away}: {la_xgb:.3f}")

    # Combinación (promedio de ambos modelos)
    lh = (lh_bayes + lh_xgb) / 2.0
    la = (la_bayes + la_xgb) / 2.0
    print(f"\n   xG combinado  -> {home}: {lh:.3f} | {away}: {la:.3f}")

    # 6) Dixon-Coles: estimar rho con las tasas históricas del modelo bayesiano
    print("\n[3/3] Estimando ajuste Dixon-Coles (rho)...")
    lh_hist, la_hist = lambdas_historicos(trace, idx, df)
    rho = estimar_rho(lh_hist, la_hist,
                      df["home_score"].to_numpy(), df["away_score"].to_numpy())

    # 7) Matriz de probabilidades
    M = matriz_probabilidades(lh, la, rho=rho, max_goles=7)
    p_local, p_empate, p_visit = resumen_resultados(M)
    print(f"\nResultado 1X2 -> Gana {home}: {p_local:.1%} | "
          f"Empate: {p_empate:.1%} | Gana {away}: {p_visit:.1%}")
    print("Marcador más probable:", top_marcadores(M, home, away, 1)[0][0],
          f"({top_marcadores(M, home, away, 1)[0][1]:.1%})")

    # 8) Dashboard
    dashboard(M, home, away, lh, la)

    return {
        "df": df, "df_feat": df_feat, "snapshot": snapshot,
        "trace": trace, "idx": idx, "xgb_home": m_home, "xgb_away": m_away,
        "xg": (lh, la), "rho": rho, "matriz": M,
        "resultado": {"local": p_local, "empate": p_empate, "visitante": p_visit},
    }


if __name__ == "__main__":
    # Escenario: Mexico (local) vs Czech Republic (visitante), cancha neutral.
    # Si quieres que Mexico juegue con ventaja de localia real, usa neutral=False.
    resultados = ejecutar_pipeline(home="Mexico", away="Czech Republic", neutral=True)
