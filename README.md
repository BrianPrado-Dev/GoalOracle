# ⚽ GoalOracle

**Sistema avanzado de predicción de marcadores de fútbol internacional** que combina tres
enfoques de modelado para estimar el resultado de un partido entre dos selecciones.

A partir del histórico público de partidos internacionales (desde 2018), GoalOracle calcula
los **Goles Esperados (xG)** de cada equipo, construye una **matriz de probabilidades de
marcador 8×8** y genera un **dashboard visual** con la probabilidad de victoria/empate/derrota
y los marcadores más probables.

---

## 🧠 ¿Cómo funciona?

GoalOracle integra **tres modelos complementarios**:

| # | Modelo | Qué aporta |
|---|--------|------------|
| 1 | **MCMC Bayesiano jerárquico** (PyMC v5) | Estima fuerza de ataque/defensa por selección + ventaja de localía, con verosimilitud de Poisson. |
| 2 | **Regresión XGBoost** (`objective="count:poisson"`) | Aprende patrones de goles a partir de features históricas (forma reciente, descanso, etc.). |
| 3 | **Ajuste Dixon-Coles** | Corrige la dependencia en marcadores bajos (0-0, 1-0, 0-1, 1-1), donde Poisson puro falla. |

Los xG de los modelos 1 y 2 se **promedian**, se aplica la corrección de Dixon-Coles y se
genera la matriz de probabilidades final.

---

## 📂 Estructura del proyecto

```
GoalOracle/
├── GoalOracle_Prediccion.ipynb   # Notebook listo para Jupyter / Google Colab
├── goaloracle_notebook.py        # Mismo contenido en formato script (celdas '# %%')
├── _build_ipynb.py               # Conversor .py -> .ipynb (regenera el notebook)
├── requirements.txt              # Dependencias para instalación local
└── README.md
```

> El `.ipynb` y el `.py` contienen **el mismo código**. Edita el `.py` y regenera el
> notebook con `python _build_ipynb.py`, o trabaja directamente sobre el `.ipynb`.

---

## 🚀 Uso con Google Colab (recomendado, sin instalar nada)

1. Abre [Google Colab](https://colab.research.google.com/).
2. `Archivo → Subir notebook` y selecciona **`GoalOracle_Prediccion.ipynb`**.
3. Ejecuta la **primera celda de instalación** (descomenta la línea `!pip install ...`):
   ```python
   !pip install -q "pymc>=5.10" "xgboost>=2.0" arviz pandas numpy matplotlib seaborn scipy
   ```
4. Menú `Entorno de ejecución → Ejecutar todo`.

La última celda corre el escenario de prueba y muestra el dashboard.

> 💡 **Tip:** activa la GPU/CPU alta en `Entorno de ejecución → Cambiar tipo de entorno`
> si el muestreo MCMC va lento.

---

## 💻 Uso local

### Requisitos
- **Python 3.10 – 3.12** (probado en 3.11).
- Las dependencias de [`requirements.txt`](requirements.txt).

### Instalación

```bash
# 1) Clona o descarga el proyecto y entra en la carpeta
cd GoalOracle

# 2) (Recomendado) crea un entorno virtual
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3) Instala las dependencias
pip install -r requirements.txt
```

### Ejecución

**Opción A — como script** (corre el pipeline completo y abre el dashboard):

```bash
python goaloracle_notebook.py
```

**Opción B — como notebook en Jupyter:**

```bash
pip install jupyter
jupyter notebook GoalOracle_Prediccion.ipynb
```

**Opción C — en VS Code:** abre `goaloracle_notebook.py`; los marcadores `# %%` se
reconocen como celdas interactivas ("Run Cell").

---

## 🎯 Cómo cambiar el partido a predecir

La función principal es `ejecutar_pipeline`. Modifica la última celda / el bloque
`if __name__ == "__main__":`:

```python
resultados = ejecutar_pipeline(
    home="Mexico",            # equipo local
    away="Czech Republic",    # equipo visitante
    neutral=True,             # True = cancha neutral (sin ventaja de localía)
    draws=1000,               # nº de muestras MCMC por cadena
    chains=2,                 # nº de cadenas MCMC
)
```

- **`neutral=True`** → el equipo `home` es el local *nominal* (aparece como local en la
  matriz), pero **no** recibe la ventaja de localía del modelo bayesiano.
- **`neutral=False`** → el equipo `home` juega con ventaja de localía real.
- Para una corrida rápida de prueba, baja `draws` (p. ej. `draws=500`).

### ¿No sabes el nombre exacto de un equipo?

El dataset usa nombres en inglés (p. ej. `"Czech Republic"`, no "Czechia"). Para listarlos:

```python
df = cargar_datos()
equipos = sorted(set(df.home_team) | set(df.away_team))
print([t for t in equipos if "Czech" in t])   # filtra por subcadena
```

---

## 📊 Salida

El pipeline imprime los xG de cada modelo y genera un **dashboard de 3 paneles**:

1. **Heatmap 8×8** — probabilidad (%) de cada marcador exacto (0–7 goles por equipo).
2. **Resultado 1X2** — probabilidad de que gane el local, empate o gane el visitante.
3. **Top 10 marcadores** — los resultados exactos más probables (ej. `MEX 2-0 CZE`).

Además, `ejecutar_pipeline` devuelve un diccionario con todos los artefactos
(`df`, `trace`, modelos XGBoost, `matriz`, `xg`, `rho`, probabilidades `resultado`, etc.)
para análisis posteriores.

---

## 🔬 Detalles técnicos

- **Fuente de datos:** [`martj42/international_results`](https://github.com/martj42/international_results)
  (CSV crudo de GitHub). Se filtra de **2018-01-01 hasta hoy**.
- **Sin fuga de datos:** las features de XGBoost se calculan con `shift(1)` (solo usan
  partidos *anteriores* a cada encuentro). Para predecir un partido futuro se usa un
  *snapshot* con todo el historial disponible.
- **Features:** promedio de goles a favor/en contra (últimos 5 y 10 partidos), % de
  victorias (últimos 10) y días de descanso.
- **Identificabilidad bayesiana:** los parámetros de ataque/defensa se centran (suma cero)
  frente al intercepto.
- **Rho (Dixon-Coles):** se estima por máxima verosimilitud sobre el histórico con
  `scipy.optimize.minimize`.

---

## 🛠️ Solución de problemas

| Problema | Solución |
|----------|----------|
| `Sin historial para el equipo '...'` | El nombre no coincide o el equipo no tiene partidos desde 2018. Lista los nombres exactos (ver arriba). |
| El muestreo MCMC es muy lento | Reduce `draws` (p. ej. 500) y/o `chains`. En Colab, usa un entorno con más CPU. |
| `No se pudo cargar el dataset` | Revisa tu conexión a internet; la fuente es un CSV remoto de GitHub. |
| Errores al instalar `pymc` en Windows | Usa el entorno virtual y `pip install -r requirements.txt`; o ejecútalo en Google Colab. |

---

## 🧩 Posibles extensiones

- **Backtesting** con métricas tipo *Ranked Probability Score* (RPS) o *log-loss*.
- **Selector interactivo** de equipos con `ipywidgets`.
- Incorporar **ranking FIFA** o variables contextuales (competición, altitud) como features.

---

## ⚠️ Aviso

Proyecto con fines **educativos y de análisis deportivo**. Las predicciones son estimaciones
estadísticas y no garantizan resultados. No está pensado para apuestas.
