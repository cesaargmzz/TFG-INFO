<div align="center">

<img src="assets/banner.svg" alt="GDPulse Banner" width="100%"/>

<br/>

<p>
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=400&size=16&duration=3500&pause=800&color=94A3B8&center=true&vCenter=true&width=700&height=35&lines=Modelos+base+%E2%86%92+Panel+combinado+%E2%86%92+Geo+features+%E2%86%92+Transferencia;MAE+m%C3%ADnimo+0.193+p.p.+en+Francia+%C2%B7+XGBoost+ext;8+trimestres+de+test%3A+2023Q4+%E2%80%93+2025Q3" alt="Subtítulo rotatorio" />
</p>

<br/>

<!-- Badges -->
<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/XGBoost-2.x-FF6600?style=for-the-badge&logo=xgboost&logoColor=white" />
<img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
<img src="https://img.shields.io/badge/FastAPI-0.11x-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Three.js-WebGL-black?style=for-the-badge&logo=threedotjs&logoColor=white" />

<br/><br/>

<img src="https://img.shields.io/badge/Estado-Completado-22c55e?style=for-the-badge" />
<img src="https://img.shields.io/badge/Licencia-MIT-a78bfa?style=for-the-badge" />

</div>

---

## ¿Qué es GDPulse?

**GDPulse** es un proyecto de análisis comparativo de modelos de aprendizaje automático para la estimación del crecimiento trimestral del PIB (QoQ %) en tres economías de la zona euro: **España**, **Francia** e **Italia**.

A diferencia de los modelos econométricos clásicos (ARIMA, OLS), los modelos basados en **gradient boosting** capturan relaciones no lineales entre indicadores macroeconómicos y el ciclo económico — incluso en periodos de crisis como 2008 o COVID-2020.

---

## Resultados experimentales

Se compararon **4 bloques de experimentos** con distintas estrategias de entrenamiento:

| Bloque | Estrategia | Mejor MAE (ES) | Mejor MAE (FR) | Mejor MAE (IT) |
|--------|-----------|:--------------:|:--------------:|:--------------:|
| **B0** — Por país | Modelos independientes | 0.314 | 0.193 | 0.344 |
| **B1** — Panel combinado | ES + FR + IT juntos | 0.297 | 0.215 | 0.376 |
| **B2** — Panel con geo | B1 + dummies de país | 0.290 | 0.224 | 0.365 |
| **B3** — Transferencia | Entrenado en 2, evaluado en 1 | 0.337 | 0.220 | **0.311** |

> 📌 El error se mide en **puntos porcentuales** de variación trimestral del PIB.
> El mejor resultado para Italia (MAE = 0.311) se obtiene con transferencia ES+FR → IT.

---

## Arquitectura del pipeline

```
Eurostat API
    │
    ▼
data/raw/*.parquet          ← fetch_eurostat_*.py
    │
    ▼
data/processed/*.parquet    ← transform_*.py  (QoQ %, lags)
    │
    ▼
dataset_panel_v1.parquet    ← build_dataset_panel.py
    │
    ▼
reports/*.csv               ← train_xgboost_panel*.py
    │
    ├── Streamlit app        ← src/app/streamlit/app.py
    └── GDPulse (React)      ← src/app/web/
```

---

## Stack tecnológico

<table>
<tr>
<td valign="top" width="50%">

### 🔬 ML Pipeline
- **Pandas** — transformación de datos
- **XGBoost** — modelo principal
- **Scikit-learn** — métricas y splits
- **Eurostat REST API** — fuente de datos

</td>
<td valign="top" width="50%">

### 🌐 Frontend (GDPulse)
- **React 18 + Vite** — SPA
- **Three.js** — globo terráqueo 3D
- **FastAPI** — backend API
- **Recharts** — visualización

</td>
</tr>
</table>

---

## Ejecutar el proyecto

### Pipeline de datos y modelos

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar datos de Eurostat
cd src/sources
python fetch_eurostat_gdp.py --geos ES IT FR

# 3. Transformar y construir dataset
cd ../transform && python transform_gdp_qoq_panel.py
cd ../build    && python build_dataset_panel.py

# 4. Entrenar modelos
cd ../models
python train_xgboost_panel.py --geo ES
python train_xgboost_panel.py --geo FR
python train_xgboost_panel.py --geo IT
```

### App Streamlit (legacy)

```bash
streamlit run src/app/streamlit/app.py
```

### GDPulse (React + FastAPI)

```bash
# Backend
cd src/app/web/backend
uvicorn main:app --reload --port 8000

# Frontend
cd src/app/web/frontend
npm install && npm run dev
```

Abre [http://localhost:5173](http://localhost:5173) 🚀

---

## Estructura del repositorio

```
TFG_INFO/
├── data/
│   ├── raw/              # Parquets de Eurostat
│   └── processed/        # Dataset transformado con lags
├── src/
│   ├── sources/          # fetch_eurostat_*.py
│   ├── transform/        # transform_*.py
│   ├── build/            # build_dataset*.py
│   ├── models/           # train_xgboost_panel*.py
│   └── app/
│       ├── streamlit/    # Dashboard Streamlit
│       └── web/          # GDPulse (React + FastAPI)
├── reports/
│   ├── figures/          # Gráficas PNG
│   └── *.csv             # Métricas y predicciones
├── memoria/              # LaTeX — TFG completo
└── requirements.txt
```

---

## Variables e indicadores

| Indicador | Variable | Fuente |
|-----------|----------|--------|
| PIB trimestral | `gdp_qoq_pct` | Eurostat `namq_10_gdp` |
| Tasa de desempleo | `unemployment_rate_l1` | Eurostat `une_rt_q` |
| Inflación QoQ | `inflation_qoq_pct_l1` | Eurostat `prc_hicp_midx` |
| Producción industrial | `ipi_qoq_pct_l1` | Eurostat `sts_inpr_q` |
| Ventas al por menor | `retail_qoq_pct_l1` | Eurostat `sts_trtu_q` |

> Todos los features usan lag `_l1` o `_l2` para evitar data leakage.

---

<div align="center">

<br/>

*TFG — Análisis comparativo de modelos de aprendizaje automático para la estimación del crecimiento del PIB en la zona euro*

**Universidad · 2026**

</div>
