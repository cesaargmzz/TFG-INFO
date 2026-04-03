# FRONTEND_PLAN.md
# Plan de desarrollo: Dashboard TFG — PIB QoQ

## Visión general
Frontend standalone (sin tocar Streamlit) que combina una intro cinematográfica
con un dashboard académico de alto nivel visual. Stack: React + Vite + Three.js.
Todo el texto en español.

---

## Stack técnico

| Capa | Tecnología | Motivo |
|------|-----------|--------|
| Backend | FastAPI (Python) | Lee los CSVs de reports/ directamente |
| Frontend | React + Vite | Rápido, ecosistema maduro |
| 3D / Intro | Three.js | Tierra fotorrealista, WebGL |
| Gráficas | Recharts | Declarativo, buena integración React |
| Estilos | CSS custom con variables | Design system de DESIGN.md |
| Tipografía | Manrope (titulares) + Inter (UI) | Ver DESIGN.md |
| HTTP client | fetch nativo | Simple |

---

## Fases de desarrollo

### F0 — Intro: Tierra desde el espacio ✅ COMPLETADA
Pantalla de entrada con globo terráqueo 3D girando lentamente.

**Lo que hay:**
- Fondo negro con campo de estrellas (8000 partículas)
- Esfera 3D (Three.js) con texturas NASA locales en `public/textures/`:
  - `earth.jpg` — superficie Blue Marble
  - `water.png` — specular map para océanos
  - `night.jpg` — luces nocturnas de ciudades (AdditiveBlending)
  - Nubes eliminadas (404 en todas las CDNs disponibles)
- Glow atmosférico azul (dos capas: FrontSide + BackSide)
- Texto izquierdo: "GDPulse" (Manrope 800) + descripción + países
- Transición al dashboard: scroll con rueda → zoom hacia la Tierra
  - `progress` (crudo): controla fade del texto, respuesta inmediata
  - `smoothProgress` (lerp 0.07/frame): controla cámara, FOV y overlay
  - Al llegar al final → fade a negro → dashboard
- Hint de scroll con chevrones animados en la parte inferior

**Decisiones técnicas clave:**
- Todo el scroll gestionado con variables locales dentro del `useEffect` de Three.js
  → cero re-renders de React durante el scroll → fluidez máxima
- La animación CSS de entrada de `.intro__content` solo anima `transform`
  (NO `opacity`) para que JS tenga control exclusivo del fade-out
- `StrictMode` eliminado de `main.jsx` (causaba doble mount del efecto Three.js)
- Texturas servidas desde `public/textures/` (local), no CDN

**Archivos:**
```
src/app/web/frontend/
  public/textures/earth.jpg, water.png, night.jpg
  src/
    main.jsx
    App.jsx
    App.css
    components/
      EarthScene.jsx      ← Three.js, scroll, zoom
      IntroScreen.jsx     ← Layout, refs DOM
      IntroScreen.css     ← Estilos landing
```

---

### F1 — Backend FastAPI ⬜ PENDIENTE
Expone los CSVs de reports/ como endpoints JSON.

**Endpoints:**
```
GET /api/health
GET /api/predictions/{bloque}/{geo}/{modelo}
GET /api/metrics/{bloque}/{geo}
GET /api/summary
```

**Lógica de rutas de CSV:**
```
bloque 0: reports/metrics_xgboost_{geo}_compare.csv
          reports/predictions_xgboost_{base|ext}_{geo}.csv

bloque 1: reports/metrics_panel_combined_compare.csv  (filtrar por geo)
          reports/predictions_panel_combined_{base|ext}_{geo}.csv

bloque 2: reports/metrics_panel_geo_compare.csv  (filtrar por geo)
          reports/predictions_panel_geo_{base|ext}_{geo}.csv

bloque 3: reports/metrics_panel_transfer_compare.csv  (filtrar por geo)
          reports/predictions_panel_transfer_ext_{geo}.csv
          (solo modelo ext)
```

**Arrancar:**
```bash
cd src/app/web/backend
uvicorn main:app --reload --port 8000
```

CORS habilitado para http://localhost:5173

---

### F2 — Esqueleto React ⬜ PENDIENTE
Layout base con sidebar + área principal.

**Sidebar (dark, inverse_surface #2e3132 con gradiente 15°):**
- Select: Bloque experimental
- Select: País (ES / FR / IT)
- Select: Modelo (dinámico según bloque)
- Divider
- Toggle: Mostrar tablas de datos
- Toggle: Mostrar resumen global

**Área principal:**
- Título dinámico: "Dashboard — PIB QoQ · {geo} · {bloque}"
- Scroll vertical

---

### F3 — Design System (variables CSS) ⬜ PENDIENTE
Definir en `design-tokens.css` todas las variables de DESIGN.md:
superficies, primarios, tipografía, espaciado, radios, sombras.
Google Fonts: Manrope + Inter.

Reglas clave:
- Sin bordes 1px sólidos para separar secciones
- Cards: surface-container-lowest (#fff) sobre surface-container (#edeeef)
- Sombras: blur 24px, 4% opacidad, tintadas con primary
- Gráficas: ratio 1:1.618 (áureo), padding spacing-8

---

### F4 — KPI Cards ⬜ PENDIENTE
4 cards en fila: Modelo activo, MAE, RMSE, Nº variables.
Banner informativo en Bloque 3: "Entrenado con X — evaluado en Y"

---

### F5 — Gráficas principales ⬜ PENDIENTE
- LineChart: Real vs Predicción (negro + azul)
- BarChart: Error por trimestre (rojo/azul según signo)

---

### F6 — Gráfica comparativa base vs ampliado ⬜ PENDIENTE
Solo bloques B0, B1, B2. LineChart con 3 series:
Real (negro), Base (azul punteado), Ampliado (azul sólido)

---

### F7 — Tabla resumen global ⬜ PENDIENTE
Pivot MAE modelo ampliado × bloque × país.
Highlight verde (mínimo) / rojo (máximo) por columna.
Solo visible si toggle activo.

---

## Estructura de carpetas objetivo

```
src/app/web/
  backend/
    main.py
    routers/
      metrics.py
      predictions.py
      summary.py
    utils.py
  frontend/
    public/
      textures/           ← earth.jpg, water.png, night.jpg
    src/
      main.jsx
      App.jsx
      App.css
      components/
        EarthScene.jsx       ← Three.js globe + scroll
        IntroScreen.jsx      ← Pantalla de entrada
        IntroScreen.css
        Sidebar.jsx
        KpiCards.jsx
        PredictionChart.jsx
        ErrorChart.jsx
        ComparisonChart.jsx
        SummaryTable.jsx
        BlockDescription.jsx
      hooks/
        useExperimentData.js
      styles/
        design-tokens.css
        index.css
```

---

## Cómo arrancar la siguiente sesión

1. Leer este archivo completo
2. Leer `DESIGN.md` para el design system
3. Leer `src/app/streamlit/app.py` para entender la lógica actual
4. Empezar por **F1 — Backend FastAPI**
5. Verificar con `curl http://localhost:8000/api/health` antes de seguir

**Trabajar una fase a la vez. No avanzar a la siguiente sin verificar la actual.**

---

## Notas

- La app Streamlit NO se toca — conviven en paralelo
- Los CSVs de reports/ son la única fuente de verdad
- El backend solo lee, nunca escribe
- REPORTS_DIR = raíz del repo + /reports
- No implementar entrenamiento desde la web
- Todo texto visible al usuario en español
