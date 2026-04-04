import './KpiCards.css'

const TRANSFER_TRAIN = {
  ES: 'FR + IT',
  FR: 'ES + IT',
  IT: 'ES + FR',
}

export default function KpiCards({ bloque, geo, modelo, metrics: apiMetrics, loading }) {
  // Buscar la fila del modelo seleccionado en los datos de la API
  const row = apiMetrics?.find(m => m.model.toLowerCase().includes(modelo)) ?? null

  const mae    = loading ? '…' : row ? row.mae.toFixed(3)      : '—'
  const rmse   = loading ? '…' : row ? row.rmse.toFixed(3)     : '—'
  const nVars  = loading ? '…' : row ? row.n_features          : '—'
  const modeloLabel = modelo === 'ext' ? 'Ampliado' : 'Base'

  return (
    <div className="kpi-section">
      {bloque === '3' && (
        <div className="kpi-banner">
          <span className="kpi-banner__icon">⟳</span>
          Entrenado con <strong>{TRANSFER_TRAIN[geo]}</strong> — evaluado en <strong>{geo}</strong>
        </div>
      )}

      <div className="kpi-grid" key={`${bloque}-${geo}-${modelo}`}>
        {/* Modelo — lila tint */}
        <div className="kpi-card kpi-card--lila">
          <p className="kpi-card__label">Modelo activo</p>
          <p className="kpi-card__value">{modeloLabel}</p>
          <p className="kpi-card__sub">Bloque {bloque} · {geo}</p>
        </div>

        {/* MAE — oscuro */}
        <div className="kpi-card kpi-card--dark">
          <p className="kpi-card__label">MAE</p>
          <p className="kpi-card__value">{mae}</p>
          <p className="kpi-card__sub">Error absoluto medio</p>
        </div>

        {/* RMSE — oscuro */}
        <div className="kpi-card kpi-card--dark">
          <p className="kpi-card__label">RMSE</p>
          <p className="kpi-card__value">{rmse}</p>
          <p className="kpi-card__sub">Raíz del error cuadrático</p>
        </div>

        {/* Nº variables — azul tint */}
        <div className="kpi-card kpi-card--blue">
          <p className="kpi-card__label">Variables</p>
          <p className="kpi-card__value">{nVars}</p>
          <p className="kpi-card__sub">Features del modelo</p>
        </div>
      </div>
    </div>
  )
}
