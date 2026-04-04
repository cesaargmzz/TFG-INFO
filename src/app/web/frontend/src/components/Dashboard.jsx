import { useState } from 'react'
import Sidebar from './Sidebar'
import KpiCards from './KpiCards'
import PredictionChart from './PredictionChart'
import ErrorChart from './ErrorChart'
import ComparisonChart from './ComparisonChart'
import SummaryTable from './SummaryTable'
import { useExperimentData, useComparisonData, useSummary } from '../hooks/useExperimentData'
import './Dashboard.css'

const GEO_LABEL = { ES: 'España', FR: 'Francia', IT: 'Italia' }
const BLOQUE_LABEL = {
  '0': 'Bloque 0 — Por país',
  '1': 'Bloque 1 — Panel combinado',
  '2': 'Bloque 2 — Panel con geo',
  '3': 'Bloque 3 — Transferencia',
}

export default function Dashboard() {
  const [state, setState] = useState({
    bloque: '0',
    geo: 'ES',
    modelo: 'ext',
    showSummary: false,
  })

  const { predictions, metrics, loading, error } = useExperimentData(
    state.bloque, state.geo, state.modelo
  )
  const { summary } = useSummary()
  const { data: compData } = useComparisonData(state.bloque, state.geo)

  return (
    <div className="dashboard">
      <Sidebar state={state} onChange={setState} />

      <main className="dashboard__main">
        <header className="dashboard__header">
          <div>
            <p className="dashboard__header-label">GDPulse · Análisis de predicciones</p>
            <h1 className="dashboard__header-title">
              {GEO_LABEL[state.geo]}
              <span className="dashboard__header-sep">·</span>
              {BLOQUE_LABEL[state.bloque]}
            </h1>
          </div>
        </header>

        <div className="dashboard__content">
          {error && <p className="dashboard__error">Error al cargar datos: {error}</p>}
          <KpiCards
            bloque={state.bloque}
            geo={state.geo}
            modelo={state.modelo}
            metrics={metrics}
            loading={loading}
          />
          <div className="dashboard__charts">
            <div className="dashboard__charts-row">
              <PredictionChart predictions={predictions} loading={loading} />
              <ErrorChart predictions={predictions} loading={loading} />
            </div>
            <div className={`dashboard__bottom-row${state.showSummary ? ' dashboard__bottom-row--with-summary' : ''}`}>
              <ComparisonChart bloque={state.bloque} geo={state.geo} data={compData} />
              {state.showSummary && <SummaryTable summary={summary} />}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
