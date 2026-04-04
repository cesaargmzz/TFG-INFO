import './SummaryTable.css'

const BLOQUES = ['0', '1', '2', '3']
const BLOQUE_LABEL = {
  '0': 'Bloque 0 — Por país',
  '1': 'Bloque 1 — Panel combinado',
  '2': 'Bloque 2 — Panel con geo',
  '3': 'Bloque 3 — Transferencia',
}
const GEOS = ['ES', 'FR', 'IT']

// Extrae MAE ext por bloque y geo de los datos de la API
function buildTable(summary) {
  const table = {}
  BLOQUES.forEach(b => { table[b] = {} })

  summary?.forEach(row => {
    if (!row.model.toLowerCase().includes('ext')) return
    const b = String(row.bloque)
    const g = row.geo
    if (table[b] && GEOS.includes(g)) table[b][g] = row.mae
  })
  return table
}

export default function SummaryTable({ summary }) {
  const table = buildTable(summary)

  const stats = {}
  GEOS.forEach(geo => {
    const values = BLOQUES.map(b => table[b]?.[geo]).filter(v => v != null)
    stats[geo] = { min: Math.min(...values), max: Math.max(...values) }
  })

  return (
    <div className="summary-card">
      <div className="summary-card__header">
        <h2 className="chart-card__title">Resumen global — MAE modelo ampliado</h2>
        <p className="chart-card__sub">Por bloque y país · Verde: mejor resultado · Rojo: peor resultado</p>
      </div>

      <div className="summary-card__body">
        <table className="summary-table">
          <thead>
            <tr>
              <th className="summary-table__th summary-table__th--left">Bloque</th>
              {GEOS.map(geo => (
                <th key={geo} className="summary-table__th">{geo}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {BLOQUES.map(bloque => (
              <tr key={bloque} className="summary-table__row">
                <td className="summary-table__td summary-table__td--label">
                  {BLOQUE_LABEL[bloque]}
                </td>
                {GEOS.map(geo => {
                  const val = table[bloque]?.[geo]
                  if (val == null) return <td key={geo} className="summary-table__td">—</td>
                  const isMin = val === stats[geo].min
                  const isMax = val === stats[geo].max
                  return (
                    <td
                      key={geo}
                      className={`summary-table__td summary-table__td--value
                        ${isMin ? 'summary-table__td--best' : ''}
                        ${isMax ? 'summary-table__td--worst' : ''}
                      `}
                    >
                      {val.toFixed(3)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
