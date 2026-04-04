import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Cell, ResponsiveContainer, ReferenceLine
} from 'recharts'
import './PredictionChart.css'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const val = payload[0].value
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__label">{label}</p>
      <p style={{ color: val >= 0 ? '#3b6fe8' : '#e85555' }}>
        Error: <strong>{val > 0 ? '+' : ''}{val.toFixed(3)}</strong>
      </p>
    </div>
  )
}

export default function ErrorChart({ predictions, loading }) {
  const data = (predictions ?? []).map(d => ({
    period: d.period,
    error: parseFloat(d.error.toFixed(3)),
  }))

  return (
    <div className="chart-card">
      <div className="chart-card__header">
        <h2 className="chart-card__title">Error por trimestre</h2>
        <p className="chart-card__sub">Predicción − Real · Azul: sobreestimación · Rojo: subestimación</p>
      </div>
      <div className="chart-card__body">
        <ResponsiveContainer width="100%" height={640}>
          <BarChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,110,160,0.1)" vertical={false} />
            <XAxis
              dataKey="period"
              tick={{ fontFamily: 'Inter', fontSize: 11, fill: '#555a7a' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontFamily: 'Inter', fontSize: 11, fill: '#555a7a' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => v.toFixed(2)}
            />
            <ReferenceLine y={0} stroke="rgba(100,110,160,0.3)" />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(100,110,160,0.05)' }} />
            <Bar dataKey="error" name="Error" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.error >= 0 ? '#3b6fe8' : '#e85555'}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
