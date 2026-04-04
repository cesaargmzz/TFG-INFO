import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts'
import './PredictionChart.css'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__label">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <strong>{p.value?.toFixed(3)}</strong>
        </p>
      ))}
    </div>
  )
}

export default function PredictionChart({ predictions, loading }) {
  const data = (predictions ?? []).map(d => ({
    period: d.period,
    real: d.y_true,
    pred: d.y_pred,
  }))

  return (
    <div className="chart-card">
      <div className="chart-card__header">
        <h2 className="chart-card__title">Real vs Predicción</h2>
        <p className="chart-card__sub">PIB QoQ % · Período de test (2023Q4 – 2025Q3)</p>
      </div>
      <div className="chart-card__body">
        <ResponsiveContainer width="100%" height={640}>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,110,160,0.1)" />
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
              tickFormatter={v => `${v.toFixed(1)}%`}
            />
            <ReferenceLine y={0} stroke="rgba(100,110,160,0.2)" />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontFamily: 'Inter', fontSize: 12, color: '#555a7a' }}
            />
            <Line
              type="monotone"
              dataKey="real"
              name="Real"
              stroke="#1a1c2a"
              strokeWidth={2}
              dot={{ r: 4, fill: '#1a1c2a' }}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name="Predicción"
              stroke="#3b6fe8"
              strokeWidth={2}
              strokeDasharray="5 3"
              dot={{ r: 4, fill: '#3b6fe8' }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
