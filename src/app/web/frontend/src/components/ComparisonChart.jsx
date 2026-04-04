import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts'
import './PredictionChart.css'

// Datos de ambos modelos combinados por periodo
const MOCK_BASE = {
  ES: [
    { period: '2023Q4', real: 0.63, base: 0.51, ext: 0.70 },
    { period: '2024Q1', real: 0.72, base: 0.68, ext: 0.55 },
    { period: '2024Q2', real: 0.81, base: 0.74, ext: 0.43 },
    { period: '2024Q3', real: 0.79, base: 0.82, ext: 1.45 },
    { period: '2024Q4', real: 0.71, base: 0.58, ext: 0.62 },
    { period: '2025Q1', real: 0.62, base: 0.70, ext: 0.55 },
    { period: '2025Q2', real: 0.55, base: 0.61, ext: 0.48 },
    { period: '2025Q3', real: 0.60, base: 0.53, ext: 0.52 },
  ],
  FR: [
    { period: '2023Q4', real: 0.20, base: 0.35, ext: 0.22 },
    { period: '2024Q1', real: 0.25, base: 0.40, ext: 0.27 },
    { period: '2024Q2', real: 0.30, base: 0.22, ext: 0.28 },
    { period: '2024Q3', real: 0.40, base: 0.55, ext: 0.38 },
    { period: '2024Q4', real: 0.10, base: 0.28, ext: 0.12 },
    { period: '2025Q1', real: 0.22, base: 0.31, ext: 0.21 },
    { period: '2025Q2', real: 0.18, base: 0.25, ext: 0.19 },
    { period: '2025Q3', real: 0.28, base: 0.20, ext: 0.26 },
  ],
  IT: [
    { period: '2023Q4', real: 0.10, base: 0.55, ext: 0.25 },
    { period: '2024Q1', real: -0.10, base: 0.42, ext: 0.08 },
    { period: '2024Q2', real: 0.20, base: 0.38, ext: 0.32 },
    { period: '2024Q3', real: 0.00, base: 0.60, ext: 0.15 },
    { period: '2024Q4', real: 0.15, base: 0.55, ext: 0.22 },
    { period: '2025Q1', real: -0.20, base: 0.48, ext: -0.10 },
    { period: '2025Q2', real: 0.10, base: 0.35, ext: 0.18 },
    { period: '2025Q3', real: 0.05, base: 0.42, ext: 0.12 },
  ],
}

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

export default function ComparisonChart({ bloque, geo, data }) {
  if (bloque === '3') return null

  const chartData = data ?? []

  return (
    <div className="chart-card">
      <div className="chart-card__header">
        <h2 className="chart-card__title">Base vs Ampliado</h2>
        <p className="chart-card__sub">Comparativa de modelos · {geo} · Bloque {bloque}</p>
      </div>
      <div className="chart-card__body">
        <ResponsiveContainer width="100%" height={640}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
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
            <Legend wrapperStyle={{ fontFamily: 'Inter', fontSize: 12, color: '#555a7a' }} />
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
              dataKey="base"
              name="Base"
              stroke="#3b6fe8"
              strokeWidth={2}
              strokeDasharray="5 3"
              dot={{ r: 3, fill: '#3b6fe8' }}
              activeDot={{ r: 5 }}
            />
            <Line
              type="monotone"
              dataKey="ext"
              name="Ampliado"
              stroke="#7c5cbf"
              strokeWidth={2}
              dot={{ r: 3, fill: '#7c5cbf' }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
