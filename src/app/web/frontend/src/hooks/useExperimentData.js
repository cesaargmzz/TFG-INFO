import { useState, useEffect } from 'react'

const API = 'http://localhost:8000'

export function useExperimentData(bloque, geo, modelo) {
  const [predictions, setPredictions] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)

    const fetchData = async () => {
      try {
        const [predRes, metRes] = await Promise.all([
          fetch(`${API}/api/predictions/${bloque}/${geo}/${modelo}`),
          fetch(`${API}/api/metrics/${bloque}/${geo}`),
        ])

        if (!predRes.ok) throw new Error(`Predictions: ${predRes.status}`)
        if (!metRes.ok) throw new Error(`Metrics: ${metRes.status}`)

        const [predData, metData] = await Promise.all([
          predRes.json(),
          metRes.json(),
        ])

        setPredictions(predData)
        setMetrics(metData)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [bloque, geo, modelo])

  return { predictions, metrics, loading, error }
}

export function useComparisonData(bloque, geo) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (bloque === '3') { setData(null); setLoading(false); return }

    setLoading(true)
    Promise.all([
      fetch(`${API}/api/predictions/${bloque}/${geo}/base`).then(r => r.json()),
      fetch(`${API}/api/predictions/${bloque}/${geo}/ext`).then(r => r.json()),
    ])
      .then(([base, ext]) => {
        const merged = base.map((row, i) => ({
          period: row.period,
          real: row.y_true,
          base: row.y_pred,
          ext: ext[i]?.y_pred ?? null,
        }))
        setData(merged)
      })
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [bloque, geo])

  return { data, loading }
}

export function useSummary() {
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/api/summary`)
      .then(r => r.json())
      .then(setSummary)
      .finally(() => setLoading(false))
  }, [])

  return { summary, loading }
}
