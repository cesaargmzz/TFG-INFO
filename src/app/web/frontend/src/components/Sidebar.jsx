import './Sidebar.css'

const BLOQUES = [
  { value: '0', label: 'Bloque 0 — Por país' },
  { value: '1', label: 'Bloque 1 — Panel combinado' },
  { value: '2', label: 'Bloque 2 — Panel con geo' },
  { value: '3', label: 'Bloque 3 — Transferencia' },
]

const PAISES = [
  { value: 'ES', label: 'España' },
  { value: 'FR', label: 'Francia' },
  { value: 'IT', label: 'Italia' },
]

const MODELOS = {
  '0': [{ value: 'base', label: 'Base' }, { value: 'ext', label: 'Ampliado' }],
  '1': [{ value: 'base', label: 'Base' }, { value: 'ext', label: 'Ampliado' }],
  '2': [{ value: 'base', label: 'Base' }, { value: 'ext', label: 'Ampliado' }],
  '3': [{ value: 'ext', label: 'Ampliado' }],
}

export default function Sidebar({ state, onChange }) {
  const { bloque, geo, modelo, showSummary } = state
  const modeloOptions = MODELOS[bloque]

  const set = (key, value) => onChange({ ...state, [key]: value })

  const handleBloque = (value) => {
    const opts = MODELOS[value]
    const newModelo = opts.find(o => o.value === modelo) ? modelo : opts[0].value
    onChange({ ...state, bloque: value, modelo: newModelo })
  }

  return (
    <aside className="sidebar">
      <div className="sidebar__logo">
        <span className="sidebar__logo-title">GDPulse</span>
        <span className="sidebar__logo-sub">PIB QoQ · Zona Euro</span>
      </div>

      <nav className="sidebar__controls">
        <div className="sidebar__group">
          <label className="sidebar__label">Bloque experimental</label>
          <select
            className="sidebar__select"
            value={bloque}
            onChange={e => handleBloque(e.target.value)}
          >
            {BLOQUES.map(b => (
              <option key={b.value} value={b.value}>{b.label}</option>
            ))}
          </select>
        </div>

        <div className="sidebar__group">
          <label className="sidebar__label">País</label>
          <select
            className="sidebar__select"
            value={geo}
            onChange={e => set('geo', e.target.value)}
          >
            {PAISES.map(p => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>

        <div className="sidebar__group">
          <label className="sidebar__label">Modelo</label>
          <select
            className="sidebar__select"
            value={modelo}
            onChange={e => set('modelo', e.target.value)}
          >
            {modeloOptions.map(m => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>

        <div className="sidebar__divider" />

        <div className="sidebar__toggle-group">
          <label className="sidebar__toggle">
            <span className="sidebar__toggle-label">Resumen global</span>
            <input
              type="checkbox"
              checked={showSummary}
              onChange={e => set('showSummary', e.target.checked)}
            />
            <span className="sidebar__toggle-track" />
          </label>
        </div>
      </nav>

      <div className="sidebar__footer">
        <span>TFG · 2026</span>
      </div>
    </aside>
  )
}
