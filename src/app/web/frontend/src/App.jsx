import { useState } from 'react'
import IntroScreen from './components/IntroScreen'
import './App.css'

function App() {
  const [showDashboard, setShowDashboard] = useState(false)

  return (
    <>
      {!showDashboard && (
        <IntroScreen onEnter={() => setShowDashboard(true)} />
      )}
      {showDashboard && (
        <div className="dashboard-placeholder">
          <p>Dashboard — próximamente</p>
        </div>
      )}
    </>
  )
}

export default App
