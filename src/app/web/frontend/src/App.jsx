import { useState } from 'react'
import IntroScreen from './components/IntroScreen'
import Dashboard from './components/Dashboard'
import './App.css'

function App() {
  const [showDashboard, setShowDashboard] = useState(false)
  const [introVisible, setIntroVisible] = useState(true)

  const [entering, setEntering] = useState(false)

  const handleEnter = () => {
    setIntroVisible(false)
    setShowDashboard(true)
    setEntering(true)
    setTimeout(() => setEntering(false), 1000)
  }

  return (
    <>
      {introVisible && <IntroScreen onEnter={handleEnter} />}
      {showDashboard && (
        <div className={entering ? 'dashboard-entering' : ''}>
          <Dashboard />
        </div>
      )}
    </>
  )
}

export default App
