import { useState, useRef, useEffect } from 'react'
import EarthScene from './EarthScene'
import './IntroScreen.css'

export default function IntroScreen({ onEnter }) {
  const [fadeOut, setFadeOut] = useState(false)
  const overlayRef  = useRef(null)
  const hintRef     = useRef(null)
  const contentRef  = useRef(null)

  const handleComplete = () => {
    setFadeOut(true)
    setTimeout(onEnter, 800)
  }

  // Fade-in del texto tras la animación de entrada (0.4s delay + 1.2s duración)
  useEffect(() => {
    const t = setTimeout(() => {
      if (contentRef.current) contentRef.current.style.opacity = '1'
    }, 1000)
    return () => clearTimeout(t)
  }, [])

  return (
    <div className="intro">
      <div className="intro__canvas">
        <EarthScene
          onComplete={handleComplete}
          overlayRef={overlayRef}
          hintRef={hintRef}
          contentRef={contentRef}
        />
      </div>

      <div ref={overlayRef} className="intro__overlay" />

      {/* Contenido izquierdo */}
      <div ref={contentRef} className="intro__content">
        <p className="intro__label">Trabajo de Fin de Grado · 2026</p>
        <h1 className="intro__title">GDPulse</h1>
        <p className="intro__desc">
          Análisis comparativo de modelos de aprendizaje automático
          para la estimación del crecimiento del PIB en la zona euro.
        </p>
        <p className="intro__countries">España · Francia · Italia</p>
      </div>

      {/* Hint de scroll */}
      <div ref={hintRef} className="intro__hint">
        <div className="intro__hint-icon">
          <span /><span /><span />
        </div>
        <p>Desplaza para continuar</p>
      </div>

      {fadeOut && <div className="intro__fadeout" />}
    </div>
  )
}
