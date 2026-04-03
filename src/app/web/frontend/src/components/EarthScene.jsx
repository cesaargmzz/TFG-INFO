import { useEffect, useRef } from 'react'
import * as THREE from 'three'

// Texturas locales — servidas desde public/textures/
const TEX = {
  earth: '/textures/earth.jpg',
  water: '/textures/water.png',
  night: '/textures/night.jpg',
}

const CAM_START     = 2.8
const CAM_END       = 0.98
const SCROLL_SPEED  = 0.0012
const OVERLAY_START = 0.6

export default function EarthScene({ onComplete, overlayRef, hintRef, contentRef }) {
  const mountRef = useRef(null)

  useEffect(() => {
    const mount = mountRef.current
    const W = mount.clientWidth
    const H = mount.clientHeight

    // ── Renderer ──────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(W, H)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.1
    // Canvas invisible hasta que todas las texturas estén listas
    renderer.domElement.style.opacity   = '0'
    renderer.domElement.style.transition = 'opacity 1.2s ease'
    mount.appendChild(renderer.domElement)

    // ── Scene & Camera ─────────────────────────────
    const scene  = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, W / H, 0.01, 1000)
    camera.position.z = CAM_START

    // ── Estrellas ──────────────────────────────────
    const starPos = new Float32Array(8000 * 3)
    for (let i = 0; i < starPos.length; i++) starPos[i] = (Math.random() - 0.5) * 400
    const starGeo = new THREE.BufferGeometry()
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
    scene.add(new THREE.Points(starGeo,
      new THREE.PointsMaterial({ color: 0xffffff, size: 0.18, sizeAttenuation: true })
    ))

    // ── Iluminación ────────────────────────────────
    scene.add(new THREE.AmbientLight(0x111122, 1.2))
    const sun = new THREE.DirectionalLight(0xfff5e0, 3.5)
    sun.position.set(5, 3, 5)
    scene.add(sun)

    // ── Texturas locales con LoadingManager ────────
    const manager = new THREE.LoadingManager()
    manager.onLoad = () => {
      renderer.domElement.style.opacity = '1'
    }

    const loader = new THREE.TextureLoader(manager)
    const tex = {
      earth: loader.load(TEX.earth),
      water: loader.load(TEX.water),
      night: loader.load(TEX.night),
    }

    // ── Tierra ─────────────────────────────────────
    const earth = new THREE.Mesh(
      new THREE.SphereGeometry(1, 64, 64),
      new THREE.MeshPhongMaterial({
        map: tex.earth, specularMap: tex.water,
        specular: new THREE.Color(0x2244aa), shininess: 25,
      })
    )
    scene.add(earth)

    // Luces nocturnas
    const nightMesh = new THREE.Mesh(
      new THREE.SphereGeometry(1.001, 64, 64),
      new THREE.MeshLambertMaterial({
        map: tex.night, transparent: true,
        blending: THREE.AdditiveBlending, opacity: 0.9,
      })
    )
    scene.add(nightMesh)

    // Atmósfera interior
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1.08, 64, 64),
      new THREE.MeshPhongMaterial({
        color: 0x4488ff, transparent: true, opacity: 0.08,
        side: THREE.FrontSide, depthWrite: false,
      })
    ))

    // Glow exterior
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(1.18, 64, 64),
      new THREE.MeshPhongMaterial({
        color: 0x2255cc, transparent: true, opacity: 0.04,
        side: THREE.BackSide, depthWrite: false,
      })
    ))

    // ── Scroll — todo en variables locales, sin React state ─
    let progress       = 0   // valor objetivo (salta en cada wheel)
    let smoothProgress = 0   // valor suavizado (lerp en el loop)
    let completed      = false

    const onWheel = (e) => {
      if (completed) return
      e.preventDefault()
      progress = Math.min(Math.max(progress + e.deltaY * SCROLL_SPEED, 0), 1)
      if (hintRef?.current && progress > 0.05) hintRef.current.style.opacity = '0'
    }
    window.addEventListener('wheel', onWheel, { passive: false })

    // ── Resize ─────────────────────────────────────
    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', onResize)

    // ── Loop ───────────────────────────────────────
    let animId
    const animate = () => {
      animId = requestAnimationFrame(animate)
      earth.rotation.y     += 0.0008
      nightMesh.rotation.y += 0.0008

      // Suavizar progress — cámara y FOV van juntos al mismo ritmo
      smoothProgress += (progress - smoothProgress) * 0.07

      camera.position.z = CAM_START + (CAM_END - CAM_START) * smoothProgress
      camera.fov = 45 + smoothProgress * 20
      camera.updateProjectionMatrix()

      // Overlay oscuro
      if (overlayRef?.current) {
        const op = Math.max(0, (smoothProgress - OVERLAY_START) / (1 - OVERLAY_START))
        overlayRef.current.style.opacity = op
      }

      // Texto: desvanece progresivamente entre 0% y 70% del scroll
      if (contentRef?.current) {
        const op = Math.max(0, 1 - progress / 0.7)
        contentRef.current.style.opacity = op
      }

      // Disparar transición cuando smoothProgress llega al final
      if (!completed && smoothProgress > 0.97) {
        completed = true
        onComplete?.()
      }

      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('wheel', onWheel)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
}
