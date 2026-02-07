import { useRef, useEffect } from 'react'

interface SparklineProps {
  data: number[]
  data2?: number[]
  width?: number
  height?: number
  color?: string
  color2?: string
  lineWidth?: number
  className?: string
  showDots?: boolean
  animated?: boolean
}

export default function Sparkline({
  data,
  data2,
  width = 120,
  height = 32,
  color = '#22c55e',
  color2 = '#3b82f6',
  lineWidth = 1.5,
  className = '',
  showDots = false,
  animated = true,
}: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length < 2) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Handle retina displays
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    // Clear
    ctx.clearRect(0, 0, width, height)

    // Combine all data for consistent scaling
    const allData = [...data, ...(data2 || [])]
    const min = Math.min(...allData)
    const max = Math.max(...allData)
    const range = max - min || 1

    const padding = 2

    const drawLine = (points: number[], strokeColor: string) => {
      if (points.length < 2) return

      const xStep = (width - padding * 2) / (points.length - 1)

      ctx.beginPath()
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = lineWidth
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'

      points.forEach((value, i) => {
        const x = padding + i * xStep
        const y = padding + (height - padding * 2) * (1 - (value - min) / range)

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()

      // Draw end dot
      if (showDots && points.length > 0) {
        const lastX = padding + (points.length - 1) * xStep
        const lastY = padding + (height - padding * 2) * (1 - (points[points.length - 1] - min) / range)
        ctx.beginPath()
        ctx.arc(lastX, lastY, 2, 0, Math.PI * 2)
        ctx.fillStyle = strokeColor
        ctx.fill()
      }

      // Draw gradient fill
      const gradient = ctx.createLinearGradient(0, 0, 0, height)
      gradient.addColorStop(0, strokeColor + '20')
      gradient.addColorStop(1, strokeColor + '00')

      ctx.beginPath()
      points.forEach((value, i) => {
        const x = padding + i * xStep
        const y = padding + (height - padding * 2) * (1 - (value - min) / range)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.lineTo(padding + (points.length - 1) * xStep, height)
      ctx.lineTo(padding, height)
      ctx.closePath()
      ctx.fillStyle = gradient
      ctx.fill()
    }

    // Draw lines
    drawLine(data, color)
    if (data2 && data2.length >= 2) {
      drawLine(data2, color2)
    }
  }, [data, data2, width, height, color, color2, lineWidth, showDots, animated])

  if (data.length < 2) {
    return <div className={className} style={{ width, height }} />
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className={className}
    />
  )
}

// Utility: generate sparkline data from price history
export function generateSparklineData(prices: number[], points = 20): number[] {
  if (prices.length <= points) return prices
  const step = (prices.length - 1) / (points - 1)
  return Array.from({ length: points }, (_, i) => {
    const idx = Math.min(Math.round(i * step), prices.length - 1)
    return prices[idx]
  })
}
