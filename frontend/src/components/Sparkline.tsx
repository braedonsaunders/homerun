import { useRef, useEffect } from 'react'

interface SparklineProps {
  data: number[]
  data2?: number[]
  width?: number
  height?: number
  minValue?: number
  maxValue?: number
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
  minValue,
  maxValue,
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
    const hasManualScale =
      Number.isFinite(minValue)
      && Number.isFinite(maxValue)
      && Number(maxValue) > Number(minValue)
    let min = hasManualScale ? Number(minValue) : Math.min(...allData)
    let max = hasManualScale ? Number(maxValue) : Math.max(...allData)
    if (!hasManualScale && Number.isFinite(min) && Number.isFinite(max)) {
      const dynamicRange = max - min
      const minVisualRange = 0.01
      if (dynamicRange > 0 && dynamicRange < minVisualRange) {
        const mid = (min + max) / 2
        min = mid - minVisualRange / 2
        max = mid + minVisualRange / 2
      }
    }
    const range = max - min

    const padding = 3

    const toY = (value: number): number => {
      if (!(range > 0)) return height / 2
      return padding + (height - padding * 2) * (1 - (value - min) / range)
    }

    const drawLine = (
      points: number[],
      strokeColor: string,
      options: { drawFill?: boolean; drawDot?: boolean; width?: number } = {}
    ) => {
      if (points.length < 2) return

      const xStep = (width - padding * 2) / (points.length - 1)
      const drawFill = options.drawFill ?? true
      const drawDot = options.drawDot ?? showDots
      const strokeWidth = options.width ?? lineWidth

      ctx.beginPath()
      ctx.strokeStyle = strokeColor
      ctx.lineWidth = strokeWidth
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'

      points.forEach((value, i) => {
        const x = padding + i * xStep
        const y = toY(value)

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()

      // Draw end dot
      if (drawDot && points.length > 0) {
        const lastX = padding + (points.length - 1) * xStep
        const lastY = toY(points[points.length - 1])
        ctx.beginPath()
        ctx.arc(lastX, lastY, 2, 0, Math.PI * 2)
        ctx.fillStyle = strokeColor
        ctx.fill()
      }

      // Draw gradient fill
      if (!drawFill) return
      const gradient = ctx.createLinearGradient(0, 0, 0, height)
      gradient.addColorStop(0, strokeColor + '20')
      gradient.addColorStop(1, strokeColor + '00')

      ctx.beginPath()
      points.forEach((value, i) => {
        const x = padding + i * xStep
        const y = toY(value)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.lineTo(padding + (points.length - 1) * xStep, height)
      ctx.lineTo(padding, height)
      ctx.closePath()
      ctx.fillStyle = gradient
      ctx.fill()
    }

    // Draw secondary first, then primary so YES is never visually hidden.
    if (data2 && data2.length >= 2) {
      drawLine(data2, color2, {
        drawFill: false,
        drawDot: false,
        width: Math.max(1, lineWidth - 0.2),
      })
    }
    drawLine(data, color, { drawFill: true, drawDot: showDots, width: lineWidth })
  }, [data, data2, width, height, minValue, maxValue, color, color2, lineWidth, showDots, animated])

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
