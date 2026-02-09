import { useEffect, useRef, useState } from 'react'
import { motion, useSpring, useMotionValue } from 'framer-motion'

interface AnimatedNumberProps {
  value: number
  prefix?: string
  suffix?: string
  decimals?: number
  className?: string
  duration?: number
}

export default function AnimatedNumber({
  value,
  prefix = '',
  suffix = '',
  decimals = 1,
  className = '',
  duration = 0.6,
}: AnimatedNumberProps) {
  const motionValue = useMotionValue(0)
  const springValue = useSpring(motionValue, {
    stiffness: 100,
    damping: 20,
    duration: duration * 1000,
  })
  const [displayValue, setDisplayValue] = useState(value)
  const prevValue = useRef(value)

  useEffect(() => {
    motionValue.set(prevValue.current)
    // Small delay to trigger the spring
    const raf = requestAnimationFrame(() => {
      motionValue.set(value)
    })
    prevValue.current = value
    return () => cancelAnimationFrame(raf)
  }, [value, motionValue])

  useEffect(() => {
    const unsubscribe = springValue.on('change', (latest) => {
      setDisplayValue(latest)
    })
    return unsubscribe
  }, [springValue])

  const formatted = `${prefix}${displayValue.toFixed(decimals)}${suffix}`

  return (
    <span className={className}>
      {formatted}
    </span>
  )
}

/**
 * Simple animated counter that flashes green/red on change
 */
export function FlashNumber({
  value,
  prefix = '',
  suffix = '',
  decimals = 2,
  className = '',
  positiveClass = 'data-glow-green',
  negativeClass = 'data-glow-red',
}: AnimatedNumberProps & {
  positiveClass?: string
  negativeClass?: string
}) {
  const [flash, setFlash] = useState<'up' | 'down' | null>(null)
  const prevValue = useRef(value)

  useEffect(() => {
    if (value !== prevValue.current) {
      setFlash(value > prevValue.current ? 'up' : 'down')
      prevValue.current = value
      const timer = setTimeout(() => setFlash(null), 600)
      return () => clearTimeout(timer)
    }
  }, [value])

  return (
    <motion.span
      className={`${className} ${flash === 'up' ? positiveClass : flash === 'down' ? negativeClass : ''}`}
      animate={flash ? { scale: [1, 1.05, 1] } : {}}
      transition={{ duration: 0.3 }}
    >
      {prefix}{value.toFixed(decimals)}{suffix}
    </motion.span>
  )
}
