import { useState, useEffect, useRef, useCallback } from 'react'

interface SimulatableItem {
  id: string
  roi_percent: number
  net_profit: number
  total_cost: number
  markets: Array<{
    id: string
    yes_price: number
    no_price: number
    liquidity: number
    question: string
  }>
  [key: string]: any
}

interface UseDataSimulationOptions {
  enabled?: boolean
  intervalMs?: number
  changeChance?: number
  maxChangePercent?: number
}

export function useDataSimulation<T extends SimulatableItem>(
  data: T[],
  options: UseDataSimulationOptions = {}
): { simulatedData: T[]; isSimulating: boolean; lastSimulatedAt: string | null } {
  const {
    enabled = true,
    intervalMs = 3000,
    changeChance = 0.2,
    maxChangePercent = 0.5,
  } = options

  const [simulatedData, setSimulatedData] = useState<T[]>(data)
  const [isSimulating, setIsSimulating] = useState(false)
  const [lastSimulatedAt, setLastSimulatedAt] = useState<string | null>(null)
  const dataRef = useRef(data)

  // Reset simulated data when real data changes
  useEffect(() => {
    dataRef.current = data
    setSimulatedData(data)
  }, [data])

  const simulateUpdate = useCallback(() => {
    if (!enabled || dataRef.current.length === 0) return

    setIsSimulating(true)
    setSimulatedData(prev => {
      return prev.map(item => {
        // Each item has a chance of being updated
        if (Math.random() > changeChance) return item

        // Simulate small market price changes
        const updatedMarkets = item.markets.map(market => {
          const changePercent = (Math.random() - 0.5) * 2 * (maxChangePercent / 100)
          const newYesPrice = Math.max(0.01, Math.min(0.99, market.yes_price * (1 + changePercent)))
          const newNoPrice = Math.max(0.01, Math.min(0.99, 1 - newYesPrice + (Math.random() - 0.5) * 0.005))

          return {
            ...market,
            yes_price: Math.round(newYesPrice * 10000) / 10000,
            no_price: Math.round(newNoPrice * 10000) / 10000,
          }
        })

        // Recalculate profit metrics
        const totalCost = updatedMarkets.reduce((sum, m) => sum + Math.min(m.yes_price, m.no_price), 0)
        const expectedPayout = 1.0
        const grossProfit = expectedPayout - totalCost
        const fee = grossProfit * 0.02
        const netProfit = grossProfit - fee
        const roiPercent = totalCost > 0 ? (netProfit / totalCost) * 100 : 0

        return {
          ...item,
          markets: updatedMarkets,
          total_cost: Math.round(totalCost * 10000) / 10000,
          net_profit: Math.round(netProfit * 10000) / 10000,
          roi_percent: Math.round(roiPercent * 100) / 100,
        }
      })
    })

    setLastSimulatedAt(new Date().toISOString())

    // Brief flash to show simulation happened
    setTimeout(() => setIsSimulating(false), 200)
  }, [enabled, changeChance, maxChangePercent])

  useEffect(() => {
    if (!enabled) return

    const interval = setInterval(simulateUpdate, intervalMs)
    return () => clearInterval(interval)
  }, [enabled, intervalMs, simulateUpdate])

  return { simulatedData, isSimulating, lastSimulatedAt }
}
