import { useState, useEffect, useRef, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  Search,
  Brain,
  Shield,
  TrendingUp,
  Newspaper,
  MessageCircle,
  X,
  RefreshCw,
  Sparkles,
  Command,
} from 'lucide-react'
import clsx from 'clsx'
import { sendAIChat, searchMarkets, MarketSearchResult } from '../services/api'

interface AICommandBarProps {
  isOpen: boolean
  onClose: () => void
  onNavigateToAI?: (section: string) => void
  onOpenCopilot?: (contextType?: string, contextId?: string, label?: string) => void
}

type CommandMode = 'search' | 'ask' | 'market-search'

export default function AICommandBar({
  isOpen,
  onClose,
  onNavigateToAI,
  onOpenCopilot,
}: AICommandBarProps) {
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<CommandMode>('search')
  const [marketResults, setMarketResults] = useState<MarketSearchResult[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>()

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        if (isOpen) {
          onClose()
        }
      }
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  useEffect(() => {
    if (isOpen) {
      setInput('')
      setMode('search')
      setMarketResults([])
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [isOpen])

  // Market search with debounce
  const handleMarketSearch = useCallback(async (query: string) => {
    if (query.length < 2) {
      setMarketResults([])
      return
    }
    try {
      const data = await searchMarkets(query, 8)
      setMarketResults(data.results)
    } catch {
      setMarketResults([])
    }
  }, [])

  useEffect(() => {
    if (mode === 'market-search' && input.length >= 2) {
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(() => handleMarketSearch(input), 300)
    }
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [input, mode, handleMarketSearch])

  // Quick ask mutation
  const askMutation = useMutation({
    mutationFn: async (question: string) => {
      const result = await sendAIChat({ message: question })
      return result
    },
  })

  const commands = [
    {
      id: 'ask-ai',
      label: 'Ask AI a question',
      description: 'Chat with the AI copilot about anything',
      icon: <MessageCircle className="w-4 h-4" />,
      color: 'text-purple-400',
      action: () => {
        setMode('ask')
        setInput('')
      },
    },
    {
      id: 'find-market',
      label: 'Find a market',
      description: 'Search markets by name (no more manual IDs)',
      icon: <Search className="w-4 h-4" />,
      color: 'text-blue-400',
      action: () => {
        setMode('market-search')
        setInput('')
      },
    },
    {
      id: 'resolution-analysis',
      label: 'Resolution Analysis',
      description: 'Analyze how a market will resolve',
      icon: <Shield className="w-4 h-4" />,
      color: 'text-green-400',
      action: () => {
        onNavigateToAI?.('resolution')
        onClose()
      },
    },
    {
      id: 'market-analysis',
      label: 'Market Analysis',
      description: 'Deep-dive AI analysis on any topic',
      icon: <TrendingUp className="w-4 h-4" />,
      color: 'text-cyan-400',
      action: () => {
        onNavigateToAI?.('market')
        onClose()
      },
    },
    {
      id: 'news-sentiment',
      label: 'News Sentiment',
      description: 'Search news and analyze sentiment',
      icon: <Newspaper className="w-4 h-4" />,
      color: 'text-orange-400',
      action: () => {
        onNavigateToAI?.('news')
        onClose()
      },
    },
    {
      id: 'open-copilot',
      label: 'Open AI Copilot',
      description: 'Open the AI assistant panel',
      icon: <Sparkles className="w-4 h-4" />,
      color: 'text-purple-400',
      action: () => {
        onOpenCopilot?.()
        onClose()
      },
    },
  ]

  const filteredCommands = mode === 'search'
    ? commands.filter(
        (c) =>
          input === '' ||
          c.label.toLowerCase().includes(input.toLowerCase()) ||
          c.description.toLowerCase().includes(input.toLowerCase())
      )
    : []

  const handleKeyDown = (e: React.KeyboardEvent) => {
    const items = mode === 'search' ? filteredCommands : mode === 'market-search' ? marketResults : []
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex((i) => Math.min(i + 1, items.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (mode === 'search' && filteredCommands[selectedIndex]) {
        filteredCommands[selectedIndex].action()
      } else if (mode === 'ask' && input.trim()) {
        askMutation.mutate(input.trim())
      } else if (mode === 'market-search' && marketResults[selectedIndex]) {
        const m = marketResults[selectedIndex]
        onNavigateToAI?.('resolution')
        onClose()
        // Store selected market for the resolution panel to pick up
        window.dispatchEvent(
          new CustomEvent('market-selected', { detail: m })
        )
      }
    } else if (e.key === 'Backspace' && input === '' && mode !== 'search') {
      setMode('search')
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[15vh]">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Command Bar */}
      <div className="relative w-full max-w-xl bg-[#0f0f0f] border border-gray-800 rounded-2xl shadow-2xl shadow-purple-500/10 overflow-hidden">
        {/* Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-800">
          {mode === 'search' && (
            <div className="flex items-center gap-1 text-gray-500">
              <Command className="w-4 h-4" />
              <span className="text-xs">K</span>
            </div>
          )}
          {mode === 'ask' && (
            <Brain className="w-4 h-4 text-purple-400 flex-shrink-0" />
          )}
          {mode === 'market-search' && (
            <Search className="w-4 h-4 text-blue-400 flex-shrink-0" />
          )}
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              setSelectedIndex(0)
            }}
            onKeyDown={handleKeyDown}
            placeholder={
              mode === 'search'
                ? 'Search AI commands...'
                : mode === 'ask'
                  ? 'Ask anything about markets, strategies, risk...'
                  : 'Type to search markets...'
            }
            className="flex-1 bg-transparent text-sm text-white placeholder-gray-600 focus:outline-none"
          />
          {mode !== 'search' && (
            <button
              onClick={() => {
                setMode('search')
                setInput('')
              }}
              className="text-xs text-gray-500 hover:text-gray-400 px-2 py-1 bg-gray-800 rounded-lg"
            >
              ESC
            </button>
          )}
          <button onClick={onClose} className="p-1 hover:bg-gray-800 rounded-lg">
            <X className="w-4 h-4 text-gray-500" />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[400px] overflow-y-auto">
          {/* Command list */}
          {mode === 'search' && (
            <div className="p-2">
              {filteredCommands.map((cmd, i) => (
                <button
                  key={cmd.id}
                  onClick={cmd.action}
                  onMouseEnter={() => setSelectedIndex(i)}
                  className={clsx(
                    'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-colors',
                    i === selectedIndex ? 'bg-[#1a1a1a]' : 'hover:bg-[#141414]'
                  )}
                >
                  <div className={clsx('flex-shrink-0', cmd.color)}>
                    {cmd.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white">{cmd.label}</p>
                    <p className="text-xs text-gray-500">{cmd.description}</p>
                  </div>
                </button>
              ))}
              {filteredCommands.length === 0 && (
                <p className="text-sm text-gray-500 text-center py-4">No matching commands</p>
              )}
            </div>
          )}

          {/* Market search results */}
          {mode === 'market-search' && (
            <div className="p-2">
              {marketResults.length > 0 ? (
                marketResults.map((m, i) => (
                  <button
                    key={m.market_id}
                    onMouseEnter={() => setSelectedIndex(i)}
                    onClick={() => {
                      onNavigateToAI?.('resolution')
                      onClose()
                      window.dispatchEvent(
                        new CustomEvent('market-selected', { detail: m })
                      )
                    }}
                    className={clsx(
                      'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-colors',
                      i === selectedIndex ? 'bg-[#1a1a1a]' : 'hover:bg-[#141414]'
                    )}
                  >
                    <Shield className="w-4 h-4 text-green-400 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-white truncate">{m.question}</p>
                      <p className="text-xs text-gray-500">
                        {m.event_title && <span>{m.event_title} | </span>}
                        {m.category && <span>{m.category} | </span>}
                        YES: ${m.yes_price?.toFixed(2)} | Liq: ${m.liquidity?.toFixed(0)}
                      </p>
                    </div>
                  </button>
                ))
              ) : input.length >= 2 ? (
                <p className="text-sm text-gray-500 text-center py-4">No markets found</p>
              ) : (
                <p className="text-sm text-gray-500 text-center py-4">
                  Start typing to search markets...
                </p>
              )}
            </div>
          )}

          {/* Ask AI response */}
          {mode === 'ask' && (
            <div className="p-4">
              {askMutation.isPending && (
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <RefreshCw className="w-4 h-4 animate-spin text-purple-400" />
                  Thinking...
                </div>
              )}
              {askMutation.data && (
                <div className="bg-[#1a1a1a] rounded-xl p-4 border border-gray-800">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <span className="text-xs text-gray-500">AI Response</span>
                  </div>
                  <p className="text-sm text-gray-300 whitespace-pre-wrap">
                    {askMutation.data.response}
                  </p>
                </div>
              )}
              {askMutation.error && (
                <p className="text-sm text-red-400">
                  {(askMutation.error as Error).message}
                </p>
              )}
              {!askMutation.isPending && !askMutation.data && !askMutation.error && (
                <p className="text-sm text-gray-500 text-center">
                  Type your question and press Enter
                </p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-gray-800 flex items-center gap-4 text-[10px] text-gray-600">
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-gray-800 rounded text-gray-500">Enter</kbd> select
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-gray-800 rounded text-gray-500">&uarr;&darr;</kbd> navigate
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-gray-800 rounded text-gray-500">Esc</kbd> close
          </span>
        </div>
      </div>
    </div>
  )
}
