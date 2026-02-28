import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  X,
  Send,
  Bot,
  User,
  RefreshCw,
  Minimize2,
  Maximize2,
  Trash2,
  Sparkles,
} from 'lucide-react'
import { cn } from '../lib/utils'
import { sendAIChat, AIChatMessage, getAIChatSession, archiveAIChatSession } from '../services/api'
import { Button } from './ui/button'
import { Card } from './ui/card'
import { ScrollArea } from './ui/scroll-area'
import { Separator } from './ui/separator'

interface AICopilotPanelProps {
  isOpen: boolean
  onClose: () => void
  contextType?: string
  contextId?: string
  contextLabel?: string
  seedPrompt?: {
    id: number
    prompt: string
    autoSend: boolean
  } | null
}

export default function AICopilotPanel({
  isOpen,
  onClose,
  contextType,
  contextId,
  contextLabel,
  seedPrompt,
}: AICopilotPanelProps) {
  const queryClient = useQueryClient()
  const [messages, setMessages] = useState<AIChatMessage[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [isExpanded, setIsExpanded] = useState(false)
  const [isSessionReady, setIsSessionReady] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const lastSeedPromptIdRef = useRef<number | null>(null)
  const sessionStorageKey = useMemo(() => {
    const ctxType = contextType || 'general'
    const ctxId = contextId || 'default'
    return `ai-copilot-session:${ctxType}:${ctxId}`
  }, [contextType, contextId])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) {
      setIsSessionReady(false)
      return
    }
    let cancelled = false
    setIsSessionReady(false)
    const existingSessionId = window.localStorage.getItem(sessionStorageKey)
    if (!existingSessionId) {
      setSessionId(null)
      setMessages([])
      setIsSessionReady(true)
      return
    }

    setMessages([])
    getAIChatSession(existingSessionId)
      .then((data) => {
        if (cancelled) return
        setSessionId(data.session_id)
        const restored = (data.messages || [])
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content }))
        setMessages(restored)
        setIsSessionReady(true)
      })
      .catch(() => {
        if (cancelled) return
        window.localStorage.removeItem(sessionStorageKey)
        setSessionId(null)
        setMessages([])
        setIsSessionReady(true)
      })

    return () => {
      cancelled = true
    }
  }, [isOpen, sessionStorageKey])

  const chatMutation = useMutation({
    mutationFn: async (message: string) => {
      const result = await sendAIChat({
        message,
        session_id: sessionId || undefined,
        context_type: contextType,
        context_id: contextId,
        allow_actions: true,
      })
      return result
    },
    onSuccess: (data) => {
      if (data.session_id) {
        setSessionId(data.session_id)
        window.localStorage.setItem(sessionStorageKey, data.session_id)
      }
      const applied = Array.isArray(data.actions_applied) ? data.actions_applied : []
      if (applied.length > 0) {
        for (const action of applied) {
          const actionType = String(action?.type || '').trim().toLowerCase()
          if (actionType === 'update_strategy') {
            queryClient.invalidateQueries({ queryKey: ['unified-strategies'] })
            queryClient.invalidateQueries({ queryKey: ['unified-strategy-versions'] })
            queryClient.invalidateQueries({ queryKey: ['strategies'] })
            queryClient.invalidateQueries({ queryKey: ['plugins'] })
            queryClient.invalidateQueries({ queryKey: ['trader-config-schema'] })
            queryClient.invalidateQueries({ queryKey: ['trader-sources'] })
          } else if (actionType === 'update_data_source') {
            queryClient.invalidateQueries({ queryKey: ['unified-data-sources'] })
            queryClient.invalidateQueries({ queryKey: ['unified-data-source'] })
            queryClient.invalidateQueries({ queryKey: ['unified-data-source-runs'] })
            queryClient.invalidateQueries({ queryKey: ['unified-data-source-records-preview'] })
            queryClient.invalidateQueries({ queryKey: ['events-sources'] })
          }
        }
      }
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response },
      ])
    },
    onError: (error: any) => {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${error?.response?.data?.detail || error?.message || 'Failed to get response'}`,
        },
      ])
    },
  })

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || chatMutation.isPending) return
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }])
    setInput('')
    chatMutation.mutate(trimmed)
  }

  useEffect(() => {
    if (!isOpen || !isSessionReady || !seedPrompt) return
    if (lastSeedPromptIdRef.current === seedPrompt.id) return
    lastSeedPromptIdRef.current = seedPrompt.id

    const prompt = String(seedPrompt.prompt || '').trim()
    if (!prompt) return

    if (!seedPrompt.autoSend || chatMutation.isPending) {
      setInput(prompt)
      setTimeout(() => inputRef.current?.focus(), 40)
      return
    }

    setMessages((prev) => [...prev, { role: 'user', content: prompt }])
    chatMutation.mutate(prompt)
  }, [isOpen, isSessionReady, seedPrompt, chatMutation])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const quickActions = useMemo(() => {
    if (contextType === 'strategy') {
      return [
        { label: 'Explain code', prompt: 'Explain this strategy code end-to-end: detect/evaluate/exit and main risk tradeoffs.' },
        { label: 'Improve evaluate()', prompt: 'Improve evaluate() to tighten risk gating and produce clearer DecisionCheck outputs.' },
        { label: 'Add feature', prompt: 'Add a configurable filter for market liquidity and wire it into default_config + config_schema. Apply the code changes directly.' },
        { label: 'Review safety', prompt: 'Review this strategy for potential logic bugs, execution hazards, and invalid assumptions.' },
      ]
    }
    if (contextType === 'data_source') {
      return [
        { label: 'Explain source', prompt: 'Explain this data source pipeline and where it may fail in production.' },
        { label: 'Harden parsing', prompt: 'Improve this source to better normalize timestamps, IDs, and categories. Apply the code changes directly.' },
        { label: 'Add geotags', prompt: 'Add robust geotag support and ensure output follows the record contract. Apply the code changes directly.' },
        { label: 'Retention review', prompt: 'Recommend retention/config changes for this source based on its expected volume and usage.' },
      ]
    }
    return [
      { label: 'Analyze risk factors', prompt: 'What are the main risk factors for this opportunity?' },
      { label: 'Resolution safety', prompt: 'How safe is the resolution criteria? Any ambiguities?' },
      { label: 'Should I trade?', prompt: 'Given the current data, should I execute this trade? What are the pros and cons?' },
      { label: 'Explain strategy', prompt: 'Explain how this arbitrage strategy works and why this opportunity exists.' },
    ]
  }, [contextType])

  if (!isOpen) return null

  return (
    <div
      className={cn(
        'fixed bottom-4 right-4 z-50 flex flex-col bg-background border border-border rounded-2xl shadow-2xl shadow-purple-500/5 transition-all',
        isExpanded ? 'w-[560px] h-[700px]' : 'w-[420px] h-[560px]'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-foreground" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">AI Copilot</h3>
            {contextLabel && (
              <p className="text-[10px] text-purple-400 truncate max-w-[200px]">
                {contextLabel}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <Button
              onClick={async () => {
                if (sessionId) {
                  try {
                    await archiveAIChatSession(sessionId)
                  } catch {
                    // ignore archive failures; local clear still proceeds
                  }
                }
                setMessages([])
                setSessionId(null)
                window.localStorage.removeItem(sessionStorageKey)
              }}
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              title="Clear chat"
            >
              <Trash2 className="w-3.5 h-3.5 text-muted-foreground" />
            </Button>
          )}
          <Button
            onClick={() => setIsExpanded(!isExpanded)}
            variant="ghost"
            size="icon"
            className="h-7 w-7"
          >
            {isExpanded ? (
              <Minimize2 className="w-3.5 h-3.5 text-muted-foreground" />
            ) : (
              <Maximize2 className="w-3.5 h-3.5 text-muted-foreground" />
            )}
          </Button>
          <Button
            onClick={onClose}
            variant="ghost"
            size="icon"
            className="h-7 w-7"
          >
            <X className="w-3.5 h-3.5 text-muted-foreground" />
          </Button>
        </div>
      </div>

      <Separator />

      {/* Messages */}
      <ScrollArea className="flex-1">
        <div className="px-4 py-3 space-y-3">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center mb-3">
                <Bot className="w-6 h-6 text-purple-400" />
              </div>
              <p className="text-sm text-muted-foreground mb-1">
                {contextType === 'strategy'
                  ? 'Ask me anything about this strategy'
                  : contextType === 'data_source'
                    ? 'Ask me anything about this data source'
                    : 'Ask me anything about your trades'}
              </p>
              <p className="text-xs text-muted-foreground mb-4">
                {contextType === 'strategy'
                  ? 'I can explain the code, suggest improvements, and apply direct strategy edits when requested.'
                  : contextType === 'data_source'
                    ? 'I can analyze ingestion logic, improve normalization, and apply direct source edits when requested.'
                    : 'I can analyze opportunities, assess risk, and help you make decisions.'}
              </p>

              {/* Quick Actions */}
              {(contextType === 'opportunity' || contextType === 'trader_signal' || contextType === 'strategy' || contextType === 'data_source') && (
                <div className="grid grid-cols-2 gap-2 w-full max-w-sm">
                  {quickActions.map((action) => (
                    <Card
                      key={action.label}
                      className="cursor-pointer text-left text-xs p-2.5 rounded-xl hover:border-purple-500/30 hover:bg-purple-500/5 transition-colors text-muted-foreground hover:text-foreground shadow-none"
                      onClick={() => {
                        setMessages((prev) => [...prev, { role: 'user', content: action.prompt }])
                        chatMutation.mutate(action.prompt)
                      }}
                    >
                      {action.label}
                    </Card>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={cn('flex gap-2', msg.role === 'user' ? 'justify-end' : 'justify-start')}
            >
              {msg.role === 'assistant' && (
                <div className="w-6 h-6 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Bot className="w-3.5 h-3.5 text-purple-400" />
                </div>
              )}
              <Card
                className={cn(
                  'max-w-[80%] rounded-xl px-3 py-2 text-sm shadow-none',
                  msg.role === 'user'
                    ? 'bg-blue-500/20 text-blue-100 border-0'
                    : 'bg-muted text-muted-foreground border border-border'
                )}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </Card>
              {msg.role === 'user' && (
                <div className="w-6 h-6 rounded-lg bg-blue-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <User className="w-3.5 h-3.5 text-blue-400" />
                </div>
              )}
            </div>
          ))}

          {chatMutation.isPending && (
            <div className="flex gap-2">
              <div className="w-6 h-6 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Bot className="w-3.5 h-3.5 text-purple-400" />
              </div>
              <Card className="rounded-xl px-3 py-2 bg-muted border border-border shadow-none">
                <RefreshCw className="w-4 h-4 text-purple-400 animate-spin" />
              </Card>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      <Separator />

      {/* Input */}
      <div className="px-3 pb-3 pt-2">
        <div className="flex items-end gap-2 bg-muted border border-border rounded-xl px-3 py-2 focus-within:border-purple-500/50 transition-colors">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about this opportunity..."
            rows={3}
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground resize-none focus:outline-none min-h-[68px] max-h-40"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || chatMutation.isPending}
            size="icon"
            variant={input.trim() && !chatMutation.isPending ? "default" : "ghost"}
            className={cn(
              "h-8 w-8 flex-shrink-0",
              input.trim() && !chatMutation.isPending
                ? "bg-purple-500 hover:bg-purple-600"
                : ""
            )}
          >
            <Send className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>
    </div>
  )
}
