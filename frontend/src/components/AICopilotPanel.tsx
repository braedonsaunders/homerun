import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { useQueryClient, useQuery } from '@tanstack/react-query'
import { useAtom, useSetAtom } from 'jotai'
import { useTranslation } from 'react-i18next'
import {
  X,
  Send,
  Bot,
  User,
  Minimize2,
  Maximize2,
  Trash2,
  Sparkles,
  ExternalLink,
  ChevronRight,
  ChevronLeft,
  Loader2,
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  AIChatMessage,
  getAIChatSession,
  archiveAIChatSession,
  listAIChatSessions,
  AIChatSession,
} from '../services/api'
import { activeChatSessionIdAtom, aiTabSubtabAtom } from '../store/atoms'
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
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [messages, setMessages] = useState<AIChatMessage[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [isExpanded, setIsExpanded] = useState(false)
  const [isSessionReady, setIsSessionReady] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [showSessionPills, setShowSessionPills] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const lastSeedPromptIdRef = useRef<number | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [activeChatSessionId, setActiveChatSessionId] = useAtom(activeChatSessionIdAtom)
  const setAiTabSubtab = useSetAtom(aiTabSubtabAtom)

  const sessionStorageKey = useMemo(() => {
    const ctxType = contextType || 'general'
    const ctxId = contextId || 'default'
    return `ai-copilot-session:${ctxType}:${ctxId}`
  }, [contextType, contextId])

  const { data: recentSessions = [] } = useQuery({
    queryKey: ['ai-chat-sessions-recent', 5],
    queryFn: async () => {
      const result = await listAIChatSessions({ limit: 5 })
      return result.sessions
    },
    enabled: isOpen,
    staleTime: 10_000,
  })

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingContent, scrollToBottom])

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  const loadSession = useCallback(async (loadSessionId: string) => {
    try {
      const data = await getAIChatSession(loadSessionId)
      setSessionId(data.session_id)
      setActiveChatSessionId(data.session_id)
      const restored = (data.messages || [])
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content }))
      setMessages(restored)
      setIsSessionReady(true)
    } catch {
      setSessionId(null)
      setActiveChatSessionId(null)
      setMessages([])
      setIsSessionReady(true)
    }
  }, [setActiveChatSessionId])

  // Load session on open — from localStorage key or activeChatSessionId
  useEffect(() => {
    if (!isOpen) {
      setIsSessionReady(false)
      return
    }
    let cancelled = false
    setIsSessionReady(false)

    const existingSessionId = window.localStorage.getItem(sessionStorageKey)
    const targetSessionId = existingSessionId || activeChatSessionId

    if (!targetSessionId) {
      setSessionId(null)
      setMessages([])
      setIsSessionReady(true)
      return
    }

    setMessages([])
    getAIChatSession(targetSessionId)
      .then((data) => {
        if (cancelled) return
        setSessionId(data.session_id)
        setActiveChatSessionId(data.session_id)
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
        setActiveChatSessionId(null)
        setMessages([])
        setIsSessionReady(true)
      })

    return () => {
      cancelled = true
    }
  }, [isOpen, sessionStorageKey, activeChatSessionId, setActiveChatSessionId])

  // React to external session changes (e.g., AI tab switched session)
  useEffect(() => {
    if (!isOpen || !isSessionReady || !activeChatSessionId) return
    if (activeChatSessionId === sessionId) return
    loadSession(activeChatSessionId)
  }, [isOpen, isSessionReady, activeChatSessionId, sessionId, loadSession])

  const sendStreamingMessage = useCallback(async (message: string) => {
    setIsStreaming(true)
    setStreamingContent('')

    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      const response = await fetch('/api/ai/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          session_id: sessionId || undefined,
          context_type: contextType,
          context_id: contextId,
          allow_actions: true,
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error')
        let detail = errorText
        try {
          const parsed = JSON.parse(errorText)
          detail = parsed.detail || errorText
        } catch { /* not JSON */ }
        throw new Error(detail)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''
      let accumulated = ''
      let receivedSessionId: string | null = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        // Keep the last potentially incomplete line in buffer
        buffer = lines.pop() || ''

        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ') && currentEvent) {
            const jsonStr = line.slice(6)
            try {
              const payload = JSON.parse(jsonStr)
              if (currentEvent === 'token') {
                accumulated += payload.text || ''
                setStreamingContent(accumulated)
              } else if (currentEvent === 'done') {
                receivedSessionId = payload.session_id || null
              } else if (currentEvent === 'error') {
                throw new Error(payload.error || 'Stream error')
              }
            } catch (parseErr) {
              if (parseErr instanceof SyntaxError) continue
              throw parseErr
            }
            currentEvent = ''
          } else if (line.trim() === '') {
            currentEvent = ''
          }
        }
      }

      // Finalize: add the completed assistant message
      const finalContent = accumulated || t('aiCopilotPanel.noResponse')
      setMessages((prev) => [...prev, { role: 'assistant', content: finalContent }])
      setStreamingContent('')

      if (receivedSessionId) {
        setSessionId(receivedSessionId)
        setActiveChatSessionId(receivedSessionId)
        window.localStorage.setItem(sessionStorageKey, receivedSessionId)
      }

      // Invalidate session list so pills update
      queryClient.invalidateQueries({ queryKey: ['ai-chat-sessions-recent'] })
    } catch (err: any) {
      if (err?.name === 'AbortError') return
      setStreamingContent('')
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: t('aiCopilotPanel.errorPrefix', { message: err?.message || t('aiCopilotPanel.failedToGetResponse') }),
        },
      ])
    } finally {
      setIsStreaming(false)
      abortControllerRef.current = null
    }
  }, [t, sessionId, contextType, contextId, sessionStorageKey, setActiveChatSessionId, queryClient])

  const handleSend = useCallback(() => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }])
    setInput('')
    sendStreamingMessage(trimmed)
  }, [input, isStreaming, sendStreamingMessage])

  useEffect(() => {
    if (!isOpen || !isSessionReady || !seedPrompt) return
    if (lastSeedPromptIdRef.current === seedPrompt.id) return
    lastSeedPromptIdRef.current = seedPrompt.id

    const prompt = String(seedPrompt.prompt || '').trim()
    if (!prompt) return

    if (!seedPrompt.autoSend || isStreaming) {
      setInput(prompt)
      setTimeout(() => inputRef.current?.focus(), 40)
      return
    }

    setMessages((prev) => [...prev, { role: 'user', content: prompt }])
    sendStreamingMessage(prompt)
  }, [isOpen, isSessionReady, seedPrompt, isStreaming, sendStreamingMessage])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleOpenInAITab = useCallback(() => {
    setAiTabSubtab('chat')
    window.dispatchEvent(new CustomEvent('navigate-to-tab', { detail: 'ai' }))
    onClose()
  }, [setAiTabSubtab, onClose])

  const handleClearChat = useCallback(async () => {
    if (sessionId) {
      try {
        await archiveAIChatSession(sessionId)
      } catch {
        // ignore archive failures; local clear still proceeds
      }
    }
    setMessages([])
    setSessionId(null)
    setActiveChatSessionId(null)
    setStreamingContent('')
    window.localStorage.removeItem(sessionStorageKey)
    queryClient.invalidateQueries({ queryKey: ['ai-chat-sessions-recent'] })
  }, [sessionId, sessionStorageKey, setActiveChatSessionId, queryClient])

  const handleSwitchSession = useCallback((session: AIChatSession) => {
    // Abort any active stream
    abortControllerRef.current?.abort()
    setIsStreaming(false)
    setStreamingContent('')
    setActiveChatSessionId(session.session_id)
    loadSession(session.session_id)
  }, [setActiveChatSessionId, loadSession])

  const quickActions = useMemo(() => {
    if (contextType === 'strategy') {
      return [
        { label: t('aiCopilotPanel.quickActions.strategy.explainCode.label'), prompt: t('aiCopilotPanel.quickActions.strategy.explainCode.prompt') },
        { label: t('aiCopilotPanel.quickActions.strategy.improveEvaluate.label'), prompt: t('aiCopilotPanel.quickActions.strategy.improveEvaluate.prompt') },
        { label: t('aiCopilotPanel.quickActions.strategy.addFeature.label'), prompt: t('aiCopilotPanel.quickActions.strategy.addFeature.prompt') },
        { label: t('aiCopilotPanel.quickActions.strategy.reviewSafety.label'), prompt: t('aiCopilotPanel.quickActions.strategy.reviewSafety.prompt') },
      ]
    }
    if (contextType === 'data_source') {
      return [
        { label: t('aiCopilotPanel.quickActions.dataSource.explainSource.label'), prompt: t('aiCopilotPanel.quickActions.dataSource.explainSource.prompt') },
        { label: t('aiCopilotPanel.quickActions.dataSource.hardenParsing.label'), prompt: t('aiCopilotPanel.quickActions.dataSource.hardenParsing.prompt') },
        { label: t('aiCopilotPanel.quickActions.dataSource.addGeotags.label'), prompt: t('aiCopilotPanel.quickActions.dataSource.addGeotags.prompt') },
        { label: t('aiCopilotPanel.quickActions.dataSource.retentionReview.label'), prompt: t('aiCopilotPanel.quickActions.dataSource.retentionReview.prompt') },
      ]
    }
    return [
      { label: t('aiCopilotPanel.quickActions.default.analyzeRisk.label'), prompt: t('aiCopilotPanel.quickActions.default.analyzeRisk.prompt') },
      { label: t('aiCopilotPanel.quickActions.default.resolutionSafety.label'), prompt: t('aiCopilotPanel.quickActions.default.resolutionSafety.prompt') },
      { label: t('aiCopilotPanel.quickActions.default.shouldITrade.label'), prompt: t('aiCopilotPanel.quickActions.default.shouldITrade.prompt') },
      { label: t('aiCopilotPanel.quickActions.default.explainStrategy.label'), prompt: t('aiCopilotPanel.quickActions.default.explainStrategy.prompt') },
    ]
  }, [t, contextType])

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

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
            <h3 className="text-sm font-semibold text-foreground">{t('aiCopilotPanel.title')}</h3>
            {contextLabel && (
              <p className="text-[10px] text-purple-400 truncate max-w-[200px]">
                {contextLabel}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            onClick={handleOpenInAITab}
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            title={t('aiCopilotPanel.openInAiTab')}
          >
            <ExternalLink className="w-3.5 h-3.5 text-muted-foreground" />
          </Button>
          {messages.length > 0 && (
            <Button
              onClick={handleClearChat}
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              title={t('aiCopilotPanel.clearChat')}
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

      {/* Session pills */}
      {recentSessions.length > 0 && (
        <div className="px-3 py-1.5 flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-5 w-5 flex-shrink-0"
            onClick={() => setShowSessionPills(!showSessionPills)}
          >
            {showSessionPills ? (
              <ChevronLeft className="w-3 h-3 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-3 h-3 text-muted-foreground" />
            )}
          </Button>
          {showSessionPills && (
            <div className="flex-1 overflow-x-auto flex gap-1 scrollbar-none">
              {recentSessions.map((s) => (
                <button
                  key={s.session_id}
                  onClick={() => handleSwitchSession(s)}
                  className={cn(
                    'flex-shrink-0 px-2 py-0.5 rounded-full text-[10px] border transition-colors truncate max-w-[120px]',
                    s.session_id === sessionId
                      ? 'border-purple-500/50 bg-purple-500/10 text-purple-300'
                      : 'border-border bg-muted/50 text-muted-foreground hover:border-purple-500/30 hover:text-foreground'
                  )}
                  title={s.title || s.session_id}
                >
                  {s.title || s.context_type || t('aiCopilotPanel.chatLabel')}
                </button>
              ))}
            </div>
          )}
          {!showSessionPills && (
            <span className="text-[10px] text-muted-foreground">
              {recentSessions.length === 1
                ? t('aiCopilotPanel.sessionsCountOne', { count: recentSessions.length })
                : t('aiCopilotPanel.sessionsCountOther', { count: recentSessions.length })}
            </span>
          )}
        </div>
      )}

      {/* Messages */}
      <ScrollArea className="flex-1">
        <div className="px-4 py-3 space-y-3">
          {messages.length === 0 && !isStreaming && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center mb-3">
                <Bot className="w-6 h-6 text-purple-400" />
              </div>
              <p className="text-sm text-muted-foreground mb-1">
                {contextType === 'strategy'
                  ? t('aiCopilotPanel.welcomeTitle.strategy')
                  : contextType === 'data_source'
                    ? t('aiCopilotPanel.welcomeTitle.dataSource')
                    : t('aiCopilotPanel.welcomeTitle.default')}
              </p>
              <p className="text-xs text-muted-foreground mb-4">
                {contextType === 'strategy'
                  ? t('aiCopilotPanel.welcomeSubtitle.strategy')
                  : contextType === 'data_source'
                    ? t('aiCopilotPanel.welcomeSubtitle.dataSource')
                    : t('aiCopilotPanel.welcomeSubtitle.default')}
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
                        sendStreamingMessage(action.prompt)
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

          {/* Streaming response in progress */}
          {isStreaming && (
            <div className="flex gap-2 justify-start">
              <div className="w-6 h-6 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Bot className="w-3.5 h-3.5 text-purple-400" />
              </div>
              <Card className="max-w-[80%] rounded-xl px-3 py-2 text-sm bg-muted text-muted-foreground border border-border shadow-none">
                {streamingContent ? (
                  <p className="whitespace-pre-wrap">{streamingContent}<span className="inline-block w-1.5 h-4 bg-purple-400 animate-pulse ml-0.5 align-text-bottom" /></p>
                ) : (
                  <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />
                )}
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
            placeholder={t('aiCopilotPanel.inputPlaceholder')}
            rows={3}
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground resize-none focus:outline-none min-h-[68px] max-h-40"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            size="icon"
            variant={input.trim() && !isStreaming ? "default" : "ghost"}
            className={cn(
              "h-8 w-8 flex-shrink-0",
              input.trim() && !isStreaming
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
