import { useState, useCallback, useRef, useMemo, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useAtom } from 'jotai'
import {
  useLocalRuntime,
  AssistantRuntimeProvider,
  type ChatModelAdapter,
  type ThreadMessageLike,
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
} from '@assistant-ui/react'
import { MarkdownTextPrimitive } from '@assistant-ui/react-markdown'
import { useMessage, useMessagePartText } from '@assistant-ui/react'
import { motion } from 'framer-motion'
import {
  encodeSeg,
  SegmentedContent,
  RichTextContent,
  markdownComponents,
} from './ChatRendering'
import {
  MessageSquare,
  Plus,
  Send,
  Trash2,
  Bot,
  User,
  Pencil,
  Check,
  X,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  ArrowDown,
  RefreshCw,
} from 'lucide-react'
import { cn } from '../../lib/utils'
import { Button } from '../ui/button'
import {
  listAIChatSessions,
  archiveAIChatSession,
  renameChatSession,
  getAIChatSession,
  streamAIChat,
  type AIChatSession,
  type AIChatSessionDetail,
  type ChatStreamEvent,
} from '../../services/api'
import { activeChatSessionIdAtom } from '../../store/atoms'

type SessionGroupKey = 'today' | 'yesterday' | 'thisWeek' | 'older'

function groupSessionsByDate(sessions: AIChatSession[]): Record<SessionGroupKey, AIChatSession[]> {
  const groups: Record<SessionGroupKey, AIChatSession[]> = { today: [], yesterday: [], thisWeek: [], older: [] }
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const yesterday = new Date(today.getTime() - 86400000)
  const weekAgo = new Date(today.getTime() - 7 * 86400000)

  for (const session of sessions) {
    const date = session.updated_at ? new Date(session.updated_at) : session.created_at ? new Date(session.created_at) : null
    let group: SessionGroupKey
    if (!date) {
      group = 'older'
    } else if (date >= today) {
      group = 'today'
    } else if (date >= yesterday) {
      group = 'yesterday'
    } else if (date >= weekAgo) {
      group = 'thisWeek'
    } else {
      group = 'older'
    }
    groups[group].push(session)
  }

  return groups
}

function toThreadMessages(messages: AIChatSessionDetail['messages'] | undefined): ThreadMessageLike[] {
  return (messages ?? [])
    .filter((message) => message.role === 'user' || message.role === 'assistant')
    .map((message) => ({
      role: message.role,
      content: [{ type: 'text' as const, text: message.content }],
    }))
}

function getSessionSortValue(session: AIChatSession): number {
  const timestamp = session.updated_at ?? session.created_at
  if (!timestamp) return 0
  const value = Date.parse(timestamp)
  return Number.isFinite(value) ? value : 0
}

function upsertSessionList(sessions: AIChatSession[], session: AIChatSession): AIChatSession[] {
  return [...sessions.filter((item) => item.session_id !== session.session_id), session]
    .sort((left, right) => getSessionSortValue(right) - getSessionSortValue(left))
}

function coerceSessionPayload(data: Record<string, unknown>): AIChatSession | null {
  const sessionId = typeof data.session_id === 'string' ? data.session_id : null
  if (!sessionId) return null

  return {
    session_id: sessionId,
    context_type: typeof data.context_type === 'string' ? data.context_type : null,
    context_id: typeof data.context_id === 'string' ? data.context_id : null,
    title: typeof data.title === 'string' ? data.title : null,
    created_at: typeof data.created_at === 'string' ? data.created_at : null,
    updated_at: typeof data.updated_at === 'string' ? data.updated_at : null,
  }
}

type ChatPaneState = {
  paneKey: string
  sessionId: string | null
  initialMessages: readonly ThreadMessageLike[]
}

function SessionSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onArchiveSession,
  collapsed,
  onToggleCollapse,
}: {
  sessions: AIChatSession[]
  activeSessionId: string | null
  onSelectSession: (id: string) => void
  onNewChat: () => void
  onArchiveSession: (id: string) => void
  collapsed: boolean
  onToggleCollapse: () => void
}) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const grouped = useMemo(() => groupSessionsByDate(sessions), [sessions])
  const groupOrder: Array<{ key: SessionGroupKey; label: string }> = [
    { key: 'today', label: t('ai.chatView.groupToday') },
    { key: 'yesterday', label: t('ai.chatView.groupYesterday') },
    { key: 'thisWeek', label: t('ai.chatView.groupThisWeek') },
    { key: 'older', label: t('ai.chatView.groupOlder') },
  ]

  const handleArchive = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    await archiveAIChatSession(sessionId)
    queryClient.setQueryData<{ sessions: AIChatSession[]; total: number } | undefined>(['ai-chat-sessions'], (current) => {
      if (!current) return current
      const nextSessions = current.sessions.filter((session) => session.session_id !== sessionId)
      const removed = nextSessions.length !== current.sessions.length
      return {
        sessions: nextSessions,
        total: removed ? Math.max(0, current.total - 1) : current.total,
      }
    })
    queryClient.invalidateQueries({ queryKey: ['ai-chat-sessions'] })
    onArchiveSession(sessionId)
  }

  const startRename = (session: AIChatSession, e: React.MouseEvent) => {
    e.stopPropagation()
    setRenamingId(session.session_id)
    setRenameValue(session.title || '')
  }

  const commitRename = async (sessionId: string) => {
    const trimmed = renameValue.trim()
    if (trimmed) {
      await renameChatSession(sessionId, trimmed)
      queryClient.invalidateQueries({ queryKey: ['ai-chat-sessions'] })
    }
    setRenamingId(null)
  }

  if (collapsed) {
    return (
      <div className="w-10 border-r border-border/40 flex flex-col items-center py-3 gap-2 shrink-0">
        <button
          onClick={onToggleCollapse}
          className="p-1.5 rounded-md hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
        <button
          onClick={onNewChat}
          className="p-1.5 rounded-md hover:bg-purple-500/20 text-muted-foreground hover:text-purple-700 dark:text-purple-300 transition-colors"
        >
          <Plus className="w-4 h-4" />
        </button>
      </div>
    )
  }

  return (
    <div className="w-64 border-r border-border/40 flex flex-col shrink-0 min-w-0 max-w-64 overflow-hidden">
      <div className="p-3 flex items-center justify-between border-b border-border/20">
        <Button
          size="sm"
          onClick={onNewChat}
          className="h-8 gap-1.5 text-xs bg-purple-500/20 text-purple-700 dark:text-purple-300 border border-purple-500/30 hover:bg-purple-500/30"
        >
          <Plus className="w-3.5 h-3.5" />
          {t('ai.chatView.newChat')}
        </Button>
        <button
          onClick={onToggleCollapse}
          className="p-1.5 rounded-md hover:bg-muted/60 text-muted-foreground hover:text-foreground transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        <div className="p-2 space-y-3">
          {groupOrder.map(({ key, label }) => {
            const items = grouped[key]
            if (!items?.length) return null
            return (
              <div key={key}>
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground/60 px-2 mb-1.5">{label}</p>
                <div className="space-y-0.5">
                  {items.map((session) => (
                    <div
                      key={session.session_id}
                      onClick={() => onSelectSession(session.session_id)}
                      className={cn(
                        'group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-colors min-w-0 max-w-full overflow-hidden',
                        activeSessionId === session.session_id
                          ? 'bg-purple-500/15 text-purple-700 dark:text-purple-200'
                          : 'hover:bg-muted/40 text-muted-foreground hover:text-foreground',
                      )}
                    >
                      <MessageSquare className="w-3.5 h-3.5 shrink-0 opacity-60" />
                      {renamingId === session.session_id ? (
                        <div className="flex-1 flex items-center gap-1">
                          <input
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') commitRename(session.session_id)
                              if (e.key === 'Escape') setRenamingId(null)
                            }}
                            onClick={(e) => e.stopPropagation()}
                            autoFocus
                            className="flex-1 bg-transparent border-b border-purple-500/40 text-xs outline-none px-0.5"
                          />
                          <button onClick={(e) => { e.stopPropagation(); void commitRename(session.session_id) }}>
                            <Check className="w-3 h-3 text-emerald-400" />
                          </button>
                          <button onClick={(e) => { e.stopPropagation(); setRenamingId(null) }}>
                            <X className="w-3 h-3 text-muted-foreground" />
                          </button>
                        </div>
                      ) : (
                        <>
                          <span className="flex-1 text-xs truncate">
                            {session.title || t('ai.chatView.chatNumber', { id: session.session_id.slice(0, 8) })}
                          </span>
                          <div className="hidden group-hover:flex items-center gap-0.5">
                            <button
                              onClick={(e) => startRename(session, e)}
                              className="p-0.5 rounded hover:bg-muted/60"
                            >
                              <Pencil className="w-3 h-3" />
                            </button>
                            <button
                              onClick={(e) => { void handleArchive(session.session_id, e) }}
                              className="p-0.5 rounded hover:bg-red-500/20 text-red-400"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
          {sessions.length === 0 && (
            <div className="text-center py-8">
              <MessageSquare className="w-8 h-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-xs text-muted-foreground/50">{t('ai.chatView.noConversations')}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function UserMessage() {
  const { t } = useTranslation()
  return (
    <MessagePrimitive.Root className="flex gap-3 py-4 px-4 border-b border-border/20">
      <div className="w-7 h-7 rounded-full bg-blue-500/20 border border-blue-500/30 flex items-center justify-center shrink-0 mt-0.5">
        <User className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[10px] uppercase tracking-wider text-blue-600 dark:text-blue-400/70 mb-1">{t('ai.chatView.you')}</p>
        <MessagePrimitive.Parts
          components={{
            Text: () => (
              <div className="text-sm text-foreground/90 leading-relaxed whitespace-pre-wrap">
                <MarkdownTextPrimitive />
              </div>
            ),
          }}
        />
      </div>
    </MessagePrimitive.Root>
  )
}

function AssistantTextContent() {
  const part = useMessagePartText()
  const raw = part.text
  const isStreaming = useMessage((state) => state.status?.type === 'running')

  if (raw.includes('\u00AB\u00ABSEG:')) {
    return <SegmentedContent raw={raw} isStreaming={isStreaming} />
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.15 }}
    >
      <RichTextContent text={raw} standalone={false} isStreaming={isStreaming} contextMarkdown={<MarkdownTextPrimitive components={markdownComponents} />} />
    </motion.div>
  )
}

function AssistantMessage() {
  const { t } = useTranslation()
  return (
    <MessagePrimitive.Root className="flex gap-3 py-4 px-4 bg-muted/20 border-b border-border/20">
      <div className="w-7 h-7 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center shrink-0 mt-0.5">
        <Bot className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[10px] uppercase tracking-wider text-purple-600 dark:text-purple-400/70 mb-1">{t('ai.chatView.homerunAi')}</p>
        <MessagePrimitive.Parts
          components={{
            Text: AssistantTextContent,
          }}
        />
      </div>
    </MessagePrimitive.Root>
  )
}

function ChatThread({ autoFocus }: { autoFocus: boolean }) {
  const { t } = useTranslation()
  return (
    <ThreadPrimitive.Root className="flex flex-col h-full border-l border-border/20">
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto border-b border-border/20">
        <ThreadPrimitive.Empty>
          <WelcomeScreen />
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages
          components={{
            UserMessage,
            AssistantMessage,
          }}
        />
        <ThreadPrimitive.ViewportFooter className="sticky bottom-0 flex justify-center pb-2">
          <ThreadPrimitive.ScrollToBottom asChild>
            <button className="p-2 rounded-full bg-purple-500/20 border border-purple-500/30 text-purple-700 dark:text-purple-300 hover:bg-purple-500/30 transition-colors shadow-lg opacity-0 data-[state=visible]:opacity-100 transition-opacity duration-200">
              <ArrowDown className="w-4 h-4" />
            </button>
          </ThreadPrimitive.ScrollToBottom>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>

      <div className="border-t border-border/40 p-4 bg-background/50">
        <ComposerPrimitive.Root className="flex items-end gap-2 rounded-xl border border-border/40 bg-muted/30 p-2 focus-within:border-purple-500/40 transition-colors">
          <ComposerPrimitive.Input
            placeholder={t('ai.chatView.askPlaceholder')}
            className="flex-1 bg-transparent text-sm text-foreground resize-none outline-none min-h-[36px] max-h-[120px] px-2 py-1.5 placeholder:text-muted-foreground/50"
            autoFocus={autoFocus}
          />
          <ComposerPrimitive.Send asChild>
            <Button
              size="sm"
              className="h-8 w-8 p-0 bg-purple-500 hover:bg-purple-600 text-white rounded-lg shrink-0"
            >
              <Send className="w-3.5 h-3.5" />
            </Button>
          </ComposerPrimitive.Send>
        </ComposerPrimitive.Root>
        <p className="text-[10px] text-muted-foreground/40 text-center mt-2">
          {t('ai.chatView.disclaimer')}
        </p>
      </div>
    </ThreadPrimitive.Root>
  )
}

function LoadingChatPane() {
  return (
    <div className="flex-1 flex items-center justify-center border-l border-border/20">
      <RefreshCw className="w-5 h-5 animate-spin text-purple-400" />
    </div>
  )
}

function WelcomeScreen() {
  const { t } = useTranslation()
  const suggestions = [
    t('ai.chatView.suggestAnalyzeOpps'),
    t('ai.chatView.suggestExplainRisks'),
    t('ai.chatView.suggestCompareSpreads'),
    t('ai.chatView.suggestReviewPortfolio'),
  ]

  return (
    <div className="flex flex-col items-center justify-center h-full px-8 py-12">
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/20 flex items-center justify-center mb-6">
        <Sparkles className="w-8 h-8 text-purple-600 dark:text-purple-400" />
      </div>
      <h2 className="text-xl font-semibold text-foreground mb-2">{t('ai.chatView.homerunAi')}</h2>
      <p className="text-sm text-muted-foreground text-center max-w-md mb-8">
        {t('ai.chatView.welcomeSubtitle')}
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
        {suggestions.map((suggestion) => (
          <ThreadPrimitive.Suggestion
            key={suggestion}
            prompt={suggestion}
            method="replace"
            autoSend
            asChild
          >
            <button className="flex items-center justify-center px-4 py-3 rounded-xl border border-border/40 bg-muted/30 hover:bg-muted/50 hover:border-purple-500/20 transition-colors text-center">
              <span className="text-sm text-muted-foreground">{suggestion}</span>
            </button>
          </ThreadPrimitive.Suggestion>
        ))}
      </div>
    </div>
  )
}

function ChatRuntime({
  paneKey,
  sessionId,
  initialMessages,
  isActive,
  onSessionBound,
}: {
  paneKey: string
  sessionId: string | null
  initialMessages: readonly ThreadMessageLike[]
  isActive: boolean
  onSessionBound: (paneKey: string, session: AIChatSession) => void
}) {
  const queryClient = useQueryClient()
  const sessionIdRef = useRef<string | null>(sessionId)
  sessionIdRef.current = sessionId

  const chatModelAdapter = useMemo<ChatModelAdapter>(() => ({
    async *run({ messages, abortSignal }) {
      const lastUserMessage = [...messages].reverse().find((message) => message.role === 'user')
      if (!lastUserMessage) return

      const userText = lastUserMessage.content
        .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
        .map((part) => part.text)
        .join('\n')

      const state = {
        segments: [] as string[],
        answerChunks: [] as string[],
        isThinking: true,
        done: false,
        error: null as string | null,
        sessionId: sessionIdRef.current,
        changed: true,
      }

      const THINKING_SEG = encodeSeg('thinking', { content: '' })

      streamAIChat(
        {
          message: userText,
          session_id: sessionIdRef.current || undefined,
        },
        (chunk) => {
          state.isThinking = false
          state.answerChunks.push(chunk)
          state.changed = true
        },
        (data) => {
          if (data.session_id && data.session_id !== state.sessionId) {
            state.sessionId = data.session_id
            onSessionBound(paneKey, {
              session_id: data.session_id,
              context_type: null,
              context_id: null,
              title: null,
              created_at: null,
              updated_at: new Date().toISOString(),
            })
          }
          state.isThinking = false
          state.done = true
          state.changed = true
        },
        (error) => {
          state.isThinking = false
          state.error = error
          state.changed = true
        },
        abortSignal,
        (event: ChatStreamEvent) => {
          switch (event.event) {
            case 'session': {
              const boundSession = coerceSessionPayload(event.data)
              if (!boundSession) break
              state.sessionId = boundSession.session_id
              onSessionBound(paneKey, boundSession)
              break
            }
            case 'thinking':
              state.isThinking = true
              state.changed = true
              break
            case 'tool_start':
              state.isThinking = false
              state.segments.push(encodeSeg('tool_start', {
                tool: event.data.tool,
                input: event.data.input || {},
              }))
              state.changed = true
              break
            case 'tool_end':
              state.segments.push(encodeSeg('tool_end', {
                tool: event.data.tool,
                output: event.data.output || {},
              }))
              state.isThinking = false
              state.changed = true
              break
            case 'tool_error':
              state.segments.push(encodeSeg('tool_error', {
                tool: event.data.tool,
                error: event.data.error || 'Unknown error',
              }))
              state.changed = true
              break
          }
        },
      )

      let lastYielded = ''
      const startTime = Date.now()
      const maxWait = 180_000

      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 40))

        if (state.error) {
          const errorDisplay = state.segments.join('') + `\n\n**Error:** ${state.error}`
          yield { content: [{ type: 'text' as const, text: errorDisplay }] }
          return
        }

        if (state.changed) {
          state.changed = false
          const answerSoFar = state.answerChunks.join('')
          const thinkingPrefix = state.isThinking ? THINKING_SEG : ''
          const display = thinkingPrefix + state.segments.join('') + answerSoFar

          if (display && display !== lastYielded) {
            lastYielded = display
            yield { content: [{ type: 'text' as const, text: display }] }
          }
        }

        if (state.done) {
          const finalAnswer = state.answerChunks.join('') || 'No response generated.'
          const finalDisplay = state.segments.join('') + finalAnswer
          yield { content: [{ type: 'text' as const, text: finalDisplay }] }
          queryClient.invalidateQueries({ queryKey: ['ai-chat-sessions'] })
          return
        }

        if (Date.now() - startTime > maxWait) {
          yield { content: [{ type: 'text' as const, text: state.segments.join('') + '*Request timed out.*' }] }
          return
        }
      }
    },
  }), [onSessionBound, paneKey, queryClient])

  const runtime = useLocalRuntime(chatModelAdapter, { initialMessages })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatThread autoFocus={isActive} />
    </AssistantRuntimeProvider>
  )
}

export default function AIChatView() {
  const queryClient = useQueryClient()
  const [activeSessionId, setActiveSessionId] = useAtom(activeChatSessionIdAtom)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [panes, setPanes] = useState<ChatPaneState[]>([])
  const [activePaneKey, setActivePaneKey] = useState<string | null>(null)
  const [isLoadingSession, setIsLoadingSession] = useState(false)
  const didAutoSelect = useRef(false)
  const draftCounterRef = useRef(0)
  const activePaneKeyRef = useRef<string | null>(null)
  const activeSessionIdRef = useRef<string | null>(activeSessionId)
  const sessionPaneKeysRef = useRef<Record<string, string>>({})
  const selectionRequestRef = useRef(0)

  useEffect(() => {
    activePaneKeyRef.current = activePaneKey
  }, [activePaneKey])

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId
  }, [activeSessionId])

  const { data: sessionsData } = useQuery({
    queryKey: ['ai-chat-sessions'],
    queryFn: () => listAIChatSessions({ limit: 100 }),
    refetchInterval: 15000,
  })
  const sessions = sessionsData?.sessions ?? []

  const createDraftPane = useCallback((): ChatPaneState => {
    draftCounterRef.current += 1
    return {
      paneKey: `draft:${draftCounterRef.current}`,
      sessionId: null,
      initialMessages: [],
    }
  }, [])

  const handleNewChat = useCallback(() => {
    selectionRequestRef.current += 1
    const draftPane = createDraftPane()
    setIsLoadingSession(false)
    setPanes((current) => [...current, draftPane])
    setActivePaneKey(draftPane.paneKey)
    setActiveSessionId(null)
  }, [createDraftPane, setActiveSessionId])

  const handleSelectSession = useCallback(async (sessionId: string) => {
    const requestId = ++selectionRequestRef.current
    const existingPaneKey = sessionPaneKeysRef.current[sessionId]
    const previousPaneKey = activePaneKeyRef.current
    const previousSessionId = activeSessionIdRef.current

    setActiveSessionId(sessionId)

    if (existingPaneKey) {
      setIsLoadingSession(false)
      setActivePaneKey(existingPaneKey)
      return
    }

    setIsLoadingSession(true)
    setActivePaneKey(null)

    try {
      const detail = await getAIChatSession(sessionId)
      if (selectionRequestRef.current !== requestId) return

      const paneKey = `session:${sessionId}`
      sessionPaneKeysRef.current[sessionId] = paneKey
      setPanes((current) => {
        if (current.some((pane) => pane.paneKey === paneKey)) {
          return current
        }
        return [...current, { paneKey, sessionId, initialMessages: toThreadMessages(detail.messages) }]
      })
      setActivePaneKey(paneKey)
    } catch {
      if (selectionRequestRef.current !== requestId) return

      setActiveSessionId(previousSessionId)
      if (previousPaneKey) {
        setActivePaneKey(previousPaneKey)
      } else {
        const draftPane = createDraftPane()
        setPanes((current) => [...current, draftPane])
        setActivePaneKey(draftPane.paneKey)
      }
    } finally {
      if (selectionRequestRef.current === requestId) {
        setIsLoadingSession(false)
      }
    }
  }, [createDraftPane, setActiveSessionId])

  const handleArchiveSession = useCallback((sessionId: string) => {
    const removedPaneKey = sessionPaneKeysRef.current[sessionId]
    delete sessionPaneKeysRef.current[sessionId]

    const shouldOpenDraft = activePaneKeyRef.current === removedPaneKey || activeSessionIdRef.current === sessionId
    const nextDraftPane = shouldOpenDraft ? createDraftPane() : null

    if (shouldOpenDraft) {
      selectionRequestRef.current += 1
      setIsLoadingSession(false)
    }

    setPanes((current) => {
      const filtered = current.filter((pane) => pane.sessionId !== sessionId)
      return nextDraftPane ? [...filtered, nextDraftPane] : filtered
    })

    if (nextDraftPane) {
      setActivePaneKey(nextDraftPane.paneKey)
      setActiveSessionId(null)
    }
  }, [createDraftPane, setActiveSessionId])

  const handleSessionBound = useCallback((paneKey: string, session: AIChatSession) => {
    const normalizedSession: AIChatSession = {
      ...session,
      updated_at: session.updated_at ?? session.created_at ?? new Date().toISOString(),
    }

    sessionPaneKeysRef.current[normalizedSession.session_id] = paneKey
    setPanes((current) => current.map((pane) => (
      pane.paneKey === paneKey
        ? { ...pane, sessionId: normalizedSession.session_id }
        : pane
    )))

    queryClient.setQueryData<{ sessions: AIChatSession[]; total: number } | undefined>(['ai-chat-sessions'], (current) => {
      if (!current) {
        return { sessions: [normalizedSession], total: 1 }
      }

      const exists = current.sessions.some((item) => item.session_id === normalizedSession.session_id)
      return {
        sessions: upsertSessionList(current.sessions, normalizedSession),
        total: exists ? current.total : current.total + 1,
      }
    })

    if (activePaneKeyRef.current === paneKey) {
      setActiveSessionId(normalizedSession.session_id)
    }

    queryClient.invalidateQueries({ queryKey: ['ai-chat-sessions'] })
  }, [queryClient, setActiveSessionId])

  useEffect(() => {
    if (didAutoSelect.current) return

    const sessionToLoad = activeSessionId || sessions[0]?.session_id
    if (!sessionToLoad) {
      if (sessionsData) {
        didAutoSelect.current = true
        if (panes.length === 0) {
          handleNewChat()
        }
      }
      return
    }

    didAutoSelect.current = true
    void handleSelectSession(sessionToLoad)
  }, [activeSessionId, handleNewChat, handleSelectSession, panes.length, sessions, sessionsData])

  return (
    <div className="flex h-full">
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(sessionId) => { void handleSelectSession(sessionId) }}
        onNewChat={handleNewChat}
        onArchiveSession={handleArchiveSession}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <div className="flex-1 flex flex-col min-w-0">
        {panes.map((pane) => (
          <div key={pane.paneKey} className={cn('flex-1 min-h-0', activePaneKey !== pane.paneKey && 'hidden')}>
            <ChatRuntime
              paneKey={pane.paneKey}
              sessionId={pane.sessionId}
              initialMessages={pane.initialMessages}
              isActive={activePaneKey === pane.paneKey}
              onSessionBound={handleSessionBound}
            />
          </div>
        ))}
        {(isLoadingSession || (!activePaneKey && panes.length === 0)) && <LoadingChatPane />}
      </div>
    </div>
  )
}
