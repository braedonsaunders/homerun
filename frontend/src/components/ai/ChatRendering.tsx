/**
 * Shared chat rendering utilities for AI Chat and Cortex views.
 *
 * Extracts segment parsing, OpenUI splitting, markdown rendering,
 * and structured message rendering so both views share one code path.
 */

import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import ReactMarkdown from 'react-markdown'
import { Renderer } from '@openuidev/react-lang'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ThinkingIndicator,
  ToolStartWidget,
  ToolResultWidget,
  ToolErrorWidget,
} from './ChatToolWidgets'
import { homerunLibrary } from './openui-library'
import { Loader2, Bot, ChevronRight } from 'lucide-react'
import { cn } from '../../lib/utils'

// ---------------------------------------------------------------------------
// Segment encoding/decoding
// ---------------------------------------------------------------------------

/** Segment delimiter used to encode structured content in the text stream.
 *  Format: ‹‹SEG:type:json›› */
export const SEG_RE = /\u00AB\u00ABSEG:(thinking|tool_start|tool_end|tool_error):([^\u00BB]*)\u00BB\u00BB/g

export type ChatSegment =
  | { type: 'thinking'; text: string }
  | { type: 'tool_start'; tool: string; input: Record<string, unknown> }
  | { type: 'tool_end'; tool: string; output: Record<string, unknown> }
  | { type: 'tool_error'; tool: string; error: string }
  | { type: 'text'; text: string }

export function parseSegments(raw: string): ChatSegment[] {
  const segments: ChatSegment[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  const re = new RegExp(SEG_RE.source, 'g')
  while ((match = re.exec(raw)) !== null) {
    if (match.index > lastIndex) {
      const text = raw.slice(lastIndex, match.index).trim()
      if (text) segments.push({ type: 'text', text })
    }
    const segType = match[1] as 'thinking' | 'tool_start' | 'tool_end' | 'tool_error'
    try {
      const data = JSON.parse(match[2])
      if (segType === 'thinking') {
        segments.push({ type: 'thinking', text: data.content || '' })
      } else if (segType === 'tool_start') {
        segments.push({ type: 'tool_start', tool: data.tool || '', input: data.input || {} })
      } else if (segType === 'tool_end') {
        segments.push({ type: 'tool_end', tool: data.tool || '', output: data.output || {} })
      } else if (segType === 'tool_error') {
        segments.push({ type: 'tool_error', tool: data.tool || '', error: data.error || '' })
      }
    } catch {
      // Skip malformed segments
    }
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < raw.length) {
    const text = raw.slice(lastIndex).trim()
    if (text) segments.push({ type: 'text', text })
  }
  return segments
}

/** Encode a structured segment into the text stream. */
export function encodeSeg(type: string, data: Record<string, unknown>): string {
  return `\u00AB\u00ABSEG:${type}:${JSON.stringify(data)}\u00BB\u00BB`
}

// ---------------------------------------------------------------------------
// OpenUI splitting
// ---------------------------------------------------------------------------

const OPENUI_FENCED_RE = /```openui\n([\s\S]*?)```/g
const OPENUI_RAW_RE = /^root\s*=\s*[A-Z]\w*\(/m

export function splitOpenUI(text: string, streaming = false): { type: 'text' | 'openui'; content: string }[] {
  const parts: { type: 'text' | 'openui'; content: string }[] = []
  let lastIdx = 0
  let m: RegExpExecArray | null

  const fencedRe = new RegExp(OPENUI_FENCED_RE.source, 'g')
  while ((m = fencedRe.exec(text)) !== null) {
    if (m.index > lastIdx) {
      const plain = text.slice(lastIdx, m.index).trim()
      if (plain) parts.push({ type: 'text', content: plain })
    }
    parts.push({ type: 'openui', content: m[1].trim() })
    lastIdx = m.index + m[0].length
  }

  if (parts.length > 0) {
    if (lastIdx < text.length) {
      const tail = text.slice(lastIdx).trim()
      if (tail) {
        if (streaming && /```openui\n/i.test(tail)) {
          const innerMatch = tail.match(/```openui\n([\s\S]*)$/i)
          if (innerMatch) {
            parts.push({ type: 'openui', content: innerMatch[1].trim() })
          } else {
            parts.push({ type: 'text', content: tail })
          }
        } else {
          parts.push({ type: 'text', content: tail })
        }
      }
    }
    return parts
  }

  if (streaming) {
    const incompleteFence = text.match(/^([\s\S]*?)```openui\n([\s\S]*)$/i)
    if (incompleteFence) {
      const before = incompleteFence[1].trim()
      const openUIContent = incompleteFence[2].trim()
      if (before) parts.push({ type: 'text', content: before })
      if (openUIContent) parts.push({ type: 'openui', content: openUIContent })
      return parts.length > 0 ? parts : [{ type: 'text', content: text }]
    }
  }

  const rawBlockRe = new RegExp(
    `((?:^\\w+\\s*=\\s*[A-Z]\\w*\\([\\s\\S]*?\\)\\s*\\n?)+)`,
    'gm'
  )
  while ((m = rawBlockRe.exec(text)) !== null) {
    if (!/^root\s*=/m.test(m[1])) continue
    if (m.index > lastIdx) {
      const plain = text.slice(lastIdx, m.index).trim()
      if (plain) parts.push({ type: 'text', content: plain })
    }
    parts.push({ type: 'openui', content: m[1].trim() })
    lastIdx = m.index + m[0].length
  }

  if (streaming && parts.length === 0 && OPENUI_RAW_RE.test(text)) {
    const rootIdx = text.search(/^root\s*=\s*[A-Z]\w*\(/m)
    if (rootIdx >= 0) {
      const before = text.slice(0, rootIdx).trim()
      const openUIContent = text.slice(rootIdx).trim()
      if (before) parts.push({ type: 'text', content: before })
      parts.push({ type: 'openui', content: openUIContent })
      return parts
    }
  }

  if (lastIdx < text.length) {
    const plain = text.slice(lastIdx).trim()
    if (plain) parts.push({ type: 'text', content: plain })
  }

  return parts.length > 0 ? parts : [{ type: 'text', content: text }]
}

// ---------------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------------

export const PROSE_CLASSES = "text-sm text-foreground/90 leading-relaxed prose dark:prose-invert prose-sm max-w-none prose-p:my-2 prose-headings:my-3 prose-li:my-0.5 prose-pre:bg-background/80 prose-pre:border prose-pre:border-border/40 prose-code:text-purple-700 dark:prose-code:text-purple-300 prose-code:text-xs"

export const markdownComponents: Record<string, React.ComponentType<any>> = {
  table: ({ children, ...rest }: any) => (
    <div className="my-2 rounded-lg border border-border/30 overflow-x-auto">
      <table className="w-full text-xs" {...rest}>{children}</table>
    </div>
  ),
  thead: ({ children, ...rest }: any) => (
    <thead className="bg-muted/30 border-b border-border/30" {...rest}>{children}</thead>
  ),
  th: ({ children, ...rest }: any) => (
    <th className="px-3 py-1.5 text-left text-[10px] font-medium text-muted-foreground uppercase tracking-wider" {...rest}>{children}</th>
  ),
  td: ({ children, ...rest }: any) => (
    <td className="px-3 py-1.5 text-foreground/80 border-t border-border/10" {...rest}>{children}</td>
  ),
  tr: ({ children, ...rest }: any) => (
    <tr className="hover:bg-muted/10 transition-colors" {...rest}>{children}</tr>
  ),
}

// ---------------------------------------------------------------------------
// Shared rendering components
// ---------------------------------------------------------------------------

/**
 * Renders a text block as markdown + OpenUI mixed content.
 * @param standalone — if true (default), uses ReactMarkdown.
 *                     if false, renders `contextMarkdown` instead (for assistant-ui MarkdownTextPrimitive).
 * @param contextMarkdown — React element to render when standalone=false and no OpenUI blocks exist.
 */
export function RichTextContent({ text, standalone = true, isStreaming = false, contextMarkdown }: {
  text: string
  standalone?: boolean
  isStreaming?: boolean
  contextMarkdown?: React.ReactNode
}) {
  const hasOpenUI = OPENUI_FENCED_RE.test(text) || OPENUI_RAW_RE.test(text)
  OPENUI_FENCED_RE.lastIndex = 0

  const hasPartialOpenUI = isStreaming && !hasOpenUI && /```openui\n/i.test(text)

  if (!hasOpenUI && !hasPartialOpenUI) {
    if (!standalone && contextMarkdown) {
      return <div className={PROSE_CLASSES}>{contextMarkdown}</div>
    }
    return (
      <div className={PROSE_CLASSES}>
        <ReactMarkdown components={markdownComponents}>{text}</ReactMarkdown>
      </div>
    )
  }

  const parts = splitOpenUI(text, isStreaming)
  return (
    <div className="space-y-2">
      {parts.map((part, i) =>
        part.type === 'openui' ? (
          <motion.div
            key={`oui-${i}`}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            <Renderer library={homerunLibrary} response={part.content} isStreaming={isStreaming} />
          </motion.div>
        ) : (
          <div key={`md-${i}`} className={PROSE_CLASSES}>
            <ReactMarkdown components={markdownComponents}>{part.content}</ReactMarkdown>
          </div>
        )
      )}
    </div>
  )
}

export type PairedTool = {
  tool: string
  input: Record<string, unknown>
  result?: { type: 'end'; output: Record<string, unknown> } | { type: 'error'; error: string }
}

/** Pair tool_start segments with their corresponding tool_end/tool_error. */
export function pairToolSegments(segments: ChatSegment[]): (PairedTool | ChatSegment)[] {
  const paired: (PairedTool | ChatSegment)[] = []
  const pendingStarts: PairedTool[] = []

  for (const seg of segments) {
    if (seg.type === 'thinking') continue
    if (seg.type === 'tool_start') {
      const p: PairedTool = { tool: seg.tool, input: seg.input }
      pendingStarts.push(p)
      paired.push(p)
    } else if (seg.type === 'tool_end') {
      const match = [...pendingStarts].reverse().find(p => p.tool === seg.tool && !p.result)
      if (match) {
        match.result = { type: 'end', output: seg.output }
      } else {
        paired.push(seg)
      }
    } else if (seg.type === 'tool_error') {
      const match = [...pendingStarts].reverse().find(p => p.tool === seg.tool && !p.result)
      if (match) {
        match.result = { type: 'error', error: seg.error }
      } else {
        paired.push(seg)
      }
    } else {
      paired.push(seg)
    }
  }

  return paired
}

/** Render paired segments (tool widgets + text) with animations. */
export function SegmentedContent({ raw, isStreaming = false }: { raw: string; isStreaming?: boolean }) {
  const { t } = useTranslation()
  const segments = parseSegments(raw)
  const paired = pairToolSegments(segments)

  const hasVisibleContent = paired.length > 0
  const showThinking = !hasVisibleContent && segments.some(s => s.type === 'thinking')

  const hasUnfinishedTool = paired.some(
    seg => 'input' in seg && 'tool' in seg && !('type' in seg) && !(seg as PairedTool).result
  )
  const showStillWorking = isStreaming && hasVisibleContent && !hasUnfinishedTool

  return (
    <div className="space-y-0.5">
      <AnimatePresence>
        {showThinking && <ThinkingIndicator key="thinking-dots" />}
      </AnimatePresence>
      <AnimatePresence mode="popLayout">
        {paired.map((seg, i) => {
          if ('input' in seg && 'tool' in seg && !('type' in seg)) {
            const p = seg as PairedTool
            if (p.result?.type === 'end') {
              return <ToolResultWidget key={`tool-${i}`} tool={p.tool} output={p.result.output} />
            }
            if (p.result?.type === 'error') {
              return <ToolErrorWidget key={`tool-${i}`} tool={p.tool} error={p.result.error} />
            }
            return <ToolStartWidget key={`tool-${i}`} tool={p.tool} input={p.input} />
          }
          const s = seg as ChatSegment
          switch (s.type) {
            case 'tool_end':
              return <ToolResultWidget key={`te-${i}`} tool={s.tool} output={s.output} />
            case 'tool_error':
              return <ToolErrorWidget key={`err-${i}`} tool={s.tool} error={s.error} />
            case 'text':
              return (
                <motion.div
                  key={`txt-${i}`}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  className="mt-2"
                >
                  <RichTextContent text={s.text} standalone={true} isStreaming={isStreaming} />
                </motion.div>
              )
            default:
              return null
          }
        })}
      </AnimatePresence>
      <AnimatePresence>
        {showStillWorking && (
          <motion.div
            key="still-working"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-2 mt-2 py-1.5 px-2 rounded-md bg-purple-500/5 border border-purple-500/10"
          >
            <Loader2 className="w-3.5 h-3.5 text-purple-400 animate-spin" />
            <span className="text-xs text-purple-400/80">{t('aiChatRendering.stillWorking')}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** Render a single message content string — handles both segmented and plain text. */
export function MessageContent({ content, isStreaming = false }: { content: string; isStreaming?: boolean }) {
  if (content.includes('\u00AB\u00ABSEG:')) {
    return <SegmentedContent raw={content} isStreaming={isStreaming} />
  }
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.15 }}
    >
      <RichTextContent text={content} standalone={true} isStreaming={isStreaming} />
    </motion.div>
  )
}

/** Bot message bubble with avatar — used by both AI Chat and Cortex. */
export function BotMessageBubble({ label, content, isStreaming = false }: { label: string; content: string; isStreaming?: boolean }) {
  return (
    <div className="flex gap-3 py-4 px-4 bg-muted/20 border-b border-border/20">
      <div className="w-7 h-7 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center shrink-0 mt-0.5">
        <Bot className="w-3.5 h-3.5 text-purple-600 dark:text-purple-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-[10px] uppercase tracking-wider text-purple-600 dark:text-purple-400/70 mb-1">{label}</p>
        <MessageContent content={content} isStreaming={isStreaming} />
      </div>
    </div>
  )
}

/** Collapsible thinking log. */
export function ThinkingLog({ log }: { log: string }) {
  const { t } = useTranslation()
  const [expanded, setExpanded] = useState(false)
  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <ChevronRight className={cn('w-3 h-3 transition-transform', expanded && 'rotate-90')} />
        {t('aiChatRendering.thinkingLog')}
      </button>
      {expanded && (
        <pre className="mt-2 p-3 rounded bg-muted border border-border text-xs text-muted-foreground whitespace-pre-wrap max-h-64 overflow-y-auto">
          {log}
        </pre>
      )}
    </div>
  )
}
