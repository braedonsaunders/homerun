import { useState, useRef, useEffect, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
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
import clsx from 'clsx'
import { sendAIChat, AIChatMessage } from '../services/api'

interface AICopilotPanelProps {
  isOpen: boolean
  onClose: () => void
  contextType?: string
  contextId?: string
  contextLabel?: string
}

export default function AICopilotPanel({
  isOpen,
  onClose,
  contextType,
  contextId,
  contextLabel,
}: AICopilotPanelProps) {
  const [messages, setMessages] = useState<AIChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isExpanded, setIsExpanded] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

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

  const chatMutation = useMutation({
    mutationFn: async (message: string) => {
      const result = await sendAIChat({
        message,
        context_type: contextType,
        context_id: contextId,
        history: messages,
      })
      return result
    },
    onSuccess: (data) => {
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const quickActions = [
    { label: 'Analyze risk factors', prompt: 'What are the main risk factors for this opportunity?' },
    { label: 'Resolution safety', prompt: 'How safe is the resolution criteria? Any ambiguities?' },
    { label: 'Should I trade?', prompt: 'Given the current data, should I execute this trade? What are the pros and cons?' },
    { label: 'Explain strategy', prompt: 'Explain how this arbitrage strategy works and why this opportunity exists.' },
  ]

  if (!isOpen) return null

  return (
    <div
      className={clsx(
        'fixed bottom-4 right-4 z-50 flex flex-col bg-[#0f0f0f] border border-gray-800 rounded-2xl shadow-2xl shadow-purple-500/5 transition-all',
        isExpanded ? 'w-[560px] h-[700px]' : 'w-[420px] h-[560px]'
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">AI Copilot</h3>
            {contextLabel && (
              <p className="text-[10px] text-purple-400 truncate max-w-[200px]">
                {contextLabel}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="p-1.5 hover:bg-gray-800 rounded-lg transition-colors"
              title="Clear chat"
            >
              <Trash2 className="w-3.5 h-3.5 text-gray-500" />
            </button>
          )}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 hover:bg-gray-800 rounded-lg transition-colors"
          >
            {isExpanded ? (
              <Minimize2 className="w-3.5 h-3.5 text-gray-500" />
            ) : (
              <Maximize2 className="w-3.5 h-3.5 text-gray-500" />
            )}
          </button>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X className="w-3.5 h-3.5 text-gray-500" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center mb-3">
              <Bot className="w-6 h-6 text-purple-400" />
            </div>
            <p className="text-sm text-gray-400 mb-1">
              Ask me anything about your trades
            </p>
            <p className="text-xs text-gray-600 mb-4">
              I can analyze opportunities, assess risk, and help you make decisions.
            </p>

            {/* Quick Actions */}
            {contextType === 'opportunity' && (
              <div className="grid grid-cols-2 gap-2 w-full max-w-sm">
                {quickActions.map((action) => (
                  <button
                    key={action.label}
                    onClick={() => {
                      setMessages([{ role: 'user', content: action.prompt }])
                      chatMutation.mutate(action.prompt)
                    }}
                    className="text-left text-xs p-2.5 bg-[#1a1a1a] border border-gray-800 rounded-xl hover:border-purple-500/30 hover:bg-purple-500/5 transition-colors text-gray-400 hover:text-gray-300"
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={clsx('flex gap-2', msg.role === 'user' ? 'justify-end' : 'justify-start')}
          >
            {msg.role === 'assistant' && (
              <div className="w-6 h-6 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Bot className="w-3.5 h-3.5 text-purple-400" />
              </div>
            )}
            <div
              className={clsx(
                'max-w-[80%] rounded-xl px-3 py-2 text-sm',
                msg.role === 'user'
                  ? 'bg-blue-500/20 text-blue-100'
                  : 'bg-[#1a1a1a] text-gray-300 border border-gray-800'
              )}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
            </div>
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
            <div className="bg-[#1a1a1a] rounded-xl px-3 py-2 border border-gray-800">
              <RefreshCw className="w-4 h-4 text-purple-400 animate-spin" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-3 pb-3 pt-1">
        <div className="flex items-end gap-2 bg-[#1a1a1a] border border-gray-800 rounded-xl px-3 py-2 focus-within:border-purple-500/50 transition-colors">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about this opportunity..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-white placeholder-gray-600 resize-none focus:outline-none max-h-24"
            style={{ minHeight: '24px' }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || chatMutation.isPending}
            className={clsx(
              'p-1.5 rounded-lg transition-colors flex-shrink-0',
              input.trim() && !chatMutation.isPending
                ? 'bg-purple-500 hover:bg-purple-600 text-white'
                : 'bg-gray-800 text-gray-600 cursor-not-allowed'
            )}
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  )
}
