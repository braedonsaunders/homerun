import { useEffect } from 'react'
import { X, Keyboard } from 'lucide-react'
import clsx from 'clsx'
import { Shortcut, formatShortcutKey } from '../hooks/useKeyboardShortcuts'

interface KeyboardShortcutsHelpProps {
  isOpen: boolean
  onClose: () => void
  shortcuts: Shortcut[]
}

export default function KeyboardShortcutsHelp({ isOpen, onClose, shortcuts }: KeyboardShortcutsHelpProps) {
  // Close on Escape
  useEffect(() => {
    if (!isOpen) return
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [isOpen, onClose])

  if (!isOpen) return null

  // Group shortcuts by category
  const grouped = shortcuts.reduce<Record<string, Shortcut[]>>((acc, s) => {
    if (!acc[s.category]) acc[s.category] = []
    acc[s.category].push(s)
    return acc
  }, {})

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative bg-[#141414] border border-gray-800 rounded-2xl shadow-2xl w-full max-w-lg mx-4 max-h-[80vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-500/10 rounded-lg">
              <Keyboard className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h2 className="font-semibold text-white">Keyboard Shortcuts</h2>
              <p className="text-xs text-gray-500">Navigate faster with keyboard shortcuts</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-800 rounded-lg transition-colors"
          >
            <X className="w-4 h-4 text-gray-500" />
          </button>
        </div>

        {/* Shortcuts List */}
        <div className="overflow-y-auto max-h-[60vh] p-6 space-y-6">
          {Object.entries(grouped).map(([category, categoryShortcuts]) => (
            <div key={category}>
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                {category}
              </h3>
              <div className="space-y-2">
                {categoryShortcuts.map((shortcut, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-[#1a1a1a] transition-colors"
                  >
                    <span className="text-sm text-gray-300">{shortcut.description}</span>
                    <kbd className={clsx(
                      "px-2 py-1 rounded text-xs font-mono font-medium",
                      "bg-[#1a1a1a] text-gray-400 border border-gray-700"
                    )}>
                      {formatShortcutKey(shortcut)}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-gray-800 text-center">
          <p className="text-xs text-gray-600">
            Press <kbd className="px-1.5 py-0.5 bg-[#1a1a1a] rounded text-gray-400 border border-gray-700 text-[10px] font-mono">?</kbd> to toggle this help
          </p>
        </div>
      </div>
    </div>
  )
}
