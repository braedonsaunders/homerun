import { Keyboard } from 'lucide-react'
import { cn } from '../lib/utils'
import { Shortcut, formatShortcutKey } from '../hooks/useKeyboardShortcuts'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog'
import { Badge } from './ui/badge'
import { ScrollArea } from './ui/scroll-area'
import { Separator } from './ui/separator'

interface KeyboardShortcutsHelpProps {
  isOpen: boolean
  onClose: () => void
  shortcuts: Shortcut[]
}

export default function KeyboardShortcutsHelp({ isOpen, onClose, shortcuts }: KeyboardShortcutsHelpProps) {
  // Group shortcuts by category
  const grouped = shortcuts.reduce<Record<string, Shortcut[]>>((acc, s) => {
    if (!acc[s.category]) acc[s.category] = []
    acc[s.category].push(s)
    return acc
  }, {})

  return (
    <Dialog open={isOpen} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent className="max-w-lg max-h-[80vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Keyboard className="w-5 h-5 text-primary" />
            </div>
            Keyboard Shortcuts
          </DialogTitle>
          <DialogDescription>Navigate faster with keyboard shortcuts</DialogDescription>
        </DialogHeader>

        <ScrollArea className="max-h-[60vh]">
          <div className="space-y-6 p-1">
            {Object.entries(grouped).map(([category, categoryShortcuts], groupIdx) => (
              <div key={category}>
                {groupIdx > 0 && <Separator className="mb-6" />}
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                  {category}
                </h3>
                <div className="space-y-2">
                  {categoryShortcuts.map((shortcut, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-muted transition-colors"
                    >
                      <span className="text-sm text-foreground">{shortcut.description}</span>
                      <Badge
                        variant="outline"
                        className={cn(
                          "px-2 py-1 rounded text-xs font-mono font-medium",
                          "bg-muted text-muted-foreground border-border"
                        )}
                      >
                        {formatShortcutKey(shortcut)}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="border-t border-border pt-3 text-center">
          <p className="text-xs text-muted-foreground">
            Press{' '}
            <Badge
              variant="outline"
              className="px-1.5 py-0.5 bg-muted text-muted-foreground border-border text-[10px] font-mono"
            >
              ?
            </Badge>
            {' '}to toggle this help
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}
