import { useEffect, useRef } from 'react'
import { EditorView, keymap, placeholder as placeholderExt, lineNumbers } from '@codemirror/view'
import { EditorState } from '@codemirror/state'
import { python } from '@codemirror/lang-python'
import { json } from '@codemirror/lang-json'
import { oneDark } from '@codemirror/theme-one-dark'
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands'
import {
  bracketMatching,
  foldGutter,
  foldKeymap,
  indentOnInput,
  syntaxHighlighting,
  defaultHighlightStyle,
} from '@codemirror/language'
import { closeBrackets, closeBracketsKeymap, autocompletion } from '@codemirror/autocomplete'
import { highlightSelectionMatches, searchKeymap } from '@codemirror/search'
import { cn } from '../lib/utils'

interface CodeEditorProps {
  value: string
  onChange: (value: string) => void
  language?: 'python' | 'json'
  placeholder?: string
  readOnly?: boolean
  className?: string
  minHeight?: string
}

export default function CodeEditor({
  value,
  onChange,
  language = 'python',
  placeholder = '',
  readOnly = false,
  className,
  minHeight = '400px',
}: CodeEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewRef = useRef<EditorView | null>(null)
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange

  useEffect(() => {
    if (!containerRef.current) return

    const langExtension = language === 'json' ? json() : python()

    const extensions = [
      lineNumbers(),
      history(),
      foldGutter(),
      indentOnInput(),
      bracketMatching(),
      closeBrackets(),
      autocompletion(),
      highlightSelectionMatches(),
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      langExtension,
      oneDark,
      keymap.of([
        ...defaultKeymap,
        ...historyKeymap,
        ...closeBracketsKeymap,
        ...foldKeymap,
        ...searchKeymap,
        indentWithTab,
      ]),
      EditorView.updateListener.of((update) => {
        if (update.docChanged) {
          onChangeRef.current(update.state.doc.toString())
        }
      }),
      EditorView.theme({
        '&': {
          fontSize: '12.5px',
          minHeight,
          height: '100%',
          borderRadius: '0.375rem',
        },
        '.cm-scroller': {
          fontFamily: "'JetBrains Mono', monospace",
          lineHeight: '1.6',
          overflow: 'auto !important',
        },
        '.cm-content': {
          padding: '8px 0',
          caretColor: '#00ff88',
        },
        '.cm-gutters': {
          backgroundColor: 'transparent',
          borderRight: '1px solid hsl(220 13% 16%)',
          color: 'hsl(215 20% 35%)',
          minWidth: '40px',
        },
        '.cm-activeLineGutter': {
          backgroundColor: 'transparent',
          color: 'hsl(215 20% 60%)',
        },
        '.cm-activeLine': {
          backgroundColor: 'hsla(220, 13%, 13%, 0.5)',
        },
        '.cm-selectionMatch': {
          backgroundColor: 'hsla(155, 100%, 50%, 0.1)',
        },
        '&.cm-focused .cm-cursor': {
          borderLeftColor: '#00ff88',
        },
        '&.cm-focused .cm-selectionBackground, ::selection': {
          backgroundColor: 'hsla(155, 100%, 50%, 0.15)',
        },
        '.cm-foldGutter .cm-gutterElement': {
          padding: '0 4px',
          cursor: 'pointer',
        },
      }),
      EditorState.readOnly.of(readOnly),
    ]

    if (placeholder) {
      extensions.push(placeholderExt(placeholder))
    }

    const state = EditorState.create({
      doc: value,
      extensions,
    })

    const view = new EditorView({
      state,
      parent: containerRef.current,
    })

    viewRef.current = view

    return () => {
      view.destroy()
      viewRef.current = null
    }
    // Only re-create on language or readOnly change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language, readOnly, minHeight])

  // Sync external value changes without re-creating the editor
  useEffect(() => {
    const view = viewRef.current
    if (!view) return
    const currentValue = view.state.doc.toString()
    if (currentValue !== value) {
      view.dispatch({
        changes: {
          from: 0,
          to: currentValue.length,
          insert: value,
        },
      })
    }
  }, [value])

  return (
    <div
      ref={containerRef}
      className={cn(
        'overflow-auto rounded-md border border-border/70 bg-[#282c34]',
        readOnly && 'opacity-75',
        className
      )}
    />
  )
}
