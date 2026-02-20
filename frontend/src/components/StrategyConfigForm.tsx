import { useCallback, useState } from 'react'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Switch } from './ui/switch'

interface ParamField {
  key: string
  label: string
  type: string
  min?: number
  max?: number
  options?: string[]
  item_schema?: Record<string, string>
  description?: string
}

interface StrategyConfigFormProps {
  schema: { param_fields: ParamField[] }
  values: Record<string, unknown>
  onChange: (values: Record<string, unknown>) => void
}

export default function StrategyConfigForm({ schema, values, onChange }: StrategyConfigFormProps) {
  const fields = schema?.param_fields
  if (!fields || fields.length === 0) return null

  const updateField = useCallback(
    (key: string, value: unknown) => {
      onChange({ ...values, [key]: value })
    },
    [values, onChange]
  )

  return (
    <div className="grid gap-3 grid-cols-2 xl:grid-cols-3">
      {fields.map((field) => (
        <ConfigField
          key={field.key}
          field={field}
          value={values[field.key]}
          onChange={(val) => updateField(field.key, val)}
        />
      ))}
    </div>
  )
}

function JsonArrayField({
  field,
  value,
  onChange,
}: {
  field: ParamField
  value: unknown
  onChange: (val: unknown) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const items: Record<string, unknown>[] = Array.isArray(value) ? value : []
  const schema = field.item_schema || {}
  const keys = Object.keys(schema).length > 0
    ? Object.keys(schema)
    : items.length > 0
      ? Object.keys(items[0])
      : ['value']

  const updateItem = (idx: number, key: string, val: unknown) => {
    const next = items.map((item, i) => (i === idx ? { ...item, [key]: val } : item))
    onChange(next)
  }

  const addItem = () => {
    const blank: Record<string, unknown> = {}
    for (const k of keys) {
      const stype = schema[k] || 'string'
      blank[k] = stype === 'boolean' ? false : ''
    }
    onChange([...items, blank])
  }

  const removeItem = (idx: number) => {
    onChange(items.filter((_, i) => i !== idx))
  }

  return (
    <div className="col-span-2 xl:col-span-3 space-y-2">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
      >
        <svg
          className={`w-4 h-4 transition-transform ${expanded ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        {field.label}
        <span className="ml-1 inline-flex items-center rounded-full bg-blue-100 dark:bg-blue-900 px-2 py-0.5 text-xs font-medium text-blue-800 dark:text-blue-200">
          {items.length} {items.length === 1 ? 'item' : 'items'}
        </span>
      </button>
      {field.description && (
        <p className="text-xs text-gray-500 dark:text-gray-400 ml-6">{field.description}</p>
      )}
      {expanded && (
        <div className="ml-6 space-y-3">
          {items.map((item, idx) => (
            <div
              key={idx}
              className="relative rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-3"
            >
              <button
                type="button"
                onClick={() => removeItem(idx)}
                className="absolute top-2 right-2 text-gray-400 hover:text-red-500 transition-colors"
                title="Remove item"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <div className="grid grid-cols-2 gap-2 pr-6">
                {keys.map((key) => {
                  const stype = schema[key] || 'string'
                  const itemValue = item[key]
                  if (stype === 'boolean') {
                    return (
                      <label key={key} className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                        <input
                          type="checkbox"
                          checked={Boolean(itemValue)}
                          onChange={(e) => updateItem(idx, key, e.target.checked)}
                          className="rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
                        />
                        {key}
                      </label>
                    )
                  }
                  if (stype === 'json') {
                    return (
                      <div key={key} className="space-y-1">
                        <label className="block text-xs text-gray-500 dark:text-gray-400">{key}</label>
                        <input
                          type="text"
                          value={typeof itemValue === 'object' ? JSON.stringify(itemValue) : String(itemValue ?? '')}
                          onChange={(e) => {
                            try {
                              updateItem(idx, key, JSON.parse(e.target.value))
                            } catch {
                              updateItem(idx, key, e.target.value)
                            }
                          }}
                          placeholder={`${key} (JSON)`}
                          className="block w-full rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white text-xs shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        />
                      </div>
                    )
                  }
                  return (
                    <div key={key} className="space-y-1">
                      <label className="block text-xs text-gray-500 dark:text-gray-400">{key}</label>
                      <input
                        type="text"
                        value={String(itemValue ?? '')}
                        onChange={(e) => updateItem(idx, key, e.target.value)}
                        placeholder={key}
                        className="block w-full rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white text-xs shadow-sm focus:border-blue-500 focus:ring-blue-500"
                      />
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
          <button
            type="button"
            onClick={addItem}
            className="inline-flex items-center gap-1 rounded-md border border-dashed border-gray-300 dark:border-gray-600 px-3 py-1.5 text-xs text-gray-600 dark:text-gray-400 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Add Item
          </button>
        </div>
      )}
    </div>
  )
}

function ConfigField({
  field,
  value,
  onChange,
}: {
  field: ParamField
  value: unknown
  onChange: (value: unknown) => void
}) {
  switch (field.type) {
    case 'boolean':
      return (
        <div className="flex items-center justify-between gap-2 rounded-md bg-muted/30 px-3 py-2">
          <Label className="text-[11px] text-muted-foreground cursor-pointer">{field.label}</Label>
          <Switch
            checked={Boolean(value)}
            onCheckedChange={onChange}
            className="scale-75"
          />
        </div>
      )

    case 'enum':
      return (
        <div>
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <Select value={String(value || '')} onValueChange={onChange}>
            <SelectTrigger className="mt-1 h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(field.options || []).map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {opt}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )

    case 'integer':
      return (
        <div>
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <Input
            type="number"
            step={1}
            min={field.min}
            max={field.max}
            value={value != null ? String(value) : ''}
            onChange={(e) => {
              const v = e.target.value
              onChange(v === '' ? undefined : parseInt(v, 10))
            }}
            className="mt-1 h-8 text-xs font-mono"
          />
        </div>
      )

    case 'array[string]':
      return (
        <div>
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <Input
            value={Array.isArray(value) ? (value as string[]).join(', ') : String(value || '')}
            onChange={(e) => {
              const items = e.target.value
                .split(',')
                .map((s) => s.trim())
                .filter(Boolean)
              onChange(items)
            }}
            className="mt-1 h-8 text-xs font-mono"
            placeholder="value1, value2"
          />
        </div>
      )

    case 'string':
      return (
        <div>
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <Input
            value={String(value || '')}
            onChange={(e) => onChange(e.target.value)}
            className="mt-1 h-8 text-xs"
          />
        </div>
      )

    case 'list':
      return (
        <div>
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <Input
            value={Array.isArray(value) ? (value as string[]).join(', ') : String(value || '')}
            onChange={(e) => {
              const items = e.target.value
                .split(',')
                .map((s) => s.trim())
                .filter(Boolean)
              onChange(items)
            }}
            className="mt-1 h-8 text-xs font-mono"
            placeholder="value1, value2"
          />
        </div>
      )

    case 'json':
      return (
        <JsonArrayField
          field={field}
          value={value}
          onChange={(val) => onChange(val)}
        />
      )

    case 'url':
      return (
        <div>
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <Input
            type="url"
            value={String(value || '')}
            onChange={(e) => onChange(e.target.value)}
            className="mt-1 h-8 text-xs font-mono"
            placeholder="https://..."
          />
        </div>
      )

    case 'number':
    default:
      return (
        <div>
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <Input
            type="number"
            step="any"
            min={field.min}
            max={field.max}
            value={value != null ? String(value) : ''}
            onChange={(e) => {
              const v = e.target.value
              onChange(v === '' ? undefined : parseFloat(v))
            }}
            className="mt-1 h-8 text-xs font-mono"
          />
        </div>
      )
  }
}
