import { useCallback, useEffect, useMemo, useState } from 'react'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Switch } from './ui/switch'

interface ParamField {
  key: string
  label: string
  type: string
  format?: string
  placeholder?: string
  min?: number
  max?: number
  options?: Array<string | { value?: string; label?: string }>
  item_schema?: Record<string, string>
  properties?: ParamField[]
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

function _csvListFromValue(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => String(item ?? '').trim())
      .filter(Boolean)
  }
  if (typeof value === 'string') {
    return value
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
  }
  return []
}

function CommaSeparatedListInput({
  field,
  value,
  onChange,
}: {
  field: ParamField
  value: unknown
  onChange: (value: unknown) => void
}) {
  const normalizedCsv = useMemo(() => _csvListFromValue(value).join(', '), [value])
  const [draftCsv, setDraftCsv] = useState(normalizedCsv)
  const [isFocused, setIsFocused] = useState(false)

  useEffect(() => {
    if (!isFocused) {
      setDraftCsv(normalizedCsv)
    }
  }, [isFocused, normalizedCsv])

  return (
    <div>
      <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
      <Input
        value={draftCsv}
        onFocus={() => setIsFocused(true)}
        onChange={(e) => {
          setDraftCsv(e.target.value)
        }}
        onKeyDown={(event) => {
          if (event.key !== 'Enter') return
          event.preventDefault()
          const parsed = _csvListFromValue(draftCsv)
          onChange(parsed)
          setDraftCsv(parsed.join(', '))
        }}
        onBlur={() => {
          setIsFocused(false)
          const parsed = _csvListFromValue(draftCsv)
          onChange(parsed)
          setDraftCsv(parsed.join(', '))
        }}
        className="mt-1 h-8 text-xs font-mono"
        placeholder="value1, value2"
      />
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

function ArrayStringOptionsInput({
  field,
  value,
  options,
  onChange,
}: {
  field: ParamField
  value: unknown
  options: Array<{ value: string; label: string }>
  onChange: (value: unknown) => void
}) {
  const selectedValues = useMemo(() => {
    if (!Array.isArray(value)) return new Set<string>()
    return new Set(
      value
        .map((item) => String(item || '').trim())
        .filter(Boolean)
    )
  }, [value])
  const [filterText, setFilterText] = useState('')
  const filteredOptions = useMemo(() => {
    const query = filterText.trim().toLowerCase()
    if (!query) return options
    return options.filter((option) => {
      const label = option.label.toLowerCase()
      const candidate = option.value.toLowerCase()
      return label.includes(query) || candidate.includes(query)
    })
  }, [filterText, options])

  const toggleValue = (nextValue: string) => {
    const next = new Set(selectedValues)
    if (next.has(nextValue)) {
      next.delete(nextValue)
    } else {
      next.add(nextValue)
    }
    onChange(Array.from(next))
  }

  const clearSelected = () => onChange([])

  return (
    <div className="col-span-2 xl:col-span-3 space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground">{selectedValues.size} selected</span>
          {selectedValues.size > 0 ? (
            <button
              type="button"
              onClick={clearSelected}
              className="text-[10px] text-muted-foreground underline underline-offset-2 hover:text-foreground"
            >
              Clear
            </button>
          ) : null}
        </div>
      </div>
      <Input
        value={filterText}
        onChange={(event) => setFilterText(event.target.value)}
        placeholder="Filter options..."
        className="h-8 text-xs"
      />
      <div className="max-h-36 overflow-y-auto rounded-md border border-border/70 bg-background/70 p-2 space-y-1">
        {filteredOptions.length > 0 ? (
          filteredOptions.map((option) => {
            const checked = selectedValues.has(option.value)
            return (
              <label
                key={option.value}
                className="flex cursor-pointer items-center gap-2 rounded px-1 py-0.5 text-xs hover:bg-muted/50"
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleValue(option.value)}
                  className="h-3.5 w-3.5 rounded border-border bg-background"
                />
                <span className="truncate" title={option.label}>{option.label}</span>
              </label>
            )
          })
        ) : (
          <p className="text-[11px] text-muted-foreground">No options match this filter.</p>
        )}
      </div>
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
  const enumOptions: Array<{ value: string; label: string }> = Array.isArray(field.options)
    ? field.options
      .map((option) => {
        if (typeof option === 'string') {
          const v = option.trim()
          return v ? { value: v, label: v } : null
        }
        const v = String(option.value || '').trim()
        if (!v) return null
        const label = String(option.label || v).trim() || v
        return { value: v, label }
      })
      .filter((option): option is { value: string; label: string } => Boolean(option))
    : []
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
          <Select value={String(value || enumOptions[0]?.value || '')} onValueChange={onChange}>
            <SelectTrigger className="mt-1 h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {enumOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
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
      if (enumOptions.length > 0) {
        return (
          <ArrayStringOptionsInput
            field={field}
            value={value}
            options={enumOptions}
            onChange={onChange}
          />
        )
      }
      return <CommaSeparatedListInput field={field} value={value} onChange={onChange} />

    case 'object': {
      const properties = Array.isArray(field.properties) ? field.properties : []
      const objectValue =
        value && typeof value === 'object' && !Array.isArray(value)
          ? (value as Record<string, unknown>)
          : {}
      return (
        <div className="col-span-2 xl:col-span-3 rounded-md border border-border/70 p-2.5 space-y-2">
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <div className="grid gap-2 md:grid-cols-2">
            {properties.map((property) => (
              <ConfigField
                key={`${field.key}.${property.key}`}
                field={property}
                value={objectValue[property.key]}
                onChange={(nextPropertyValue) =>
                  onChange({
                    ...objectValue,
                    [property.key]: nextPropertyValue,
                  })
                }
              />
            ))}
          </div>
        </div>
      )
    }

    case 'string':
      return (
        <div>
          <Label className="text-[11px] text-muted-foreground">{field.label}</Label>
          <Input
            type={field.format === 'time' ? 'time' : field.format === 'date' ? 'date' : 'text'}
            value={String(value || '')}
            onChange={(e) => onChange(e.target.value)}
            className="mt-1 h-8 text-xs"
            placeholder={field.placeholder}
          />
        </div>
      )

    case 'list':
      return <CommaSeparatedListInput field={field} value={value} onChange={onChange} />

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
