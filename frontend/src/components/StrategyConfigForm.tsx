import { useCallback } from 'react'
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
