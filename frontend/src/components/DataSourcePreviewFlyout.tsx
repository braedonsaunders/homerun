import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation } from '@tanstack/react-query'
import {
  ChevronDown,
  ChevronRight,
  Eye,
  Loader2,
  MapPin,
  AlertTriangle,
} from 'lucide-react'

import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { ScrollArea } from './ui/scroll-area'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from './ui/sheet'
import { cn } from '../lib/utils'
import { getUnifiedDataSourceRecords } from '../services/api'

interface PreviewRecord {
  external_id: string | null
  title: string | null
  summary: string | null
  category: string | null
  source: string | null
  url: string | null
  geotagged: boolean
  country_iso3: string | null
  latitude: number | null
  longitude: number | null
  observed_at: string | null
  payload: Record<string, any> | null
  transformed: Record<string, any> | null
  tags: string[] | null
}

function PayloadViewer({ data, label }: { data: Record<string, any> | null; label: string }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  if (!data || Object.keys(data).length === 0) return null
  return (
    <div className="mt-1">
      <button
        className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
        onClick={() => setOpen(!open)}
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {label} ({t('dataSourcePreviewFlyout.fieldsCount', { count: Object.keys(data).length })})
      </button>
      {open && (
        <pre className="mt-1 bg-[#1e1e2e] border border-border/30 rounded-md p-2 text-[10px] leading-relaxed font-mono text-gray-300 overflow-x-auto whitespace-pre max-h-48">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}

function RecordRow({ record, index }: { record: PreviewRecord; index: number }) {
  const { t } = useTranslation()
  const severity = record.payload?.severity
  return (
    <div className="border-b border-border/30 px-4 py-3 hover:bg-muted/20 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-[10px] text-muted-foreground font-mono">#{index + 1}</span>
            <span className="text-[12px] font-medium text-foreground truncate">
              {record.title || t('dataSourcePreviewFlyout.untitled')}
            </span>
          </div>
          {record.summary && (
            <p className="text-[11px] text-muted-foreground truncate">{record.summary}</p>
          )}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {record.category && (
            <Badge variant="outline" className="text-[9px] px-1.5 py-0">
              {record.category}
            </Badge>
          )}
          {record.source && (
            <Badge variant="secondary" className="text-[9px] px-1.5 py-0">
              {record.source}
            </Badge>
          )}
        </div>
      </div>
      <div className="flex items-center gap-3 mt-1.5 text-[10px] text-muted-foreground">
        {record.country_iso3 && <span>{record.country_iso3}</span>}
        {record.geotagged && record.latitude != null && record.longitude != null && (
          <span className="flex items-center gap-0.5">
            <MapPin className="w-2.5 h-2.5" />
            {record.latitude.toFixed(2)}, {record.longitude.toFixed(2)}
          </span>
        )}
        {severity != null && (
          <span className={cn(
            severity >= 0.7 ? 'text-red-400' : severity >= 0.4 ? 'text-amber-400' : 'text-emerald-400'
          )}>
            sev={typeof severity === 'number' ? severity.toFixed(2) : severity}
          </span>
        )}
        {record.observed_at && (
          <span>{new Date(record.observed_at).toLocaleString()}</span>
        )}
      </div>
      <PayloadViewer data={record.payload} label="payload" />
    </div>
  )
}

export default function DataSourcePreviewFlyout({
  open,
  onOpenChange,
  sourceId,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  sourceId: string | null
}) {
  const { t } = useTranslation()
  const previewMutation = useMutation({
    mutationFn: async () => {
      if (!sourceId) throw new Error('No source selected')
      return getUnifiedDataSourceRecords(sourceId, { limit: 25, offset: 0 })
    },
  })

  const prevSourceIdRef = useRef(sourceId)
  useEffect(() => {
    if (sourceId !== prevSourceIdRef.current) {
      prevSourceIdRef.current = sourceId
      previewMutation.reset()
    }
    if (open && sourceId) {
      previewMutation.mutate()
    }
  }, [sourceId, open])

  const handleOpenChange = (isOpen: boolean) => {
    onOpenChange(isOpen)
    if (isOpen && sourceId) {
      previewMutation.mutate()
    }
  }

  const data = previewMutation.data

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-2xl p-0">
        <SheetHeader className="px-4 pr-10 pt-4 pb-3 border-b border-border/50">
          <div className="flex items-center justify-between">
            <div>
              <SheetTitle className="flex items-center gap-2 text-sm">
                <Eye className="w-4 h-4 text-cyan-400" />
                {t('dataSourcePreviewFlyout.preview')}
              </SheetTitle>
              <SheetDescription className="text-[11px] mt-0.5">
                {t('dataSourcePreviewFlyout.liveFetchNoRecords')}
              </SheetDescription>
            </div>
            <div className="flex items-center gap-2">
              {data && (
                <span className="text-[10px] text-muted-foreground">
                  {t('dataSourcePreviewFlyout.recordSummary', { shown: data.records.length, total: data.total })}
                </span>
              )}
              <Button
                variant="outline"
                size="sm"
                className="h-6 text-[10px] px-2 gap-1"
                onClick={() => previewMutation.mutate()}
                disabled={previewMutation.isPending || !sourceId}
              >
                {previewMutation.isPending ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <Eye className="w-3 h-3" />
                )}
                {t('common.refresh')}
              </Button>
            </div>
          </div>
        </SheetHeader>

        <ScrollArea className="h-[calc(100vh-100px)]">
          {previewMutation.isPending && (
            <div className="flex items-center justify-center py-16">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                <span className="text-[11px] text-muted-foreground">{t('dataSourcePreviewFlyout.fetchingPreview')}</span>
              </div>
            </div>
          )}

          {previewMutation.isError && (
            <div className="mx-4 mt-4 p-3 rounded-md bg-red-500/10 border border-red-500/30">
              <div className="flex items-center gap-2 text-red-400 text-[11px]">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                <span>{(previewMutation.error as Error)?.message || t('dataSourcePreviewFlyout.previewFailed')}</span>
              </div>
            </div>
          )}

          {data && !previewMutation.isPending && (
            <>
              {data.records.length === 0 ? (
                <div className="flex items-center justify-center py-16">
                  <span className="text-[11px] text-muted-foreground">
                    {t('dataSourcePreviewFlyout.zeroRecords')}
                  </span>
                </div>
              ) : (
                <div>
                  {data.records.map((record: PreviewRecord, idx: number) => (
                    <RecordRow key={record.external_id || idx} record={record} index={idx} />
                  ))}
                </div>
              )}
            </>
          )}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}
