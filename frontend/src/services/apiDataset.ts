import axios from 'axios'

const api = axios.create({ baseURL: '/api', timeout: 600000 })

export type DatasetColumnType = 'string' | 'int' | 'float' | 'datetime' | 'json' | 'enum'

export interface DatasetColumn {
  key: string
  label: string
  type: DatasetColumnType
  sortable: boolean
  default_visible: boolean
  enum_values: string[] | null
  description: string
}

export type DatasetFilterKind =
  | 'eq'
  | 'contains'
  | 'time_range_start'
  | 'time_range_end'
  | 'enum_in'

export interface DatasetFilter {
  key: string
  column: string
  label: string
  kind: DatasetFilterKind
  description: string
}

export interface DatasetSummary {
  name: string
  label: string
  description: string
  row_count: number
  default_sort: string
  default_sort_dir: 'asc' | 'desc'
  columns: DatasetColumn[]
  filters: DatasetFilter[]
}

export interface DatasetQueryResult {
  dataset: string
  label: string
  total: number
  limit: number
  offset: number
  order_by: string
  order_dir: 'asc' | 'desc'
  columns: DatasetColumn[]
  filters: DatasetFilter[]
  rows: Array<Record<string, unknown>>
}

export type DatasetFilterValues = Record<string, string | string[] | undefined>

export interface DatasetQueryParams {
  limit?: number
  offset?: number
  order_by?: string
  order_dir?: 'asc' | 'desc'
  filters?: DatasetFilterValues
}

export async function listDatasets(): Promise<DatasetSummary[]> {
  const { data } = await api.get<{ datasets: DatasetSummary[] }>('/dataset')
  return data.datasets ?? []
}

function buildParams(p: DatasetQueryParams): Record<string, string | number> {
  const out: Record<string, string | number> = {}
  if (p.limit != null) out.limit = p.limit
  if (p.offset != null) out.offset = p.offset
  if (p.order_by) out.order_by = p.order_by
  if (p.order_dir) out.order_dir = p.order_dir
  if (p.filters) {
    for (const [k, v] of Object.entries(p.filters)) {
      if (v == null || v === '') continue
      if (Array.isArray(v)) {
        if (v.length > 0) out[k] = v.join(',')
      } else {
        out[k] = v
      }
    }
  }
  return out
}

export async function queryDataset(
  name: string,
  params: DatasetQueryParams = {},
): Promise<DatasetQueryResult> {
  const { data } = await api.get<DatasetQueryResult>(`/dataset/${encodeURIComponent(name)}`, {
    params: buildParams(params),
  })
  return data
}

export interface DatasetStorageRow {
  name: string
  label: string
  table_name: string
  row_count: number
  size_bytes: number | null
  oldest_at: string | null
  newest_at: string | null
}

export interface DatasetStorageSummary {
  tables: DatasetStorageRow[]
  total_rows: number
  total_bytes: number | null
}

export async function getDatasetStorageSummary(): Promise<DatasetStorageSummary> {
  const { data } = await api.get<DatasetStorageSummary>('/dataset/storage/summary')
  return data
}

export async function getDatasetDistinct(
  name: string,
  column: string,
  limit = 200,
): Promise<string[]> {
  const { data } = await api.get<{ column: string; values: unknown[] }>(
    `/dataset/${encodeURIComponent(name)}/distinct/${encodeURIComponent(column)}`,
    { params: { limit } },
  )
  return (data.values ?? []).map((v) => String(v))
}

/** Build a CSV-export URL the browser can navigate to (triggers download). */
export function datasetCsvUrl(
  name: string,
  params: DatasetQueryParams & { columns?: string[]; max_rows?: number } = {},
): string {
  const q = new URLSearchParams()
  const built = buildParams(params)
  for (const [k, v] of Object.entries(built)) {
    q.set(k, String(v))
  }
  if (params.columns && params.columns.length > 0) {
    q.set('columns', params.columns.join(','))
  }
  if (params.max_rows != null) {
    q.set('max_rows', String(params.max_rows))
  }
  return `/api/dataset/${encodeURIComponent(name)}/csv?${q.toString()}`
}
