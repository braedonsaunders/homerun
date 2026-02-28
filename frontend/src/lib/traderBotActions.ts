import {
  createTrader,
  getTraders,
  updateTrader,
  type Trader,
  type TraderSourceConfig,
} from '../services/api'

type TraderScopeMode = 'tracked' | 'pool' | 'individual' | 'group'

const VALID_SCOPE_MODES = new Set<TraderScopeMode>(['tracked', 'pool', 'individual', 'group'])

function normalizeWalletAddress(value: unknown): string {
  return String(value || '').trim().toLowerCase()
}

function normalizeUniqueStringList(value: unknown, normalizeLower = false): string[] {
  if (!Array.isArray(value)) return []
  const out: string[] = []
  const seen = new Set<string>()
  for (const rawItem of value) {
    const item = normalizeLower
      ? String(rawItem || '').trim().toLowerCase()
      : String(rawItem || '').trim()
    if (!item || seen.has(item)) continue
    seen.add(item)
    out.push(item)
  }
  return out
}

function normalizeScopeModes(value: unknown, fallback: TraderScopeMode[]): TraderScopeMode[] {
  if (!Array.isArray(value)) return [...fallback]
  const out: TraderScopeMode[] = []
  const seen = new Set<TraderScopeMode>()
  for (const rawMode of value) {
    const mode = String(rawMode || '').trim().toLowerCase() as TraderScopeMode
    if (!VALID_SCOPE_MODES.has(mode) || seen.has(mode)) continue
    seen.add(mode)
    out.push(mode)
  }
  return out.length > 0 ? out : [...fallback]
}

function mergeWalletIntoTradersScope(
  rawScope: unknown,
  walletAddress: string,
  fallbackModes: TraderScopeMode[],
): Record<string, unknown> {
  const scope = rawScope && typeof rawScope === 'object' ? { ...(rawScope as Record<string, unknown>) } : {}
  const modes = normalizeScopeModes(scope.modes, fallbackModes)
  const individualWallets = normalizeUniqueStringList(scope.individual_wallets, true)
  const groupIds = normalizeUniqueStringList(scope.group_ids, false)

  if (!modes.includes('individual')) {
    modes.push('individual')
  }
  if (!individualWallets.includes(walletAddress)) {
    individualWallets.push(walletAddress)
  }

  return {
    ...scope,
    modes,
    individual_wallets: individualWallets,
    group_ids: groupIds,
  }
}

function mergeWalletIntoSourceConfigs(sourceConfigs: TraderSourceConfig[], walletAddress: string): TraderSourceConfig[] {
  const nextSourceConfigs = Array.isArray(sourceConfigs)
    ? sourceConfigs
        .filter((sourceConfig) => sourceConfig && typeof sourceConfig === 'object')
        .map((sourceConfig) => ({ ...sourceConfig }))
    : []

  let hasTradersSource = false

  for (let index = 0; index < nextSourceConfigs.length; index += 1) {
    const sourceConfig = nextSourceConfigs[index]
    const sourceKey = String(sourceConfig.source_key || '').trim().toLowerCase()
    if (sourceKey !== 'traders') continue

    hasTradersSource = true
    const strategyParams =
      sourceConfig.strategy_params && typeof sourceConfig.strategy_params === 'object'
        ? { ...sourceConfig.strategy_params }
        : {}

    strategyParams.traders_scope = mergeWalletIntoTradersScope(
      strategyParams.traders_scope,
      walletAddress,
      ['tracked', 'pool'],
    )

    nextSourceConfigs[index] = {
      ...sourceConfig,
      source_key: 'traders',
      strategy_key: String(sourceConfig.strategy_key || '').trim().toLowerCase() || 'traders_copy_trade',
      strategy_version: sourceConfig.strategy_version ?? null,
      strategy_params: strategyParams,
    }
  }

  if (!hasTradersSource) {
    nextSourceConfigs.push({
      source_key: 'traders',
      strategy_key: 'traders_copy_trade',
      strategy_version: null,
      strategy_params: {
        traders_scope: {
          modes: ['individual'],
          individual_wallets: [walletAddress],
          group_ids: [],
        },
      },
    })
  }

  return nextSourceConfigs
}

function shortWalletLabel(walletAddress: string): string {
  if (walletAddress.length < 12) return walletAddress
  return `${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`
}

function resolveUniqueTraderName(baseName: string, traders: Trader[]): string {
  const cleanBase = String(baseName || '').trim() || 'Wallet Copy Bot'
  const existingNames = new Set(traders.map((trader) => String(trader.name || '').trim().toLowerCase()))
  if (!existingNames.has(cleanBase.toLowerCase())) {
    return cleanBase
  }
  let suffix = 2
  while (suffix < 10_000) {
    const candidate = `${cleanBase} ${suffix}`
    if (!existingNames.has(candidate.toLowerCase())) {
      return candidate
    }
    suffix += 1
  }
  return `${cleanBase} ${Date.now()}`
}

function resolveExistingTrader(traders: Trader[], identifier: string): Trader | null {
  const clean = String(identifier || '').trim()
  if (!clean) return null
  const byId = traders.find((trader) => String(trader.id || '').trim() === clean)
  if (byId) return byId
  const lowered = clean.toLowerCase()
  return traders.find((trader) => String(trader.name || '').trim().toLowerCase() === lowered) || null
}

export type AddWalletToTraderBotTarget = 'new' | 'existing'

export interface AddWalletToTraderBotRequest {
  walletAddress: string
  walletLabel?: string
  target: AddWalletToTraderBotTarget
  newTraderName?: string
  existingTraderIdOrName?: string
  mode?: 'paper' | 'live'
  tradersSnapshot?: Trader[]
}

export interface AddWalletToTraderBotResult {
  trader: Trader
  created: boolean
}

export async function addWalletToTraderBot(
  request: AddWalletToTraderBotRequest,
): Promise<AddWalletToTraderBotResult> {
  const walletAddress = normalizeWalletAddress(request.walletAddress)
  if (!walletAddress) {
    throw new Error('Wallet address is required')
  }

  const traders = Array.isArray(request.tradersSnapshot)
    ? request.tradersSnapshot
    : await getTraders()

  if (request.target === 'new') {
    const defaultName = `${request.walletLabel?.trim() || shortWalletLabel(walletAddress)} Copy Bot`
    const traderName = resolveUniqueTraderName(request.newTraderName || defaultName, traders)
    const trader = await createTrader({
      name: traderName,
      description: `Wallet copy bot seeded from ${walletAddress}`,
      mode: request.mode === 'live' ? 'live' : 'paper',
      source_configs: [
        {
          source_key: 'traders',
          strategy_key: 'traders_copy_trade',
          strategy_version: null,
          strategy_params: {
            traders_scope: {
              modes: ['individual'],
              individual_wallets: [walletAddress],
              group_ids: [],
            },
          },
        },
      ],
    })
    return { trader, created: true }
  }

  const targetIdentifier = String(request.existingTraderIdOrName || '').trim()
  if (!targetIdentifier) {
    throw new Error('Select an existing bot')
  }
  const existingTrader = resolveExistingTrader(traders, targetIdentifier)
  if (!existingTrader) {
    throw new Error(`Bot '${targetIdentifier}' not found`)
  }

  const nextSourceConfigs = mergeWalletIntoSourceConfigs(existingTrader.source_configs || [], walletAddress)
  const trader = await updateTrader(existingTrader.id, { source_configs: nextSourceConfigs })
  return { trader, created: false }
}
