import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  Search,
  AlertTriangle,
  CheckCircle,
  XCircle,
  TrendingUp,
  Eye,
  RefreshCw,
  Shield
} from 'lucide-react'
import clsx from 'clsx'
import {
  analyzeWallet,
  findProfitableWallets,
  getAnomalies,
  quickCheckWallet,
  WalletAnalysis
} from '../services/api'

export default function AnomalyPanel() {
  const [searchAddress, setSearchAddress] = useState('')
  const [analysisResult, setAnalysisResult] = useState<WalletAnalysis | null>(null)

  const { data: anomalies = [] } = useQuery({
    queryKey: ['anomalies'],
    queryFn: () => getAnomalies({ limit: 50 }),
  })

  const { data: profitableWallets } = useQuery({
    queryKey: ['profitable-wallets'],
    queryFn: () => findProfitableWallets({
      min_trades: 30,
      min_win_rate: 0.6,
      min_pnl: 500,
      max_anomaly_score: 0.5
    }),
  })

  const analyzeMutation = useMutation({
    mutationFn: analyzeWallet,
    onSuccess: (data) => setAnalysisResult(data)
  })

  const handleAnalyze = () => {
    if (searchAddress.trim()) {
      analyzeMutation.mutate(searchAddress.trim())
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold">Wallet Analysis & Anomaly Detection</h2>
        <p className="text-sm text-gray-500">
          Find profitable wallets to copy and detect suspicious trading patterns
        </p>
      </div>

      {/* Search */}
      <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
        <h3 className="font-medium mb-3">Analyze a Wallet</h3>
        <div className="flex gap-3">
          <input
            type="text"
            value={searchAddress}
            onChange={(e) => setSearchAddress(e.target.value)}
            placeholder="Enter wallet address (0x...)"
            className="flex-1 bg-[#1a1a1a] border border-gray-700 rounded-lg px-4 py-2"
          />
          <button
            onClick={handleAnalyze}
            disabled={!searchAddress.trim() || analyzeMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg font-medium disabled:opacity-50"
          >
            {analyzeMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            Analyze
          </button>
        </div>
      </div>

      {/* Analysis Result */}
      {analysisResult && (
        <AnalysisResultCard analysis={analysisResult} />
      )}

      {/* Profitable Wallets */}
      <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium">Discovered Profitable Wallets</h3>
          <span className="text-xs text-gray-500">
            {profitableWallets?.count || 0} found
          </span>
        </div>

        {!profitableWallets?.wallets?.length ? (
          <p className="text-gray-500 text-center py-8">
            No profitable wallets discovered yet. Wallets are analyzed as you track them.
          </p>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {profitableWallets.wallets.map((wallet: any) => (
              <div
                key={wallet.address}
                className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3"
              >
                <div>
                  <p className="text-sm font-mono">{wallet.address.slice(0, 10)}...{wallet.address.slice(-8)}</p>
                  <p className="text-xs text-gray-500">
                    Win: {(wallet.win_rate * 100).toFixed(1)}% |
                    ROI: {wallet.avg_roi.toFixed(1)}% |
                    PnL: ${wallet.total_pnl.toFixed(0)}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className={clsx(
                    "px-2 py-0.5 rounded text-xs",
                    wallet.anomaly_score < 0.3 ? "bg-green-500/20 text-green-400" :
                    wallet.anomaly_score < 0.6 ? "bg-yellow-500/20 text-yellow-400" :
                    "bg-red-500/20 text-red-400"
                  )}>
                    Risk: {(wallet.anomaly_score * 100).toFixed(0)}%
                  </span>
                  <button
                    onClick={() => {
                      setSearchAddress(wallet.address)
                      analyzeMutation.mutate(wallet.address)
                    }}
                    className="p-1 hover:bg-gray-700 rounded"
                  >
                    <Eye className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent Anomalies */}
      <div className="bg-[#141414] border border-gray-800 rounded-lg p-4">
        <h3 className="font-medium mb-4">Recent Anomalies Detected</h3>

        {anomalies.anomalies?.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No anomalies detected</p>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {anomalies.anomalies?.slice(0, 10).map((anomaly: any) => (
              <AnomalyRow key={anomaly.id} anomaly={anomaly} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function AnalysisResultCard({ analysis }: { analysis: WalletAnalysis }) {
  const isProfitable = analysis.is_profitable_pattern
  const isRisky = analysis.anomaly_score > 0.5

  return (
    <div className={clsx(
      "border rounded-lg p-4",
      isProfitable && !isRisky ? "bg-green-500/5 border-green-500/30" :
      isRisky ? "bg-red-500/5 border-red-500/30" :
      "bg-[#141414] border-gray-800"
    )}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="font-mono text-sm">{analysis.wallet}</p>
          <div className="flex items-center gap-2 mt-1">
            {isProfitable && !isRisky ? (
              <span className="flex items-center gap-1 text-green-400 text-xs">
                <CheckCircle className="w-3 h-3" /> Profitable Pattern
              </span>
            ) : isRisky ? (
              <span className="flex items-center gap-1 text-red-400 text-xs">
                <XCircle className="w-3 h-3" /> Suspicious
              </span>
            ) : (
              <span className="flex items-center gap-1 text-gray-400 text-xs">
                <Shield className="w-3 h-3" /> Neutral
              </span>
            )}
          </div>
        </div>
        <div className={clsx(
          "px-3 py-1 rounded-lg text-sm font-medium",
          analysis.anomaly_score < 0.3 ? "bg-green-500/20 text-green-400" :
          analysis.anomaly_score < 0.6 ? "bg-yellow-500/20 text-yellow-400" :
          "bg-red-500/20 text-red-400"
        )}>
          Risk Score: {(analysis.anomaly_score * 100).toFixed(0)}%
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-5 gap-4 mb-4">
        <StatBox label="Trades" value={analysis.stats.total_trades.toString()} />
        <StatBox label="Win Rate" value={`${(analysis.stats.win_rate * 100).toFixed(1)}%`} />
        <StatBox label="Total PnL" value={`$${analysis.stats.total_pnl.toFixed(0)}`} />
        <StatBox label="Avg ROI" value={`${analysis.stats.avg_roi.toFixed(1)}%`} />
        <StatBox label="Max ROI" value={`${analysis.stats.max_roi.toFixed(1)}%`} />
      </div>

      {/* Strategies */}
      {analysis.strategies_detected.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">Strategies Detected</p>
          <div className="flex flex-wrap gap-2">
            {analysis.strategies_detected.map((strategy) => (
              <span key={strategy} className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs">
                {strategy.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Anomalies */}
      {analysis.anomalies.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">Anomalies Detected</p>
          <div className="space-y-1">
            {analysis.anomalies.map((anomaly, idx) => (
              <div key={idx} className="flex items-center gap-2 text-sm">
                <AlertTriangle className={clsx(
                  "w-3 h-3",
                  anomaly.severity === 'critical' ? 'text-red-500' :
                  anomaly.severity === 'high' ? 'text-orange-500' :
                  anomaly.severity === 'medium' ? 'text-yellow-500' :
                  'text-gray-500'
                )} />
                <span className="text-gray-300">{anomaly.description}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendation */}
      <div className={clsx(
        "p-3 rounded-lg",
        isProfitable && !isRisky ? "bg-green-500/10" :
        isRisky ? "bg-red-500/10" :
        "bg-gray-800"
      )}>
        <p className="text-sm font-medium">{analysis.recommendation}</p>
      </div>
    </div>
  )
}

function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="font-mono font-medium">{value}</p>
    </div>
  )
}

function AnomalyRow({ anomaly }: { anomaly: any }) {
  const severityColors: Record<string, string> = {
    critical: 'bg-red-500/20 text-red-400',
    high: 'bg-orange-500/20 text-orange-400',
    medium: 'bg-yellow-500/20 text-yellow-400',
    low: 'bg-gray-500/20 text-gray-400'
  }

  return (
    <div className="flex items-center justify-between bg-[#1a1a1a] rounded-lg p-3">
      <div className="flex items-center gap-3">
        <AlertTriangle className={clsx(
          "w-4 h-4",
          anomaly.severity === 'critical' ? 'text-red-500' :
          anomaly.severity === 'high' ? 'text-orange-500' :
          'text-yellow-500'
        )} />
        <div>
          <p className="text-sm">{anomaly.type.replace(/_/g, ' ')}</p>
          <p className="text-xs text-gray-500">
            {anomaly.wallet?.slice(0, 10)}...
          </p>
        </div>
      </div>
      <span className={clsx("px-2 py-0.5 rounded text-xs", severityColors[anomaly.severity])}>
        {anomaly.severity}
      </span>
    </div>
  )
}
