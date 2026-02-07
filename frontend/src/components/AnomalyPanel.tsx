import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  Search,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Shield,
  AlertOctagon,
  Info
} from 'lucide-react'
import { cn } from '../lib/utils'
import {
  analyzeWallet,
  getAnomalies,
  WalletAnalysis
} from '../services/api'

export default function AnomalyPanel() {
  const [searchAddress, setSearchAddress] = useState('')
  const [analysisResult, setAnalysisResult] = useState<WalletAnalysis | null>(null)

  const { data: anomalies = { anomalies: [], count: 0 }, refetch: refetchAnomalies } = useQuery({
    queryKey: ['anomalies'],
    queryFn: () => getAnomalies({ limit: 50 }),
  })

  const analyzeMutation = useMutation({
    mutationFn: analyzeWallet,
    onSuccess: (data) => {
      setAnalysisResult(data)
      refetchAnomalies()
    }
  })

  const handleAnalyze = () => {
    if (searchAddress.trim()) {
      analyzeMutation.mutate(searchAddress.trim())
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAnalyze()
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold">Anomaly Detection</h2>
        <p className="text-sm text-gray-500">
          Detect suspicious trading patterns and identify potential manipulation
        </p>
      </div>

      {/* Search */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="font-medium mb-3 flex items-center gap-2">
          <Shield className="w-4 h-4" />
          Check Wallet for Anomalies
        </h3>
        <div className="flex gap-3">
          <input
            type="text"
            value={searchAddress}
            onChange={(e) => setSearchAddress(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter wallet address (0x...)"
            className="flex-1 bg-muted border border-border rounded-lg px-4 py-2 font-mono text-sm"
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
            Scan
          </button>
        </div>
      </div>

      {/* Analysis Result */}
      {analysisResult && (
        <AnomalyResultCard analysis={analysisResult} />
      )}

      {/* Recent Anomalies */}
      <div className="bg-card border border-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium flex items-center gap-2">
            <AlertOctagon className="w-4 h-4" />
            Recent Anomalies Detected
          </h3>
          <span className="text-xs text-gray-500">
            {anomalies.count || 0} found
          </span>
        </div>

        {!anomalies.anomalies?.length ? (
          <div className="text-center py-8">
            <Shield className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-500">No anomalies detected</p>
            <p className="text-xs text-gray-600 mt-1">
              Analyze wallets to detect suspicious patterns
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {anomalies.anomalies?.slice(0, 20).map((anomaly: any) => (
              <AnomalyRow key={anomaly.id} anomaly={anomaly} />
            ))}
          </div>
        )}
      </div>

      {/* Anomaly Types Info */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="font-medium mb-4 flex items-center gap-2">
          <Info className="w-4 h-4" />
          What We Detect
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <AnomalyTypeCard
            title="Statistical Anomalies"
            items={[
              "Impossible win rates (>95%)",
              "Unusually high ROI (>20% avg)",
              "Zero losses over many trades",
              "Perfect timing patterns"
            ]}
            severity="critical"
          />
          <AnomalyTypeCard
            title="Pattern Anomalies"
            items={[
              "Wash trading (rapid buy/sell)",
              "Front-running behavior",
              "Coordinated trading",
              "Arbitrage-only patterns"
            ]}
            severity="high"
          />
        </div>
      </div>
    </div>
  )
}

function AnomalyResultCard({ analysis }: { analysis: WalletAnalysis }) {
  const isRisky = analysis.anomaly_score > 0.5
  const isSafe = analysis.anomaly_score < 0.3
  const hasAnomalies = analysis.anomalies.length > 0

  const getVerdict = () => {
    if (analysis.anomaly_score >= 0.7) return { text: 'HIGH RISK', color: 'red', icon: XCircle }
    if (analysis.anomaly_score >= 0.5) return { text: 'SUSPICIOUS', color: 'orange', icon: AlertTriangle }
    if (analysis.anomaly_score >= 0.3) return { text: 'CAUTION', color: 'yellow', icon: AlertTriangle }
    return { text: 'LOW RISK', color: 'green', icon: CheckCircle }
  }

  const verdict = getVerdict()
  const VerdictIcon = verdict.icon

  return (
    <div className={cn(
      "border rounded-lg p-4",
      verdict.color === 'red' && "bg-red-500/5 border-red-500/30",
      verdict.color === 'orange' && "bg-orange-500/5 border-orange-500/30",
      verdict.color === 'yellow' && "bg-yellow-500/5 border-yellow-500/30",
      verdict.color === 'green' && "bg-green-500/5 border-green-500/30"
    )}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="font-mono text-sm text-gray-400">Wallet Scanned</p>
          <p className="font-mono font-medium">{analysis.wallet}</p>
        </div>
        <div className={cn(
          "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium",
          verdict.color === 'red' && "bg-red-500/20 text-red-400",
          verdict.color === 'orange' && "bg-orange-500/20 text-orange-400",
          verdict.color === 'yellow' && "bg-yellow-500/20 text-yellow-400",
          verdict.color === 'green' && "bg-green-500/20 text-green-400"
        )}>
          <VerdictIcon className="w-4 h-4" />
          {verdict.text}
        </div>
      </div>

      {/* Risk Score */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-500">Anomaly Score</span>
          <span className="font-mono font-medium">{(analysis.anomaly_score * 100).toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={cn(
              "h-full transition-all",
              analysis.anomaly_score >= 0.7 && "bg-red-500",
              analysis.anomaly_score >= 0.5 && analysis.anomaly_score < 0.7 && "bg-orange-500",
              analysis.anomaly_score >= 0.3 && analysis.anomaly_score < 0.5 && "bg-yellow-500",
              analysis.anomaly_score < 0.3 && "bg-green-500"
            )}
            style={{ width: `${analysis.anomaly_score * 100}%` }}
          />
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4 mb-4 p-3 bg-muted rounded-lg">
        <div className="text-center">
          <p className="text-xs text-gray-500">Trades</p>
          <p className="font-mono font-medium">{analysis.stats.total_trades}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">Win Rate</p>
          <p className="font-mono font-medium">{(analysis.stats.win_rate * 100).toFixed(1)}%</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">Total PnL</p>
          <p className={cn(
            "font-mono font-medium",
            analysis.stats.total_pnl >= 0 ? "text-green-400" : "text-red-400"
          )}>
            ${analysis.stats.total_pnl.toFixed(0)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">Avg ROI</p>
          <p className="font-mono font-medium">{analysis.stats.avg_roi.toFixed(1)}%</p>
        </div>
      </div>

      {/* Detected Strategies */}
      {analysis.strategies_detected.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">Trading Strategies Detected</p>
          <div className="flex flex-wrap gap-2">
            {analysis.strategies_detected.map((strategy) => (
              <span key={strategy} className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs">
                {strategy.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Anomalies */}
      {hasAnomalies && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">Anomalies Detected ({analysis.anomalies.length})</p>
          <div className="space-y-2">
            {analysis.anomalies.map((anomaly, idx) => (
              <div key={idx} className="flex items-start gap-2 p-2 bg-muted rounded">
                <AlertTriangle className={cn(
                  "w-4 h-4 mt-0.5 flex-shrink-0",
                  anomaly.severity === 'critical' ? 'text-red-500' :
                  anomaly.severity === 'high' ? 'text-orange-500' :
                  anomaly.severity === 'medium' ? 'text-yellow-500' :
                  'text-gray-500'
                )} />
                <div>
                  <p className="text-sm text-gray-300">{anomaly.description}</p>
                  <span className={cn(
                    "text-xs px-1.5 py-0.5 rounded mt-1 inline-block",
                    anomaly.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                    anomaly.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                    anomaly.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-gray-500/20 text-gray-400'
                  )}>
                    {anomaly.severity}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendation */}
      <div className={cn(
        "p-3 rounded-lg",
        isRisky ? "bg-red-500/10" :
        isSafe ? "bg-green-500/10" :
        "bg-yellow-500/10"
      )}>
        <p className="text-sm font-medium">{analysis.recommendation}</p>
      </div>
    </div>
  )
}

function AnomalyRow({ anomaly }: { anomaly: any }) {
  const severityConfig: Record<string, { bg: string; text: string; icon: string }> = {
    critical: { bg: 'bg-red-500/20', text: 'text-red-400', icon: 'text-red-500' },
    high: { bg: 'bg-orange-500/20', text: 'text-orange-400', icon: 'text-orange-500' },
    medium: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', icon: 'text-yellow-500' },
    low: { bg: 'bg-gray-500/20', text: 'text-gray-400', icon: 'text-gray-500' }
  }

  const config = severityConfig[anomaly.severity] || severityConfig.low

  return (
    <div className="flex items-center justify-between bg-muted rounded-lg p-3">
      <div className="flex items-center gap-3">
        <AlertTriangle className={cn("w-4 h-4", config.icon)} />
        <div>
          <p className="text-sm font-medium">{anomaly.type.replace(/_/g, ' ')}</p>
          <p className="text-xs text-gray-500 font-mono">
            {anomaly.wallet?.slice(0, 10)}...{anomaly.wallet?.slice(-6)}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className={cn("px-2 py-0.5 rounded text-xs", config.bg, config.text)}>
          {anomaly.severity}
        </span>
        <span className="text-xs text-gray-500">
          {new Date(anomaly.detected_at).toLocaleDateString()}
        </span>
      </div>
    </div>
  )
}

function AnomalyTypeCard({ title, items, severity }: { title: string; items: string[]; severity: 'critical' | 'high' | 'medium' }) {
  const colors = {
    critical: 'border-red-500/30 bg-red-500/5',
    high: 'border-orange-500/30 bg-orange-500/5',
    medium: 'border-yellow-500/30 bg-yellow-500/5'
  }

  return (
    <div className={cn("border rounded-lg p-3", colors[severity])}>
      <h4 className="font-medium text-sm mb-2">{title}</h4>
      <ul className="space-y-1">
        {items.map((item, idx) => (
          <li key={idx} className="text-xs text-gray-400 flex items-center gap-2">
            <span className="w-1 h-1 rounded-full bg-gray-500" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}
