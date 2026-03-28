// FraudDetection.jsx — Real model data, consistent with Dashboard and Credit Scoring
import { useState, useCallback } from 'react'
import { ShieldAlert, AlertTriangle, RefreshCw, Shield, Activity, ArrowDown, ArrowUp, ChevronDown, ChevronUp, Info } from 'lucide-react'
import RiskBadge from '../components/RiskBadge'
import { useApi } from '../hooks/useApi'
import clsx from 'clsx'

function fmtMoney(n) { return `Rs. ${(n ?? 0).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` }
function fmtTime(iso) { return iso ? new Date(iso).toLocaleTimeString('en-LK', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—' }

// ─── Explanation Panel ────────────────────────────────────────────────────────

function ExplanationPanel({ explanation, fraudProb }) {
  const [open, setOpen] = useState(false)
  if (!explanation || !explanation.factors?.length) return null
  const { summary, factors } = explanation
  const impactBg = { HIGH: 'bg-red-100 text-red-700 border-red-200', MEDIUM: 'bg-amber-100 text-amber-700 border-amber-200', LOW: 'bg-gray-100 text-gray-600 border-gray-200' }

  return (
    <div className={clsx('mt-2 rounded-xl border overflow-hidden text-xs',
      fraudProb >= 0.80 ? 'border-red-300' : 'border-amber-200')}>
      <button
        onClick={e => { e.stopPropagation(); setOpen(o => !o) }}
        className={clsx('w-full flex items-center justify-between px-3 py-2 font-medium transition-colors',
          fraudProb >= 0.80 ? 'bg-red-50 text-red-800 hover:bg-red-100' : 'bg-amber-50 text-amber-800 hover:bg-amber-100')}
      >
        <div className="flex items-center gap-1.5">
          <Info size={11} />
          Why was this flagged?
        </div>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && (
        <div className="px-3 py-2 bg-white space-y-2">
          <p className="text-[11px] text-gray-600 leading-relaxed">{summary}</p>
          {factors.map((f, i) => (
            <div key={i} className={clsx('flex items-start gap-2 px-2 py-1.5 rounded-lg border',
              f.direction === 'increases_risk' ? 'bg-red-50 border-red-100' : 'bg-green-50 border-green-100')}>
              {f.direction === 'increases_risk'
                ? <ArrowUp size={11} className="text-red-500 mt-0.5 flex-shrink-0" />
                : <ArrowDown size={11} className="text-green-500 mt-0.5 flex-shrink-0" />}
              <div>
                <div className="flex items-center gap-1.5">
                  <span className={clsx('font-semibold', f.direction === 'increases_risk' ? 'text-red-700' : 'text-green-700')}>
                    {f.name}
                  </span>
                  {f.direction === 'increases_risk' && (
                    <span className={clsx('text-[9px] font-bold px-1 py-0.5 rounded-full border', impactBg[f.impact] || impactBg.LOW)}>
                      {f.impact}
                    </span>
                  )}
                </div>
                <p className="text-[10px] text-gray-500">{f.description}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function FraudDetection() {
  const { data: liveTx, refetch, loading: loadingLive } = useApi('/transactions/live', {
    poll: true, interval: 4000, params: { n: 20 }
  })
  const { data: alertsData, refetch: refetchAlerts } = useApi('/fraud/alerts', {
    poll: true, interval: 6000, params: { n: 8 }
  })

  const [selectedTx, setSelectedTx] = useState(null)

  const transactions = liveTx?.transactions || []
  const alerts       = alertsData?.alerts    || []

  const criticals = transactions.filter(t => t.status === 'CRITICAL').length
  const highs      = transactions.filter(t => t.status === 'HIGH').length
  const mediums    = transactions.filter(t => t.status === 'MEDIUM').length
  const normals    = transactions.filter(t => t.status === 'NORMAL').length

  const probColor = (p) => p >= 0.75 ? 'text-red-600' : p >= 0.51 ? 'text-red-500' : p >= 0.25 ? 'text-amber-600' : 'text-green-600'
  const probBg    = (p) => p >= 0.75 ? 'bg-red-500' : p >= 0.51 ? 'bg-orange-400' : p >= 0.25 ? 'bg-amber-400' : 'bg-green-400'
  const rowBg     = (status) => ({
    CRITICAL: 'bg-red-50/90 border-l-2 border-l-red-500',
    HIGH:     'bg-red-50/50 border-l-2 border-l-red-300',
    MEDIUM:   'bg-amber-50/40 border-l-2 border-l-amber-300',
    NORMAL:   '',
  })[status] || ''

  const handleRefresh = () => { refetch(); refetchAlerts() }

  return (
    <div className="p-6 space-y-5 fade-up">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Fraud Detection Monitor</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Real model scores · LightGBM 2M-Clean · Threshold: 0.51 · Dataset fraud rate: 3.59%
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 bg-green-50 border border-green-200 px-3 py-1.5 rounded-xl">
            <span className="w-1.5 h-1.5 bg-green-500 rounded-full pulse-dot" />
            <span className="text-xs text-green-700 font-medium">Live · Real Data</span>
          </div>
          <button onClick={handleRefresh}
            className="flex items-center gap-2 px-3 py-2 rounded-xl border border-gray-200 text-sm text-gray-500 hover:bg-gray-50 transition-all">
            <RefreshCw size={13} /> Refresh
          </button>
        </div>
      </div>

      {/* Status chips */}
      <div className="grid grid-cols-4 gap-3 fade-up-1">
        {[
          { label: 'CRITICAL', count: criticals, color: 'bg-red-500',    text: 'text-red-700',    bg: 'bg-red-50 border-red-200' },
          { label: 'HIGH',     count: highs,     color: 'bg-orange-400', text: 'text-orange-700', bg: 'bg-orange-50 border-orange-200' },
          { label: 'MEDIUM',   count: mediums,   color: 'bg-amber-400',  text: 'text-amber-700',  bg: 'bg-amber-50 border-amber-200' },
          { label: 'NORMAL',   count: normals,   color: 'bg-green-400',  text: 'text-green-700',  bg: 'bg-green-50 border-green-200' },
        ].map(({ label, count, color, text, bg }) => (
          <div key={label} className={clsx('card p-4 border flex items-center gap-3', bg)}>
            <div className={clsx('w-3 h-3 rounded-full', color, count > 0 ? 'pulse-dot' : '')} />
            <div>
              <p className={clsx('text-xl font-bold', text)}>{count}</p>
              <p className="text-xs font-medium text-gray-500">{label}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-5 fade-up-2">

        {/* Live Transaction Table */}
        <div className="card col-span-2 overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity size={15} className="text-blue-500" />
              <p className="text-sm font-semibold text-gray-900">Live Transaction Stream</p>
              <span className="text-[10px] text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">real model scores</span>
            </div>
            <div className="flex items-center gap-1.5">
              {loadingLive && <span className="text-[10px] text-blue-400">updating…</span>}
              <span className="w-1.5 h-1.5 bg-green-500 rounded-full pulse-dot" />
              <span className="text-xs text-gray-400">Auto-refreshes every 4s</span>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-100">
                  {['Tx ID', 'Account', 'Amount (Rs.)', 'Device', 'Merchant', 'Fraud Prob', 'Status'].map(h => (
                    <th key={h} className="px-4 py-3 text-left text-[10px] font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {transactions.map((tx) => (
                  <tr
                    key={tx.tx_id}
                    onClick={() => setSelectedTx(tx === selectedTx ? null : tx)}
                    className={clsx('border-b border-gray-50 cursor-pointer hover:brightness-95 transition-all',
                      rowBg(tx.status),
                      selectedTx?.tx_id === tx.tx_id && 'ring-1 ring-inset ring-blue-300')}
                  >
                    <td className="px-4 py-2.5 font-mono font-medium text-gray-700 whitespace-nowrap">{tx.tx_id}</td>
                    <td className="px-4 py-2.5">
                      <p className="font-medium text-gray-800 text-[11px] truncate max-w-[100px]">{tx.account_name || tx.account_id}</p>
                      <p className="text-[10px] text-gray-400 font-mono">{tx.account_id}</p>
                    </td>
                    <td className="px-4 py-2.5 font-semibold text-gray-800 font-mono tabular-nums">{fmtMoney(tx.amount)}</td>
                    <td className="px-4 py-2.5 text-gray-500 capitalize">{tx.device}</td>
                    <td className="px-4 py-2.5 text-gray-500 capitalize">{tx.merchant}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="w-14 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                          <div className={clsx('h-full rounded-full transition-all', probBg(tx.fraud_probability))}
                            style={{ width: `${tx.fraud_probability * 100}%` }} />
                        </div>
                        <span className={clsx('font-mono font-bold', probColor(tx.fraud_probability))}>
                          {(tx.fraud_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-1">
                        <RiskBadge level={tx.status} />
                        {(tx.status === 'HIGH' || tx.status === 'CRITICAL') && tx.fraud_explanation && (
                          <span className="text-[9px] text-blue-500">▼</span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-2 bg-gray-50/50 border-t border-gray-100 text-[10px] text-gray-400">
            Showing {transactions.length} real transactions scored by AFI LightGBM · Click flagged rows to inspect
          </div>
        </div>

        {/* Right Panel: Detail + Alerts */}
        <div className="space-y-4">

          {/* Transaction Detail */}
          {selectedTx ? (
            <div className={clsx('card border-2 overflow-hidden',
              selectedTx.status === 'CRITICAL' ? 'border-red-300' :
              selectedTx.status === 'HIGH'     ? 'border-orange-200' :
              selectedTx.status === 'MEDIUM'   ? 'border-amber-200' : 'border-blue-100')}>
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-semibold text-gray-900">Transaction Detail</p>
                  <RiskBadge level={selectedTx.status} />
                </div>
                <button onClick={() => setSelectedTx(null)} className="text-gray-300 hover:text-gray-500 text-xs">✕</button>
              </div>
              <div className="p-4 space-y-2">
                {/* Donut */}
                <div className="flex items-center justify-center py-2">
                  <div className="relative w-20 h-20">
                    <svg viewBox="0 0 80 80" className="w-full h-full -rotate-90">
                      <circle cx="40" cy="40" r="32" fill="none" stroke="#F1F5F9" strokeWidth="8" />
                      <circle cx="40" cy="40" r="32" fill="none"
                        stroke={selectedTx.fraud_probability >= 0.75 ? '#EF4444' : selectedTx.fraud_probability >= 0.25 ? '#F59E0B' : '#22C55E'}
                        strokeWidth="8"
                        strokeDasharray={`${selectedTx.fraud_probability * 200.96} 200.96`}
                        strokeLinecap="round" />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <span className="text-base font-bold text-gray-900">
                        {(selectedTx.fraud_probability * 100).toFixed(0)}%
                      </span>
                      <span className="text-[9px] text-gray-400">fraud prob</span>
                    </div>
                  </div>
                  <div className="ml-4 space-y-1">
                    <div>
                      <p className="text-[10px] text-gray-400">Credit Score</p>
                      <p className="text-sm font-bold text-gray-800">{selectedTx.credit_score?.toFixed(1)}/100</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-gray-400">Amount</p>
                      <p className="text-sm font-bold text-gray-800">{fmtMoney(selectedTx.amount)}</p>
                    </div>
                  </div>
                </div>

                {/* Details */}
                <div className="space-y-1.5">
                  {[
                    ['Tx ID',    selectedTx.tx_id],
                    ['Account',  selectedTx.account_name || selectedTx.account_id],
                    ['Receiver', selectedTx.receiver_id],
                    ['Device',   selectedTx.device],
                    ['Merchant', selectedTx.merchant],
                    ['Channel',  selectedTx.payment_channel],
                    ['Type',     selectedTx.tx_type],
                    ['Time',     fmtTime(selectedTx.timestamp)],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between text-xs">
                      <span className="text-gray-400">{k}</span>
                      <span className="font-semibold text-gray-800 font-mono capitalize truncate max-w-[120px]">{v}</span>
                    </div>
                  ))}
                </div>

                {/* Fraud explanation inline */}
                {selectedTx.fraud_explanation && (selectedTx.status === 'HIGH' || selectedTx.status === 'CRITICAL') && (
                  <ExplanationPanel explanation={selectedTx.fraud_explanation} fraudProb={selectedTx.fraud_probability} />
                )}

                {selectedTx.status === 'NORMAL' && (
                  <div className="mt-2 bg-green-50 border border-green-200 rounded-xl px-3 py-2 text-xs text-green-700 flex items-center gap-2">
                    <Shield size={12} />
                    Transaction appears legitimate. No fraud indicators detected.
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card p-5 border-2 border-dashed border-gray-200 flex flex-col items-center justify-center h-48 text-gray-300">
              <Shield size={32} strokeWidth={1} />
              <p className="text-xs mt-2 text-gray-400 text-center">Click any row to inspect the transaction and see fraud explanation</p>
            </div>
          )}

          {/* Active Fraud Alerts */}
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle size={14} className="text-red-500" />
              <p className="text-sm font-semibold text-gray-900">Top Fraud Alerts</p>
              <span className="text-[10px] bg-red-50 text-red-600 border border-red-100 px-1.5 py-0.5 rounded-full font-semibold">
                {alerts.length}
              </span>
            </div>
            <div className="space-y-2.5 max-h-[340px] overflow-y-auto">
              {alerts.length === 0 ? (
                <p className="text-xs text-gray-400 text-center py-4">No fraud alerts — data loading…</p>
              ) : alerts.map((a) => (
                <div
                  key={a.tx_id}
                  className={clsx('rounded-xl p-3 space-y-1 border cursor-pointer hover:brightness-95 transition-all',
                    a.status === 'CRITICAL'
                      ? 'bg-red-50/80 border-red-200'
                      : 'bg-orange-50/70 border-orange-200')}
                  onClick={() => setSelectedTx(a === selectedTx ? null : a)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-xs font-bold text-gray-800">{a.account_name || a.account_id}</span>
                      <span className="text-[10px] text-gray-400 ml-1 font-mono">{a.tx_id}</span>
                    </div>
                    <span className="text-xs font-bold text-red-600">{fmtMoney(a.amount)}</span>
                  </div>
                  <p className="text-[10px] text-gray-500">{a.reason}</p>
                  <p className="text-[10px] text-gray-400 capitalize">{a.merchant} · {a.device} · {fmtTime(a.timestamp)}</p>
                  <div className="flex items-center gap-1.5">
                    <div className="flex-1 h-1 bg-gray-200 rounded-full">
                      <div className="h-1 bg-red-500 rounded-full" style={{ width: `${a.fraud_probability * 100}%` }} />
                    </div>
                    <span className="text-[10px] font-mono font-bold text-red-600">
                      {(a.fraud_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  {a.fraud_explanation && (
                    <ExplanationPanel explanation={a.fraud_explanation} fraudProb={a.fraud_probability} />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}