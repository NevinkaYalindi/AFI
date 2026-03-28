import { useState, useEffect, useCallback } from 'react'
import {
  CreditCard, Search, TrendingUp, TrendingDown,
  AlertTriangle, CheckCircle, Clock, DollarSign,
  Activity, Shield, ChevronRight, RefreshCw, Eye,
  BarChart3, Calendar, MapPin, Smartphone, Info,
  ArrowUp, ArrowDown, ChevronDown, ChevronUp
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import GaugeChart from '../components/GaugeChart'
import RiskBadge from '../components/RiskBadge'
import axios from 'axios'
import clsx from 'clsx'

const BASE = (import.meta.env.VITE_API_URL || '') + '/api'

// ─── Helpers ──────────────────────────────────────────────────────────────────

const tierColor = {
  EXCELLENT: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', bar: '#22C55E' },
  GOOD:      { bg: 'bg-blue-50',    text: 'text-blue-700',    border: 'border-blue-200',    bar: '#60A5FA' },
  FAIR:      { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-200',   bar: '#F59E0B' },
  POOR:      { bg: 'bg-red-50',     text: 'text-red-700',     border: 'border-red-200',     bar: '#EF4444' },
}
const statusDot = { ACTIVE: 'bg-green-500', UNDER_REVIEW: 'bg-amber-500' }

function fmtMoney(n) { return `Rs. ${(n ?? 0).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` }
function fmtDate(iso) { return iso ? new Date(iso).toLocaleDateString('en-LK', { day: '2-digit', month: 'short', year: 'numeric' }) : '—' }
function fmtTime(iso) { return iso ? new Date(iso).toLocaleTimeString('en-LK', { hour: '2-digit', minute: '2-digit' }) : '—' }
function creditTierLookup(s) { return s >= 80 ? 'EXCELLENT' : s >= 60 ? 'GOOD' : s >= 40 ? 'FAIR' : 'POOR' }
function scoreChange(history) {
  if (!history || history.length < 2) return 0
  return +(history[history.length - 1].score - history[history.length - 2].score).toFixed(1)
}

// ─── Explanation Panel ────────────────────────────────────────────────────────

function CreditExplanation({ explanation }) {
  const [open, setOpen] = useState(true)
  if (!explanation) return null
  const { summary, factors = [] } = explanation
  return (
    <div className="card border border-blue-100 overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-blue-50/60 hover:bg-blue-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Info size={14} className="text-blue-600" />
          <p className="text-xs font-semibold text-blue-800">Why this credit score?</p>
        </div>
        {open ? <ChevronUp size={14} className="text-blue-400" /> : <ChevronDown size={14} className="text-blue-400" />}
      </button>
      {open && (
        <div className="px-4 pb-4 pt-3">
          <p className="text-xs text-gray-600 mb-3 leading-relaxed">{summary}</p>
          <div className="space-y-2">
            {factors.map((f, i) => (
              <div key={i} className={clsx(
                'flex items-start gap-2.5 px-3 py-2 rounded-xl text-xs',
                f.impact === 'positive' ? 'bg-green-50 border border-green-100' : 'bg-red-50 border border-red-100'
              )}>
                {f.impact === 'positive'
                  ? <ArrowUp size={13} className="text-green-600 mt-0.5 flex-shrink-0" />
                  : <ArrowDown size={13} className="text-red-500 mt-0.5 flex-shrink-0" />}
                <div>
                  <p className={clsx('font-semibold', f.impact === 'positive' ? 'text-green-700' : 'text-red-700')}>
                    {f.name}
                  </p>
                  <p className="text-gray-500 text-[11px] mt-0.5">{f.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function FraudExplanation({ explanation, fraudProb }) {
  const [open, setOpen] = useState(true)
  if (!explanation || !explanation.factors?.length) return null
  const { summary, factors } = explanation
  const levelColor = fraudProb >= 0.80 ? 'red' : fraudProb > 0.51 ? 'red' : 'amber'
  const impactBg = { HIGH: 'bg-red-100 text-red-700', MEDIUM: 'bg-amber-100 text-amber-700', LOW: 'bg-gray-100 text-gray-600' }
  return (
    <div className={clsx('card overflow-hidden border', levelColor === 'red' ? 'border-red-200' : 'border-amber-200')}>
      <button
        onClick={() => setOpen(o => !o)}
        className={clsx('w-full flex items-center justify-between px-4 py-3 transition-colors',
          levelColor === 'red' ? 'bg-red-50/60 hover:bg-red-50' : 'bg-amber-50/60 hover:bg-amber-50')}
      >
        <div className="flex items-center gap-2">
          <AlertTriangle size={14} className={levelColor === 'red' ? 'text-red-600' : 'text-amber-600'} />
          <p className={clsx('text-xs font-semibold', levelColor === 'red' ? 'text-red-800' : 'text-amber-800')}>
            Why was this flagged? ({(fraudProb * 100).toFixed(0)}% fraud probability)
          </p>
        </div>
        {open ? <ChevronUp size={14} className="text-gray-400" /> : <ChevronDown size={14} className="text-gray-400" />}
      </button>
      {open && (
        <div className="px-4 pb-4 pt-3">
          <p className="text-xs text-gray-600 mb-3 leading-relaxed">{summary}</p>
          <div className="space-y-2">
            {factors.map((f, i) => (
              <div key={i} className={clsx(
                'flex items-start gap-2.5 px-3 py-2 rounded-xl border text-xs',
                f.direction === 'increases_risk'
                  ? 'bg-red-50/70 border-red-100'
                  : 'bg-green-50/70 border-green-100'
              )}>
                {f.direction === 'increases_risk'
                  ? <AlertTriangle size={12} className="text-red-500 mt-0.5 flex-shrink-0" />
                  : <CheckCircle size={12} className="text-green-500 mt-0.5 flex-shrink-0" />}
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <p className={clsx('font-semibold', f.direction === 'increases_risk' ? 'text-red-700' : 'text-green-700')}>
                      {f.name}
                    </p>
                    {f.direction === 'increases_risk' && (
                      <span className={clsx('text-[10px] font-bold px-1.5 py-0.5 rounded-full', impactBg[f.impact] || impactBg.LOW)}>
                        {f.impact}
                      </span>
                    )}
                  </div>
                  <p className="text-gray-500 text-[11px] mt-0.5">{f.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function AccountCard({ account, selected, onClick }) {
  const tier = tierColor[account.credit_tier] || tierColor.FAIR
  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-full text-left px-4 py-3.5 rounded-xl border transition-all duration-150',
        selected ? 'bg-[#1E3A8A] border-[#1E3A8A] shadow-md' : 'bg-white border-gray-100 hover:border-blue-200 hover:shadow-sm'
      )}
    >
      <div className="flex items-center gap-3">
        <div className={clsx('w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold',
          selected ? 'bg-white/20 text-white' : `${tier.bg} ${tier.text}`)}>
          {account.name.split(' ').map(w => w[0]).slice(0, 2).join('')}
        </div>
        <div className="flex-1 min-w-0">
          <p className={clsx('text-xs font-semibold truncate', selected ? 'text-white' : 'text-gray-900')}>{account.name}</p>
          <p className={clsx('text-[10px] font-mono', selected ? 'text-blue-200' : 'text-gray-400')}>{account.id}</p>
        </div>
        <div className="text-right flex-shrink-0">
          <p className={clsx('text-sm font-bold tabular-nums', selected ? 'text-white' : tier.text)}>
            {account.credit_score?.toFixed(0)}
          </p>
          <div className="flex items-center gap-1 justify-end">
            <span className={clsx('w-1.5 h-1.5 rounded-full', selected ? 'bg-green-300' : statusDot[account.status] || 'bg-gray-400')} />
            <p className={clsx('text-[9px] font-medium', selected ? 'text-blue-200' : 'text-gray-400')}>
              {account.fraud_alerts > 0 ? `${account.fraud_alerts} alerts` : account.status?.replace('_', ' ')}
            </p>
          </div>
        </div>
      </div>
    </button>
  )
}

function MetricPill({ icon: Icon, label, value, sub, color = 'blue' }) {
  const colors = {
    blue:   'bg-blue-50 text-blue-600 border-blue-100',
    green:  'bg-green-50 text-green-600 border-green-100',
    amber:  'bg-amber-50 text-amber-600 border-amber-100',
    red:    'bg-red-50 text-red-600 border-red-100',
    indigo: 'bg-indigo-50 text-indigo-600 border-indigo-100',
  }
  return (
    <div className="card p-4 flex items-start gap-3">
      <div className={clsx('w-8 h-8 rounded-lg flex items-center justify-center border flex-shrink-0', colors[color])}>
        {Icon && <Icon size={15} strokeWidth={2} />}
      </div>
      <div>
        <p className="text-lg font-bold text-gray-900 leading-none">{value}</p>
        <p className="text-xs text-gray-500 mt-0.5">{label}</p>
        {sub && <p className="text-[10px] text-gray-400 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

function TxRow({ tx }) {
  const [showExpl, setShowExpl] = useState(false)
  const probColor = tx.fraud_probability >= 0.5 ? 'text-red-600' : tx.fraud_probability >= 0.25 ? 'text-amber-600' : 'text-green-600'
  const probBg    = tx.fraud_probability >= 0.5 ? 'bg-red-500' : tx.fraud_probability >= 0.25 ? 'bg-amber-400' : 'bg-green-400'
  const isHighRisk = tx.status === 'HIGH' || tx.status === 'CRITICAL'
  return (
    <>
      <tr
        className={clsx('border-b border-gray-50 text-xs', isHighRisk ? 'bg-red-50/30' : 'hover:bg-blue-50/20', 'cursor-pointer')}
        onClick={() => isHighRisk && setShowExpl(o => !o)}
      >
        <td className="px-3 py-2.5 font-mono text-gray-500 whitespace-nowrap">{fmtTime(tx.timestamp)}</td>
        <td className="px-3 py-2.5 font-mono text-gray-700">{tx.tx_id}</td>
        <td className="px-3 py-2.5 text-gray-600 capitalize">{tx.tx_type}</td>
        <td className="px-3 py-2.5 text-gray-600 capitalize">{tx.merchant}</td>
        <td className="px-3 py-2.5 font-semibold text-gray-900 tabular-nums">{fmtMoney(tx.amount)}</td>
        <td className="px-3 py-2.5">
          <div className="flex items-center gap-1.5">
            <div className="w-12 h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div className={clsx('h-full rounded-full', probBg)} style={{ width: `${tx.fraud_probability * 100}%` }} />
            </div>
            <span className={clsx('font-mono font-semibold', probColor)}>{(tx.fraud_probability * 100).toFixed(0)}%</span>
          </div>
        </td>
        <td className="px-3 py-2.5">
          <div className="flex items-center gap-1.5">
            <RiskBadge level={tx.status} />
            {isHighRisk && (
              <span className="text-[9px] text-blue-500 font-medium">{showExpl ? '▲ hide' : '▼ why?'}</span>
            )}
          </div>
        </td>
      </tr>
      {showExpl && isHighRisk && tx.fraud_explanation && (
        <tr className="bg-red-50/20">
          <td colSpan={7} className="px-4 py-3">
            <FraudExplanation explanation={tx.fraud_explanation} fraudProb={tx.fraud_probability} />
          </td>
        </tr>
      )}
    </>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl px-3 py-2 shadow-lg text-xs">
      <p className="text-gray-500 mb-1">{label}</p>
      <p className="font-bold text-[#1E3A8A]">{payload[0].value?.toFixed(1)} / 100</p>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function CreditScoring() {
  const [accounts, setAccounts]           = useState([])
  const [search, setSearch]               = useState('')
  const [selectedId, setSelectedId]       = useState(null)
  const [accountData, setAccountData]     = useState(null)
  const [loadingList, setLoadingList]     = useState(true)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [activeTab, setActiveTab]         = useState('overview')
  const [refreshing, setRefreshing]       = useState(false)

  const fetchAccounts = useCallback(async (q = '') => {
    setLoadingList(true)
    try {
      const res = await axios.get(`${BASE}/credit/accounts`, { params: q ? { search: q } : {} })
      setAccounts(res.data.accounts || [])
    } catch { setAccounts([]) }
    finally { setLoadingList(false) }
  }, [])

  useEffect(() => { fetchAccounts() }, [fetchAccounts])
  useEffect(() => {
    const t = setTimeout(() => fetchAccounts(search), 300)
    return () => clearTimeout(t)
  }, [search, fetchAccounts])

  const fetchDetail = useCallback(async (id) => {
    setLoadingDetail(true)
    setAccountData(null)
    try {
      const res = await axios.get(`${BASE}/credit/account/${id}`)
      setAccountData(res.data)
    } catch { setAccountData(null) }
    finally { setLoadingDetail(false) }
  }, [])

  useEffect(() => {
    if (selectedId) fetchDetail(selectedId)
  }, [selectedId, fetchDetail])

  const handleRefresh = async () => {
    setRefreshing(true)
    if (selectedId) await fetchDetail(selectedId)
    setRefreshing(false)
  }

  const tier    = accountData ? (tierColor[accountData.credit_tier] || tierColor.FAIR) : tierColor.GOOD
  const hist    = accountData?.credit_score_history || []
  const delta   = scoreChange(hist)
  const txns    = accountData?.transactions || []
  const highRiskTxns = txns.filter(t => t.status === 'HIGH' || t.status === 'CRITICAL')

  return (
    <div className="flex h-full overflow-hidden fade-up" style={{ height: 'calc(100vh - 56px)' }}>

      {/* ── Left Sidebar ────────────────────────────────────────────────── */}
      <aside className="w-72 flex-shrink-0 bg-white border-r border-gray-100 flex flex-col overflow-hidden">
        <div className="px-4 py-4 border-b border-gray-100">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-bold text-gray-900">Account Portfolio</p>
            <span className="text-[10px] font-semibold bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full border border-blue-100">
              {accounts.length} accounts
            </span>
          </div>
          <div className="relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search by name or account ID…"
              className="w-full pl-8 pr-3 py-2 text-xs border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-300 bg-gray-50"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto py-2 px-3 space-y-1.5">
          {loadingList
            ? Array(8).fill(0).map((_, i) => <div key={i} className="h-14 rounded-xl bg-gray-100 animate-pulse" />)
            : accounts.length === 0
            ? <p className="text-center text-xs text-gray-400 py-8">No accounts found</p>
            : accounts.map(acct => (
                <AccountCard
                  key={acct.id}
                  account={acct}
                  selected={selectedId === acct.id}
                  onClick={() => { setSelectedId(acct.id); setActiveTab('overview') }}
                />
              ))
          }
        </div>

        {/* Legend */}
        <div className="px-4 py-3 border-t border-gray-100 bg-gray-50">
          <p className="text-[10px] text-gray-400 font-semibold mb-1.5">CREDIT SCORE LEGEND</p>
          {[
            { label: 'Excellent', range: '80–100', color: '#22C55E' },
            { label: 'Good',      range: '60–79',  color: '#60A5FA' },
            { label: 'Fair',      range: '40–59',  color: '#F59E0B' },
            { label: 'Poor',      range: '0–39',   color: '#EF4444' },
          ].map(({ label, range, color }) => (
            <div key={label} className="flex items-center justify-between text-[10px] mb-1">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: color }} />
                <span className="text-gray-600 font-medium">{label}</span>
              </div>
              <span className="text-gray-400 font-mono">{range}</span>
            </div>
          ))}
        </div>
      </aside>

      {/* ── Main Panel ──────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        {!selectedId ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-300">
            <div className="w-20 h-20 rounded-2xl bg-gray-100 flex items-center justify-center mb-4">
              <CreditCard size={36} strokeWidth={1} className="text-gray-300" />
            </div>
            <p className="text-sm font-semibold text-gray-400">Select an account to view credit details</p>
            <p className="text-xs text-gray-300 mt-1">All data is real — scored by AFI LightGBM models</p>
          </div>
        ) : loadingDetail ? (
          <div className="p-6 space-y-4 animate-pulse">
            <div className="h-8 bg-gray-100 rounded-xl w-64" />
            <div className="grid grid-cols-4 gap-4">{[...Array(4)].map((_, i) => <div key={i} className="h-20 bg-gray-100 rounded-xl" />)}</div>
            <div className="h-48 bg-gray-100 rounded-xl" />
          </div>
        ) : accountData ? (
          <div className="p-6 space-y-5">

            {/* Header */}
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-4">
                <div className={clsx('w-14 h-14 rounded-2xl flex items-center justify-center text-lg font-bold border-2',
                  tier.bg, tier.text, tier.border)}>
                  {accountData.name.split(' ').map(w => w[0]).slice(0, 2).join('')}
                </div>
                <div>
                  <h1 className="text-xl font-extrabold text-gray-900">{accountData.name}</h1>
                  <div className="flex items-center gap-3 mt-0.5 flex-wrap">
                    <span className="text-xs text-gray-500 font-mono">{accountData.id}</span>
                    <span className="text-gray-300">·</span>
                    <span className="text-xs text-gray-500">{accountData.account_type}</span>
                    <span className="text-gray-300">·</span>
                    <div className="flex items-center gap-1">
                      <span className={clsx('w-1.5 h-1.5 rounded-full', statusDot[accountData.status] || 'bg-gray-400')} />
                      <span className="text-xs text-gray-500">{accountData.status?.replace('_', ' ')}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <div className="flex items-center gap-1 text-[10px] text-gray-400">
                      <MapPin size={10} /> {accountData.location}
                    </div>
                    <div className="flex items-center gap-1 text-[10px] text-gray-400">
                      <Calendar size={10} /> Since {new Date(accountData.join_date).getFullYear()}
                    </div>
                    {accountData.fraud_alerts > 0 && (
                      <div className="flex items-center gap-1 text-[10px] text-red-500 font-semibold">
                        <AlertTriangle size={10} /> {accountData.fraud_alerts} fraud alerts
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <button
                onClick={handleRefresh} disabled={refreshing}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-gray-200 text-xs text-gray-500 hover:bg-gray-50 transition-all"
              >
                <RefreshCw size={12} className={refreshing ? 'animate-spin' : ''} /> Refresh
              </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 bg-gray-100 p-1 rounded-xl w-fit">
              {[
                { id: 'overview',     label: 'Overview',      icon: Eye },
                { id: 'history',      label: 'Score History', icon: TrendingUp },
                { id: 'transactions', label: 'Transactions',  icon: Activity },
                { id: 'risk',         label: 'Risk Analysis', icon: Shield },
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={clsx(
                    'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                    activeTab === id ? 'bg-white text-[#1E3A8A] shadow-sm font-semibold' : 'text-gray-500 hover:text-gray-700'
                  )}
                >
                  <Icon size={12} /> {label}
                  {id === 'transactions' && highRiskTxns.length > 0 && (
                    <span className="bg-red-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full ml-0.5">
                      {highRiskTxns.length}
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* ── OVERVIEW ── */}
            {activeTab === 'overview' && (
              <div className="space-y-4 fade-up">
                <div className="grid grid-cols-3 gap-4">
                  {/* Gauge */}
                  <div className="card p-5 col-span-1 flex flex-col items-center">
                    <p className="text-xs font-semibold text-gray-500 mb-1 self-start">CREDIT SCORE</p>
                    <GaugeChart score={accountData.credit_score} size={200} />
                    <div className={clsx('mt-2 flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold',
                      tier.bg, tier.text, tier.border)}>
                      {delta >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                      {delta >= 0 ? '+' : ''}{delta} pts this period
                    </div>
                  </div>
                  {/* Key Metrics */}
                  <div className="col-span-2 grid grid-cols-2 gap-3">
                    <MetricPill icon={Activity}      label="Transactions"     value={accountData.total_transactions}
                      sub={`Avg: ${fmtMoney(accountData.avg_amount)}`} color="blue" />
                    <MetricPill icon={DollarSign}    label="Total Volume"     value={fmtMoney(accountData.total_volume)}
                      sub="Cumulative" color="indigo" />
                    <MetricPill icon={AlertTriangle} label="Fraud Alerts"     value={accountData.fraud_alerts}
                      sub={accountData.fraud_alerts > 0 ? 'Flagged transactions' : 'No alerts raised'}
                      color={accountData.fraud_alerts > 0 ? 'red' : 'green'} />
                    <MetricPill icon={Clock}         label="Fraud Risk"       value={`${(accountData.fraud_risk * 100).toFixed(1)}%`}
                      sub={accountData.fraud_risk_level} color={accountData.fraud_risk > 0.5 ? 'red' : accountData.fraud_risk > 0.25 ? 'amber' : 'green'} />
                    <MetricPill icon={BarChart3}     label="Fraud History"    value={`${(accountData.fraud_history_ratio * 100).toFixed(2)}%`}
                      sub="Historical fraud rate" color={accountData.fraud_history_ratio > 0.05 ? 'red' : 'green'} />
                    <MetricPill icon={DollarSign}    label="Max Transaction"  value={fmtMoney(accountData.max_amount)}
                      sub="Single largest" color="blue" />
                  </div>
                </div>

                {/* Recommendation Banner */}
                <div className={clsx('px-4 py-3 rounded-xl border flex items-center gap-3',
                  accountData.recommendation.startsWith('APPROVE')
                    ? 'bg-green-50 border-green-200'
                    : accountData.recommendation.startsWith('REVIEW')
                    ? 'bg-amber-50 border-amber-200'
                    : 'bg-red-50 border-red-200'
                )}>
                  {accountData.recommendation.startsWith('APPROVE')
                    ? <CheckCircle size={16} className="text-green-600 flex-shrink-0" />
                    : <AlertTriangle size={16} className="text-amber-600 flex-shrink-0" />}
                  <p className={clsx('text-sm font-bold',
                    accountData.recommendation.startsWith('APPROVE') ? 'text-green-800'
                    : accountData.recommendation.startsWith('REVIEW') ? 'text-amber-800'
                    : 'text-red-800')}>{accountData.recommendation}</p>
                </div>

                {/* Credit Explanation */}
                <CreditExplanation explanation={accountData.credit_explanation} />

                {/* Fraud Explanation (if account is high risk) */}
                {accountData.fraud_risk > 0.25 && (
                  <FraudExplanation explanation={accountData.fraud_explanation} fraudProb={accountData.fraud_risk} />
                )}

                {/* Recent Transactions Preview */}
                <div className="card overflow-hidden">
                  <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
                    <p className="text-sm font-semibold text-gray-900">Recent Transactions</p>
                    <button onClick={() => setActiveTab('transactions')}
                      className="flex items-center gap-1 text-xs text-blue-600 font-medium hover:text-blue-800">
                      View all <ChevronRight size={12} />
                    </button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-gray-50 border-b border-gray-100">
                          {['Time', 'Tx ID', 'Type', 'Merchant', 'Amount', 'Fraud Prob', 'Status'].map(h => (
                            <th key={h} className="px-3 py-2.5 text-left text-[10px] font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {txns.slice(0, 6).map(tx => <TxRow key={tx.tx_id} tx={tx} />)}
                      </tbody>
                    </table>
                  </div>
                  {highRiskTxns.length > 0 && (
                    <div className="px-5 py-2 bg-red-50/50 border-t border-red-100 text-xs text-red-600 font-medium">
                      ⚠ Click on flagged rows (highlighted in red) to see why they were caught
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ── SCORE HISTORY ── */}
            {activeTab === 'history' && (
              <div className="space-y-4 fade-up">
                <div className="card p-5">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className="text-sm font-semibold text-gray-900">Credit Score History</p>
                      <p className="text-xs text-gray-400 mt-0.5">
                        Current: <strong>{accountData.credit_score?.toFixed(1)}</strong>/100 ·{' '}
                        Change: <strong className={delta >= 0 ? 'text-green-600' : 'text-red-500'}>
                          {delta >= 0 ? '+' : ''}{delta} pts
                        </strong>
                      </p>
                    </div>
                    <span className={clsx('text-xs font-bold px-3 py-1 rounded-full border', tier.bg, tier.text, tier.border)}>
                      {accountData.credit_tier}
                    </span>
                  </div>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={hist} margin={{ left: -20, right: 10, top: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                      <XAxis dataKey="month" tick={{ fontSize: 10, fill: '#9CA3AF' }} tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#9CA3AF' }} tickLine={false} axisLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={80} stroke="#22C55E" strokeDasharray="4 2" strokeWidth={1} label={{ value: 'Excellent', position: 'insideTopRight', fontSize: 9, fill: '#22C55E' }} />
                      <ReferenceLine y={60} stroke="#60A5FA" strokeDasharray="4 2" strokeWidth={1} label={{ value: 'Good', position: 'insideTopRight', fontSize: 9, fill: '#60A5FA' }} />
                      <ReferenceLine y={40} stroke="#F59E0B" strokeDasharray="4 2" strokeWidth={1} label={{ value: 'Fair', position: 'insideTopRight', fontSize: 9, fill: '#F59E0B' }} />
                      <Line type="monotone" dataKey="score" stroke="#1E3A8A" strokeWidth={2.5}
                        dot={{ fill: '#1E3A8A', r: 3 }} activeDot={{ r: 5, fill: '#2563EB' }} name="Credit Score" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Credit explanation in history tab too */}
                <CreditExplanation explanation={accountData.credit_explanation} />
              </div>
            )}

            {/* ── TRANSACTIONS ── */}
            {activeTab === 'transactions' && (
              <div className="space-y-4 fade-up">
                <div className="grid grid-cols-4 gap-3">
                  {[
                    { label: 'Total Transactions', value: txns.length, color: 'blue' },
                    { label: 'Total Volume',        value: fmtMoney(txns.reduce((s,t) => s+t.amount, 0)), color: 'indigo' },
                    { label: 'Fraud Flagged',       value: txns.filter(t => t.status !== 'NORMAL').length, color: 'red' },
                    { label: 'Avg Amount',          value: fmtMoney(txns.reduce((s,t) => s+t.amount, 0)/(txns.length||1)), color: 'green' },
                  ].map(({ label, value, color }) => (
                    <div key={label} className={clsx('card p-4 border-l-4',
                      { 'border-blue-400': color==='blue', 'border-indigo-400': color==='indigo',
                        'border-red-400': color==='red', 'border-green-400': color==='green' })}>
                      <p className="text-xs text-gray-500 mb-0.5">{label}</p>
                      <p className="text-base font-bold text-gray-900">{value}</p>
                    </div>
                  ))}
                </div>

                {highRiskTxns.length > 0 && (
                  <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-2.5 flex items-center gap-2 text-xs text-red-700">
                    <AlertTriangle size={13} className="flex-shrink-0" />
                    <span><strong>{highRiskTxns.length} flagged transaction{highRiskTxns.length > 1 ? 's' : ''}</strong> detected. Click any highlighted row to see the fraud explanation.</span>
                  </div>
                )}

                <div className="card overflow-hidden">
                  <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
                    <p className="text-sm font-semibold text-gray-900">Transaction History</p>
                    <span className="text-xs text-gray-400">{txns.length} records (real model scores)</span>
                  </div>
                  <div className="overflow-auto max-h-[500px]">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 z-10">
                        <tr className="bg-gray-50 border-b border-gray-100">
                          {['Time', 'Tx ID', 'Type', 'Merchant', 'Amount', 'Fraud Prob', 'Status'].map(h => (
                            <th key={h} className="px-3 py-2.5 text-left text-[10px] font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {txns.map(tx => <TxRow key={tx.tx_id} tx={tx} />)}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* ── RISK ANALYSIS ── */}
            {activeTab === 'risk' && (
              <div className="space-y-4 fade-up">
                <div className="grid grid-cols-2 gap-4">
                  {/* Credit Risk */}
                  <div className="card p-5">
                    <p className="text-sm font-semibold text-gray-900 mb-3">Credit Risk Assessment</p>
                    <div className={clsx('p-4 rounded-xl border mb-4', tier.bg, tier.border)}>
                      <div className="flex items-center gap-2 mb-1">
                        {accountData.credit_score >= 60
                          ? <CheckCircle size={16} className="text-green-600" />
                          : <AlertTriangle size={16} className="text-amber-600" />}
                        <p className={clsx('text-sm font-bold', tier.text)}>{accountData.credit_tier} CREDIT RISK</p>
                      </div>
                      <p className={clsx('text-xs leading-relaxed', tier.text)}>
                        {accountData.credit_score >= 80
                          ? 'Excellent creditworthiness. Strong candidate for premium loan products.'
                          : accountData.credit_score >= 60
                          ? 'Good creditworthiness. Eligible for standard loan products.'
                          : accountData.credit_score >= 40
                          ? 'Fair creditworthiness. Manual review recommended before loan approval.'
                          : 'High credit risk. Additional collateral or guarantor required.'}
                      </p>
                    </div>
                    {[
                      { label: 'Credit Score',       value: `${accountData.credit_score?.toFixed(1)}/100`, note: accountData.credit_tier },
                      { label: 'Fraud Risk Prob',    value: `${(accountData.fraud_risk*100).toFixed(1)}%`,  note: accountData.fraud_risk_level },
                      { label: 'Fraud History',      value: `${(accountData.fraud_history_ratio*100).toFixed(2)}%`, note: 'Historical rate' },
                      { label: 'Fraud Alerts',       value: accountData.fraud_alerts, note: 'Flagged transactions' },
                      { label: 'Avg Transaction',    value: fmtMoney(accountData.avg_amount), note: '30-day average' },
                    ].map(({ label, value, note }) => (
                      <div key={label} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                        <p className="text-xs text-gray-500">{label}</p>
                        <div className="text-right">
                          <p className="text-xs font-bold text-gray-900">{value}</p>
                          <p className="text-[10px] text-gray-400">{note}</p>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Fraud Risk */}
                  <div className="card p-5">
                    <p className="text-sm font-semibold text-gray-900 mb-3">Fraud Risk Profile</p>
                    <div className={clsx('p-4 rounded-xl border mb-4',
                      accountData.fraud_alerts > 0 ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200')}>
                      <div className="flex items-center gap-2 mb-1">
                        {accountData.fraud_alerts > 0
                          ? <AlertTriangle size={16} className="text-red-600" />
                          : <Shield size={16} className="text-green-600" />}
                        <p className={clsx('text-sm font-bold', accountData.fraud_alerts > 0 ? 'text-red-700' : 'text-green-700')}>
                          {accountData.fraud_alerts > 0 ? `${accountData.fraud_alerts} FRAUD ALERTS` : 'LOW FRAUD RISK'}
                        </p>
                      </div>
                      <p className={clsx('text-xs', accountData.fraud_alerts > 0 ? 'text-red-600' : 'text-green-600')}>
                        {accountData.fraud_alerts > 0
                          ? 'Flagged transactions detected. Review fraud explanation below.'
                          : 'Transaction patterns appear normal. No immediate action required.'}
                      </p>
                    </div>
                    {/* Distribution */}
                    {['CRITICAL','HIGH','MEDIUM','NORMAL'].map(status => {
                      const count = txns.filter(t => t.status === status).length
                      const pct   = txns.length > 0 ? (count / txns.length) * 100 : 0
                      const barColor = { CRITICAL: '#DC2626', HIGH: '#EF4444', MEDIUM: '#F59E0B', NORMAL: '#22C55E' }[status]
                      return (
                        <div key={status} className="mb-2.5">
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-gray-600 font-medium">{status}</span>
                            <span className="text-gray-500 tabular-nums">{count} tx · {pct.toFixed(1)}%</span>
                          </div>
                          <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${pct}%`, background: barColor }} />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Full explanations in Risk tab */}
                <CreditExplanation explanation={accountData.credit_explanation} />
                {accountData.fraud_risk > 0.15 && (
                  <FraudExplanation explanation={accountData.fraud_explanation} fraudProb={accountData.fraud_risk} />
                )}

                {/* Loan recommendation */}
                <div className="card p-5">
                  <p className="text-sm font-semibold text-gray-900 mb-3">Loan Eligibility</p>
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { label: 'Personal Loan', eligible: accountData.credit_score >= 60 && accountData.fraud_alerts === 0,
                        note: accountData.credit_score >= 60 ? 'Up to Rs. 500,000 · 7.5% p.a.' : 'Score below threshold' },
                      { label: 'Home Financing', eligible: accountData.credit_score >= 70,
                        note: accountData.credit_score >= 70 ? 'Up to 90% margin · 6.2% p.a.' : 'Minimum score 70 required' },
                      { label: 'Business Loan', eligible: accountData.credit_score >= 50 && accountData.fraud_risk < 0.5,
                        note: accountData.credit_score >= 50 ? 'Up to Rs. 5,000,000' : 'Higher score required' },
                    ].map(({ label, eligible, note }) => (
                      <div key={label} className={clsx('p-4 rounded-xl border', eligible ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200')}>
                        <div className="flex items-center gap-2 mb-2">
                          {eligible ? <CheckCircle size={14} className="text-green-600" /> : <AlertTriangle size={14} className="text-gray-400" />}
                          <p className={clsx('text-xs font-bold', eligible ? 'text-green-700' : 'text-gray-400')}>{label}</p>
                        </div>
                        <p className={clsx('text-[10px]', eligible ? 'text-green-600' : 'text-gray-400')}>{note}</p>
                        <span className={clsx('inline-block mt-2 text-[10px] font-bold px-2 py-0.5 rounded-full',
                          eligible ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-500')}>
                          {eligible ? 'ELIGIBLE' : 'NOT ELIGIBLE'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

          </div>
        ) : (
          <div className="flex items-center justify-center h-64">
            <p className="text-sm text-red-400">Failed to load account data. Please try again.</p>
          </div>
        )}
      </main>
    </div>
  )
}
