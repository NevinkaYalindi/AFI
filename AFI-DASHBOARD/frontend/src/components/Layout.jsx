import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, CreditCard, ShieldAlert, BarChart3,
  Activity, Info, Zap, Circle
} from 'lucide-react'
import { useApi } from '../hooks/useApi'
import clsx from 'clsx'

const NAV = [
  { to: '/dashboard',    icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/credit',       icon: CreditCard,      label: 'Credit Scoring' },
  { to: '/fraud',        icon: ShieldAlert,     label: 'Fraud Detection' },
  { to: '/transactions', icon: Activity,        label: 'Transaction Analysis' },
  { to: '/performance',  icon: BarChart3,       label: 'Model Performance' },
  { to: '/about',        icon: Info,            label: 'About AFI' },
]

export default function Layout() {
  const { pathname } = useLocation()
  const navigate     = useNavigate()
  const { data: health } = useApi('/health', { poll: true, interval: 8000 })

  const isOnline   = health?.status === 'online'
  const mode       = health?.mode   ?? 'loading'
  const version    = health?.models?.version ?? '—'
  const txCount    = health?.transactions ?? 0
  const isReady    = txCount > 0

  // Colour-code mode badge
  const modeBadge =
    mode === 'production'  ? 'bg-blue-50 text-blue-700' :
    mode === 'heuristic'   ? 'bg-amber-50 text-amber-700' :
    mode === 'loading'     ? 'bg-gray-100 text-gray-400 animate-pulse' :
                             'bg-amber-50 text-amber-600'

  return (
    <div className="flex h-screen overflow-hidden bg-[#F8FAFC]">

      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside className="w-64 flex-shrink-0 flex flex-col border-r border-gray-200 bg-white">

        {/* Logo */}
        <div className="px-5 py-4 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[#1E3A8A] flex items-center justify-center shadow-md flex-shrink-0">
              <Zap size={18} className="text-[#60A5FA]" strokeWidth={2.5} />
            </div>
            <div>
              <p className="leading-tight">
                <span className="text-[15px] font-extrabold text-[#1E3A8A] tracking-tight">AFI Credit</span>
                <span className="text-[15px] font-extrabold text-[#2563EB]">.</span>
              </p>
              <p className="text-[10px] text-gray-400 font-medium leading-tight">Adaptive Financial Intelligence</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-0.5">
          {NAV.map(({ to, icon: Icon, label }) => (
            <button
              key={to}
              onClick={() => navigate(to)}
              className={clsx(
                'nav-item w-full text-left',
                pathname.startsWith(to) && 'nav-active'
              )}
            >
              <Icon size={16} strokeWidth={2} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        {/* Status footer */}
        <div className="px-4 py-4 border-t border-gray-100">
          <div className="bg-gray-50 rounded-xl p-3 space-y-2">

            {/* System / Online */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500 font-medium">System</span>
              <div className="flex items-center gap-1.5">
                <Circle size={7} className={clsx('fill-current', isOnline ? 'text-green-500 pulse-dot' : 'text-red-400')} />
                <span className={clsx('text-xs font-semibold', isOnline ? 'text-green-600' : 'text-red-500')}>
                  {isOnline ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>

            {/* Data loaded */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500 font-medium">Data</span>
              <span className={clsx('text-xs font-semibold',
                isReady ? 'text-green-600' : 'text-amber-500')}>
                {isReady ? `${txCount.toLocaleString()} tx` : 'Loading…'}
              </span>
            </div>

            {/* Mode */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500 font-medium">Mode</span>
              <span className={clsx('text-xs font-semibold px-2 py-0.5 rounded-full capitalize', modeBadge)}>
                {mode}
              </span>
            </div>

            {/* Model version — full text, wraps */}
            <div className="pt-1 border-t border-gray-200">
              <p className="text-[10px] text-gray-400 font-medium mb-0.5">Model</p>
              <p className="text-[11px] font-semibold text-gray-700 leading-snug break-words">
                {version}
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main Content ────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto grid-bg flex flex-col">
        <div className="flex-1">
          <Outlet />
        </div>

        {/* Disclaimer */}
        <footer className="px-6 py-3 border-t border-gray-200 bg-white/70 backdrop-blur-sm flex-shrink-0">
          <div className="flex items-start gap-2">
            <span className="text-amber-500 flex-shrink-0 mt-0.5 text-xs">⚠</span>
            <p className="text-[10px] text-gray-400 leading-relaxed">
              <span className="font-semibold text-gray-500">AI Disclaimer:</span>{' '}
              AFI Credit. is an AI-powered system. Credit scores, fraud probabilities, and risk
              classifications are generated by machine learning models and should not be used as the
              sole basis for financial decisions. Model outputs are probabilistic and subject to error.
            </p>
          </div>
        </footer>
      </main>
    </div>
  )
}