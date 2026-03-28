// TransactionAnalysis.jsx
import { BarChart, Bar, LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell } from 'recharts'
import { TrendingUp, BarChart3 } from 'lucide-react'
import { useApi } from '../hooks/useApi'

export function TransactionAnalysis() {
  const { data: history30 } = useApi('/transactions/history', { params: { days: 30 } })
  const { data: history14 } = useApi('/transactions/history', { params: { days: 14 } })
  const { data: perf }      = useApi('/model/performance')
  const { data: dist }      = useApi('/credit/distribution')
  const { data: hourly }    = useApi('/transactions/hourly')

  const hist30      = history30?.history || []
  const hist14      = history14?.history || []
  const creditDist  = perf?.credit_distribution || []
  const byRisk      = dist?.by_risk || []
  const hourlyData  = hourly?.hourly || []

  return (
    <div className="p-6 space-y-6 fade-up">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Transaction Analysis</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Volume trends, fraud patterns, and credit score distribution — 5M dataset · Fraud rate: 3.591%
        </p>
      </div>

      {/* Row 1 */}
      <div className="grid grid-cols-3 gap-5 fade-up-1">
        <div className="card p-5 col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-semibold text-gray-900">30-Day Transaction Volume</p>
              <p className="text-xs text-gray-400">Daily totals — ~13,700 tx/day · 3.591% fraud rate</p>
            </div>
            <TrendingUp size={16} className="text-blue-400" />
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={hist30} margin={{ left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#9CA3AF' }} tickLine={false} axisLine={false} interval={4} />
              <YAxis tick={{ fontSize: 10, fill: '#9CA3AF' }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ borderRadius: 10, border: '1px solid #E5E7EB', fontSize: 11 }} />
              <Bar dataKey="normal" name="Normal" stackId="a" fill="#2563EB" fillOpacity={0.7} radius={[0,0,0,0]} />
              <Bar dataKey="fraud"  name="Fraud"  stackId="a" fill="#EF4444" fillOpacity={0.85} radius={[3,3,0,0]} />
              <Legend formatter={v => <span style={{ fontSize: 11, color: '#6B7280' }}>{v}</span>} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-5">
          <p className="text-sm font-semibold text-gray-900 mb-1">Credit Score Distribution</p>
          <p className="text-xs text-gray-400 mb-4">1M test set — by score range</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={creditDist} layout="vertical" margin={{ left: 0, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10, fill: '#9CA3AF' }} tickLine={false} axisLine={false} />
              <YAxis dataKey="range" type="category" tick={{ fontSize: 11, fill: '#6B7280' }} tickLine={false} axisLine={false} width={45} />
              <Tooltip contentStyle={{ borderRadius: 10, fontSize: 11 }} />
              <Bar dataKey="count" radius={[0,4,4,0]} name="Transactions">
                {creditDist.map((_, i) => (
                  <Cell key={i} fill={['#EF4444','#F59E0B','#60A5FA','#2563EB','#22C55E'][i] || '#2563EB'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Row 2 */}
      <div className="grid grid-cols-2 gap-5 fade-up-2">
        <div className="card p-5">
          <p className="text-sm font-semibold text-gray-900 mb-1">Hourly Transaction Pattern</p>
          <p className="text-xs text-gray-400 mb-4">Transactions & fraud by hour — peak: 9am–6pm · Higher fraud: 12am–5am</p>
          <ResponsiveContainer width="100%" height={190}>
            <AreaChart data={hourlyData} margin={{ left: -20 }}>
              <defs>
                <linearGradient id="gTx" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2563EB" stopOpacity={0.12} />
                  <stop offset="95%" stopColor="#2563EB" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis dataKey="hour" tick={{ fontSize: 9, fill: '#9CA3AF' }} tickLine={false} axisLine={false} interval={3} />
              <YAxis tick={{ fontSize: 10, fill: '#9CA3AF' }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ borderRadius: 10, fontSize: 11 }} />
              <Area type="monotone" dataKey="transactions" stroke="#2563EB" strokeWidth={2} fill="url(#gTx)" name="Transactions" />
              <Line type="monotone" dataKey="fraud" stroke="#EF4444" strokeWidth={2} dot={false} name="Fraud" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-5">
          <p className="text-sm font-semibold text-gray-900 mb-1">Credit Risk Breakdown</p>
          <p className="text-xs text-gray-400 mb-4">1M test set — users by risk category</p>
          <div className="space-y-4 pt-2">
            {byRisk.map((r) => {
              const total = byRisk.reduce((a, x) => a + x.count, 0)
              const pct   = ((r.count / total) * 100).toFixed(1)
              return (
                <div key={r.level}>
                  <div className="flex justify-between text-xs mb-1.5">
                    <div className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full" style={{ background: r.color }} />
                      <span className="font-semibold text-gray-700">{r.level}</span>
                    </div>
                    <div className="flex gap-3 text-gray-500">
                      <span className="font-mono">{r.count.toLocaleString()}</span>
                      <span className="font-semibold" style={{ color: r.color }}>{pct}%</span>
                    </div>
                  </div>
                  <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${pct}%`, background: r.color }} />
                  </div>
                </div>
              )
            })}
          </div>
          <div className="mt-5 pt-4 border-t border-gray-100 grid grid-cols-2 gap-3">
            {[
              { label: 'Total Tested',   value: byRisk.reduce((a,x)=>a+x.count,0).toLocaleString() },
              { label: 'Fraud in Dataset', value: '179,553' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-gray-50 rounded-xl p-3">
                <p className="text-xs text-gray-400 mb-0.5">{label}</p>
                <p className="text-lg font-bold text-gray-900">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Fraud trend */}
      <div className="card p-5 fade-up-3">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-sm font-semibold text-gray-900">Fraud Detection Trend (14 Days)</p>
            <p className="text-xs text-gray-400">Daily fraud cases detected · avg ~490/day (3.591% of ~13,700)</p>
          </div>
          <BarChart3 size={16} className="text-blue-400" />
        </div>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={hist14} margin={{ left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9CA3AF' }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10, fill: '#9CA3AF' }} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={{ borderRadius: 10, fontSize: 11 }} />
            <Line type="monotone" dataKey="fraud" stroke="#EF4444" strokeWidth={2.5} dot={{ fill: '#EF4444', r: 3 }} name="Fraud Cases" />
            <Line type="monotone" dataKey="total" stroke="#2563EB" strokeWidth={1.5} strokeDasharray="4 2" dot={false} name="Total Tx" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default TransactionAnalysis