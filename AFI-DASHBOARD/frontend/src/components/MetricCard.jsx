import clsx from 'clsx'

export default function MetricCard({ icon: Icon, label, value, sub, color = 'blue', trend, className }) {
  const colors = {
    blue:   { bg: 'bg-blue-50',   icon: 'text-blue-600',   border: 'border-blue-100' },
    green:  { bg: 'bg-green-50',  icon: 'text-green-600',  border: 'border-green-100' },
    amber:  { bg: 'bg-amber-50',  icon: 'text-amber-600',  border: 'border-amber-100' },
    red:    { bg: 'bg-red-50',    icon: 'text-red-600',    border: 'border-red-100' },
    indigo: { bg: 'bg-indigo-50', icon: 'text-indigo-600', border: 'border-indigo-100' },
  }
  const c = colors[color] || colors.blue

  return (
    <div className={clsx('card p-5 flex flex-col gap-3 hover:shadow-hover transition-all duration-200', className)}>
      <div className="flex items-start justify-between">
        <div className={clsx('w-10 h-10 rounded-xl flex items-center justify-center', c.bg, `border ${c.border}`)}>
          {Icon && <Icon size={18} className={c.icon} strokeWidth={2} />}
        </div>
        {trend !== undefined && (
          <span className={clsx('text-xs font-semibold px-2 py-1 rounded-lg',
            trend >= 0 ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
          )}>
            {trend >= 0 ? '▲' : '▼'} {Math.abs(trend)}%
          </span>
        )}
      </div>
      <div>
        <p className="text-2xl font-bold text-gray-900 leading-none mb-1">{value ?? '—'}</p>
        <p className="text-sm font-medium text-gray-500">{label}</p>
        {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}