import clsx from 'clsx'

const CONFIGS = {
  CRITICAL: { cls: 'bg-red-100 text-red-800 border-red-300',   dot: 'bg-red-600' },
  HIGH:     { cls: 'bg-red-50 text-red-700 border-red-200',    dot: 'bg-red-500' },
  MEDIUM:   { cls: 'bg-amber-50 text-amber-700 border-amber-200', dot: 'bg-amber-500' },
  LOW:      { cls: 'bg-green-50 text-green-700 border-green-200', dot: 'bg-green-500' },
  NORMAL:   { cls: 'bg-green-50 text-green-700 border-green-200', dot: 'bg-green-500' },
  EXCELLENT:{ cls: 'bg-blue-50 text-blue-700 border-blue-200',  dot: 'bg-blue-500' },
  GOOD:     { cls: 'bg-indigo-50 text-indigo-700 border-indigo-200', dot: 'bg-indigo-400' },
  FAIR:     { cls: 'bg-amber-50 text-amber-700 border-amber-200', dot: 'bg-amber-500' },
  POOR:     { cls: 'bg-red-50 text-red-700 border-red-200',    dot: 'bg-red-500' },
  APPROVE:  { cls: 'bg-green-50 text-green-700 border-green-200', dot: 'bg-green-500' },
  REJECT:   { cls: 'bg-red-50 text-red-700 border-red-200',    dot: 'bg-red-500' },
  REVIEW:   { cls: 'bg-amber-50 text-amber-700 border-amber-200', dot: 'bg-amber-500' },
}

export default function RiskBadge({ level, className }) {
  const key = (level || '').toUpperCase().split(' ')[0].split('—')[0].trim()
  const cfg = CONFIGS[key] || CONFIGS.MEDIUM
  return (
    <span className={clsx(
      'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border',
      cfg.cls, className
    )}>
      <span className={clsx('w-1.5 h-1.5 rounded-full', cfg.dot)} />
      {level}
    </span>
  )
}