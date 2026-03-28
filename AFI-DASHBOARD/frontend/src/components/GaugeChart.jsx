import { useMemo } from 'react'

const polarToCartesian = (cx, cy, r, angle) => {
  const rad = (angle - 90) * (Math.PI / 180)
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

const arcPath = (cx, cy, r, start, end) => {
  const s = polarToCartesian(cx, cy, r, start)
  const e = polarToCartesian(cx, cy, r, end)
  const large = end - start <= 180 ? 0 : 1
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`
}

export default function GaugeChart({ score = 0, size = 240 }) {
  const cx = size / 2
  const cy = size / 2
  const r = size * 0.38
  const strokeW = size * 0.075

  // Map score (0–100) to angle (-135 to 135)
  const startAngle = -135
  const endAngle = 135
  const totalSpan = endAngle - startAngle
  const valueAngle = startAngle + (score / 100) * totalSpan

  const color = score >= 80 ? '#22C55E'
              : score >= 60 ? '#60A5FA'
              : score >= 40 ? '#F59E0B'
              :               '#EF4444'

  const level = score >= 80 ? 'Excellent'
              : score >= 60 ? 'Good'
              : score >= 40 ? 'Fair'
              :               'Poor'

  const segments = [
    { from: -135, to: -63,  color: '#EF4444' },   // Poor 0–25
    { from: -63,  to:  9,   color: '#F59E0B' },   // Fair 25–50
    { from:  9,   to:  81,  color: '#60A5FA' },   // Good 50–75
    { from:  81,  to:  135, color: '#22C55E' },   // Excellent 75–100
  ]

  // Needle
  const needle = polarToCartesian(cx, cy, r * 0.8, valueAngle)

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size * 0.72} viewBox={`0 0 ${size} ${size * 0.72}`}>
        {/* Track */}
        <path
          d={arcPath(cx, cy, r, -135, 135)}
          fill="none"
          stroke="#E5E7EB"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />

        {/* Colored segments */}
        {segments.map((seg, i) => (
          <path
            key={i}
            d={arcPath(cx, cy, r, seg.from, seg.to)}
            fill="none"
            stroke={seg.color}
            strokeWidth={strokeW}
            strokeOpacity={0.2}
          />
        ))}

        {/* Value arc */}
        <path
          d={arcPath(cx, cy, r, -135, Math.max(-135, valueAngle))}
          fill="none"
          stroke={color}
          strokeWidth={strokeW}
          strokeLinecap="round"
          style={{ transition: 'all 0.8s cubic-bezier(0.34,1.56,0.64,1)' }}
        />

        {/* Needle */}
        <line
          x1={cx} y1={cy}
          x2={needle.x} y2={needle.y}
          stroke={color}
          strokeWidth={2.5}
          strokeLinecap="round"
          style={{ transition: 'all 0.8s cubic-bezier(0.34,1.56,0.64,1)' }}
        />
        <circle cx={cx} cy={cy} r={size * 0.03} fill={color} />

        {/* Score text */}
        <text x={cx} y={cy * 0.82} textAnchor="middle" fontSize={size * 0.18} fontWeight="800" fill="#1F2937" fontFamily="'Plus Jakarta Sans', sans-serif">
          {Math.round(score)}
        </text>
        <text x={cx} y={cy * 0.82 + size * 0.10} textAnchor="middle" fontSize={size * 0.065} fill="#6B7280" fontFamily="'Plus Jakarta Sans', sans-serif" fontWeight="500">
          / 100
        </text>

        {/* Labels */}
        <text x={size * 0.09} y={size * 0.67} textAnchor="middle" fontSize={size * 0.052} fill="#EF4444" fontWeight="600" fontFamily="'Plus Jakarta Sans', sans-serif">Poor</text>
        <text x={size * 0.91} y={size * 0.67} textAnchor="middle" fontSize={size * 0.052} fill="#22C55E" fontWeight="600" fontFamily="'Plus Jakarta Sans', sans-serif">Excellent</text>
      </svg>

      <div className="flex flex-col items-center -mt-2">
        <span
          className="text-sm font-bold px-4 py-1.5 rounded-full"
          style={{ backgroundColor: `${color}18`, color }}
        >
          {level}
        </span>
      </div>
    </div>
  )
}