// AboutAFI.jsx
import { Zap, Brain, Shield, TrendingUp, Users, Database, Cpu, ArrowRight } from 'lucide-react'

const PIPELINE = [
  { step: '1', title: 'Raw Transaction Data',  desc: 'Mobile payments, transfers, withdrawals, deposits — 18 raw fields from the dataset', icon: Database,   color: 'blue' },
  { step: '2', title: 'Feature Engineering',   desc: '29 behavioural & temporal features — velocity, geo-anomaly, sender history, receiver risk', icon: Cpu,   color: 'indigo' },
  { step: '3', title: 'SMOTE Balancing',       desc: 'Class imbalance handled — 143K fraud → 3.8M balanced training samples post-SMOTE', icon: Brain,     color: 'purple' },
  { step: '4', title: 'Credit Scoring',        desc: 'LightGBM scores creditworthiness (0–100) trained on 7.7M SMOTE-balanced samples', icon: TrendingUp, color: 'green' },
  { step: '5', title: 'Fraud Detection',       desc: 'LightGBM flags fraudulent transactions — 100% recall on 1M test transactions', icon: Shield,    color: 'red' },
]

const INNOVATIONS = [
  {
    title: 'Alternative Data Credit Scoring',
    desc: 'Traditional credit scoring excludes unbanked populations. AFI uses mobile transaction history, spending consistency, velocity patterns, and geo-anomaly scores to generate credit scores — enabling financial inclusion for millions of Sri Lankans.',
    icon: Users,
    highlight: 'FR03 validated — 100% precision & recall',
  },
  {
    title: '5M Dataset',
    desc: 'Both models were trained on the 2-million transaction Financial Fraud Detection Dataset using Kaggle GPU T4 x2 infrastructure. Optuna Bayesian optimisation with 60 trials found globally optimal hyperparameters across both models.',
    icon: Database,
    highlight: '5M transactions · 60-trial Optuna',
  },
  {
    title: 'Sub-millisecond Latency',
    desc: 'The integrated AFI system processes each transaction through both models in under 0.65ms on average — exceeding the NFR3 requirement of 1000ms by 1500×. Throughput: 60,000 transactions/second.',
    icon: Zap,
    highlight: 'NFR3 compliant — 0.6ms avg latency',
  },
  {
    title: 'Unified Integrated System',
    desc: 'Both LightGBM models run simultaneously on the same 29-feature vector. A single pipeline processes any transaction through credit scoring and fraud detection together — returning a consolidated APPROVE / REVIEW / REJECT decision.',
    icon: Brain,
    highlight: '29-feature unified pipeline',
  },
]

const TECH_STACK = [
  { name: 'LightGBM',       role: 'Credit scoring + fraud detection models', color: 'bg-blue-50 text-blue-700 border-blue-200' },
  { name: 'Optuna',         role: 'Bayesian hyperparameter optimisation',    color: 'bg-violet-50 text-violet-700 border-violet-200' },
  { name: 'SMOTE',          role: 'Class imbalance handling',                color: 'bg-purple-50 text-purple-700 border-purple-200' },
  { name: 'scikit-learn',   role: 'Feature scaling & evaluation',            color: 'bg-green-50 text-green-700 border-green-200' },
  { name: 'pandas / numpy', role: 'Feature engineering',                     color: 'bg-teal-50 text-teal-700 border-teal-200' },
  { name: 'FastAPI',        role: 'REST API backend',                        color: 'bg-cyan-50 text-cyan-700 border-cyan-200' },
  { name: 'Kaggle GPU T4',  role: 'Model training infrastructure',           color: 'bg-orange-50 text-orange-700 border-orange-200' },
  { name: 'React + Vite',   role: 'Dashboard frontend',                      color: 'bg-sky-50 text-sky-700 border-sky-200' },
  { name: 'Recharts',       role: 'Data visualization',                      color: 'bg-amber-50 text-amber-700 border-amber-200' },
]

const colorMap = {
  blue:   { bg: 'bg-blue-50',   icon: 'text-blue-600',   border: 'border-blue-100' },
  indigo: { bg: 'bg-indigo-50', icon: 'text-indigo-600', border: 'border-indigo-100' },
  purple: { bg: 'bg-purple-50', icon: 'text-purple-600', border: 'border-purple-100' },
  green:  { bg: 'bg-green-50',  icon: 'text-green-600',  border: 'border-green-100' },
  red:    { bg: 'bg-red-50',    icon: 'text-red-600',    border: 'border-red-100' },
}

// 29 features
const FEATURES_29 = [
  'amount', 'transaction_type', 'merchant_category', 'device_used',
  'fraud_type', 'time_since_last_transaction', 'spending_deviation_score',
  'velocity_score', 'geo_anomaly_score', 'payment_channel',
  'hour', 'day_of_week', 'is_weekend', 'is_night', 'month',
  'log_amount', 'is_round_amount', 'avg_amount', 'std_amount',
  'max_amount', 'total_amount', 'fraud_history_ratio',
  'spending_consistency', 'activity_score', 'amount_deviation',
  'is_large_tx', 'recv_tx_count', 'recv_fraud_cnt', 'receiver_risk_score',
]

export default function AboutAFI() {
  return (
    <div className="p-6 space-y-8 fade-up max-w-5xl">

      {/* Hero */}
      <div className="card p-8 bg-gradient-to-br from-[#1E3A8A] to-[#2563EB] text-white border-0">
        <div className="flex items-start gap-6">
          <div className="w-14 h-14 bg-white/10 rounded-2xl flex items-center justify-center flex-shrink-0 backdrop-blur-sm">
            <Zap size={28} className="text-[#60A5FA]" />
          </div>
          <div>
            <h1 className="text-3xl font-extrabold mb-2">AFI Credit. — Adaptive Financial Intelligence</h1>
            <p className="text-blue-100 leading-relaxed text-sm max-w-2xl">
              A machine learning system integrating real-time fraud detection and alternative data credit scoring to enable financial inclusion and protect digital payment 
              ecosystems in Sri Lanka.
              Trained on 5 million real transactions using Kaggle GPU infrastructure.
              Built as part of an Integrated Project Design (IPD) thesis — N.B. Nevinka.
            </p>
            <div className="flex gap-3 mt-4 flex-wrap">
              {['Credit Scoring', 'Fraud Detection', '5M Dataset', 'LightGBM + Optuna', 'Sri Lanka'].map(tag => (
                <span key={tag} className="text-xs font-semibold px-3 py-1 bg-white/15 rounded-full backdrop-blur-sm">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Key Stats */}
      <div className="grid grid-cols-4 gap-4 fade-up-0">
        {[
          { value: '5,000,000', label: 'Training transactions', sub: 'Full dataset, Kaggle' },
          { value: '100%',      label: 'Recall & Precision',    sub: 'Both models, test set' },
          { value: '29',        label: 'Engineered features',   sub: 'From 18 raw fields' },
          { value: '0.6 ms',    label: 'Avg latency',           sub: 'NFR3: < 1000 ms' },
        ].map(({ value, label, sub }) => (
          <div key={label} className="card p-4 text-center">
            <p className="text-2xl font-extrabold text-[#1E3A8A]">{value}</p>
            <p className="text-xs font-semibold text-gray-700 mt-1">{label}</p>
            <p className="text-[10px] text-gray-400">{sub}</p>
          </div>
        ))}
      </div>

      {/* Pipeline */}
      <div className="fade-up-1">
        <h2 className="text-lg font-bold text-gray-900 mb-4">System Pipeline</h2>
        <div className="card p-6">
          <div className="flex items-start gap-0 overflow-x-auto">
            {PIPELINE.map((step, i) => {
              const Icon = step.icon
              const c    = colorMap[step.color]
              return (
                <div key={step.step} className="flex items-start flex-shrink-0">
                  <div className="flex flex-col items-center w-40 text-center">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center border ${c.bg} ${c.border} mb-3`}>
                      <Icon size={20} className={c.icon} />
                    </div>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full mb-2 ${c.bg} ${c.icon}`}>Step {step.step}</span>
                    <p className="text-xs font-bold text-gray-900 mb-1">{step.title}</p>
                    <p className="text-[10px] text-gray-400 leading-relaxed">{step.desc}</p>
                  </div>
                  {i < PIPELINE.length - 1 && (
                    <div className="flex items-center pt-5 mx-1 flex-shrink-0">
                      <ArrowRight size={14} className="text-gray-300" />
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Key Innovations */}
      <div className="fade-up-2">
        <h2 className="text-lg font-bold text-gray-900 mb-4">Key Innovations</h2>
        <div className="grid grid-cols-2 gap-4">
          {INNOVATIONS.map(({ title, desc, icon: Icon, highlight }) => (
            <div key={title} className="card p-5 hover:shadow-hover transition-all">
              <div className="flex items-start gap-3 mb-3">
                <div className="w-9 h-9 bg-blue-50 rounded-xl flex items-center justify-center flex-shrink-0 border border-blue-100">
                  <Icon size={16} className="text-blue-600" />
                </div>
                <div>
                  <p className="text-sm font-bold text-gray-900">{title}</p>
                  <span className="text-[10px] font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-100">
                    {highlight}
                  </span>
                </div>
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 29 Engineered Features */}
      <div className="fade-up-3">
        <h2 className="text-lg font-bold text-gray-900 mb-4">29 Engineered Features (feature_cols.json)</h2>
        <div className="card p-5">
          <div className="grid grid-cols-3 gap-x-6 gap-y-1.5">
            {FEATURES_29.map(f => (
              <div key={f} className="flex items-center gap-1.5 text-xs text-gray-600 font-mono py-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 flex-shrink-0" />
                {f}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="fade-up-4">
        <h2 className="text-lg font-bold text-gray-900 mb-4">Technology Stack</h2>
        <div className="flex flex-wrap gap-2">
          {TECH_STACK.map(({ name, role, color }) => (
            <div key={name} className={`px-3 py-2 rounded-xl border text-xs font-semibold ${color}`}>
              {name}
              <span className="ml-1.5 font-normal opacity-70">— {role}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset Info */}
      <div className="card p-5 border-2 border-dashed border-blue-200 bg-blue-50/40 fade-up-4">
        <div className="flex items-start gap-3">
          <Database size={18} className="text-blue-500 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm font-bold text-blue-800 mb-1">Dataset — Financial Fraud Detection Dataset (Kumar, 2025)</p>
            <p className="text-xs text-blue-600 leading-relaxed">
              5,000,000 anonymized financial transactions sourced from Kaggle. Contains 18 raw features including
              transaction type, amount, sender/receiver accounts, device, location, and pre-computed risk scores
              (spending_deviation_score, velocity_score, geo_anomaly_score). Fraud rate: 3.591% (179,553 fraud cases).
              Feature engineering expands this to 29 model-ready features. Models trained on Kaggle GPU T4 x2 infrastructure.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}