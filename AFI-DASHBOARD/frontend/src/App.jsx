import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import CreditScoring from './pages/CreditScoring'
import FraudDetection from './pages/FraudDetection'
import ModelPerformance from './pages/ModelPerformance'
import TransactionAnalysis from './pages/TransactionAnalysis'
import AboutAFI from './pages/AboutAFI'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard"    element={<Dashboard />} />
          <Route path="credit"       element={<CreditScoring />} />
          <Route path="fraud"        element={<FraudDetection />} />
          <Route path="transactions" element={<TransactionAnalysis />} />
          <Route path="performance"  element={<ModelPerformance />} />
          <Route path="about"        element={<AboutAFI />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}