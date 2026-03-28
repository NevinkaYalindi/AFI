import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const BASE = (import.meta.env.VITE_API_URL || '') + '/api'

export function useApi(endpoint, options = {}) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const { poll = false, interval = 5000, params = {} } = options

  const fetch = useCallback(async () => {
    try {
      const res = await axios.get(`${BASE}${endpoint}`, { params })
      setData(res.data)
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [endpoint, JSON.stringify(params)])

  useEffect(() => {
    fetch()
    if (poll) {
      const id = setInterval(fetch, interval)
      return () => clearInterval(id)
    }
  }, [fetch, poll, interval])

  return { data, loading, error, refetch: fetch }
}

export async function analyzeTransaction(payload) {
  const res = await axios.post(`${BASE}/afi/analyze`, payload)
  return res.data
}

export async function scoreCredit(payload) {
  const res = await axios.post(`${BASE}/credit/score`, payload)
  return res.data
}

export async function detectFraud(payload) {
  const res = await axios.post(`${BASE}/fraud/detect`, payload)
  return res.data
}