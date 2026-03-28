/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary:   { DEFAULT: '#1E3A8A', light: '#2563EB', pale: '#60A5FA' },
        surface:   { DEFAULT: '#FFFFFF', bg: '#F8FAFC', card: '#FFFFFF' },
        risk: {
          low:    '#22C55E',
          medium: '#F59E0B',
          high:   '#EF4444',
        },
        dark: {
          bg:   '#0F172A',
          card: '#1E293B',
          text: '#E2E8F0',
        }
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      boxShadow: {
        card:  '0 2px 12px 0 rgba(30,58,138,0.08)',
        hover: '0 8px 32px 0 rgba(30,58,138,0.14)',
        glow:  '0 0 24px 0 rgba(96,165,250,0.25)',
      },
      borderRadius: {
        card: '12px',
      },
    },
  },
  plugins: [],
}