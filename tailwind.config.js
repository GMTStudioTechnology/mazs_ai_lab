/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class', // Enable dark mode
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './public/index.html',
  ],
  theme: {
    extend: {
      colors: {
        brown: '#8B4513',
        silver: '#C0C0C0',
        golden: '#FFD700',
        black: '#000000',
        white: '#FFFFFF',
        goldenHour:{
          light: '#FFD700',
          DEFAULT: '#FFb347',
          dark: '#FF8C00',
        },
        softpink: '#FFB6C1',
        softAmber: '#FFC87C',
        oceanBreeze: {
          light: '#A7DFF5',   // Light blue, like the sky reflecting on water
          DEFAULT: '#5BC3EB', // Main ocean breeze blue
          dark: '#1C9DC9',    // Deep ocean blue
        },
        seaFoam: '#B2E6D4',   // Soft, pale green reminiscent of seafoam
        sandyBeach: '#F2D1A8',
      },
      boxShadow: {
        glow: '0 0 10px rgba(0, 0, 0, 0.5)',
        'glow-blue': '0 0 10px #3B82F6',
      },
    },
  },
  plugins: [],
};