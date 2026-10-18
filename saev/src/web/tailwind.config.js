/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.elm",
    "./apps/**/*.html"
  ],
  theme: {
    extend: {},
    screens: {
      "xs": "384px",
      "sm": "512px",
      "md": "768px",
      "lg": "896px",
      "xl": "1440px",
    },
  },
  plugins: [],
}

