# AI News Dashboard

A modern, modular, and interactive AI-powered news dashboard built with Next.js, Shadcn UI, and Tailwind CSS. Aggregates global news, provides smart summarization (OpenAI o4-mini-high), and features a floating News Explorer chat for deep-dive topic exploration.

---

## Features

- **Global News Aggregation:** Fetches news from Google News API (Custom Search) using your API key.
- **Smart Summarization:** Summarizes articles and YouTube videos using the o4-mini-high model (OpenAI API).
- **Search & Filter:** Dedicated search component (`SearchInput.js`) to filter news by topic/keyword.
- **YouTube News Bites:** Paste a YouTube URL to get a news-style summary of the video.
- **News Explorer Chat:** Floating popover chat (bottom-right) for deep-dive exploration of any topic, powered by live news and AI.
- **Sidebar Navigation:** Modern sidebar for navigation, inspired by top open-source AI dashboards.
- **Responsive & Accessible:** Mobile-friendly, keyboard accessible, dark/light mode support.

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ai-news-dashboard
   ```
2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```
   *Note: This project uses Tailwind CSS v4, which requires `@tailwindcss/postcss`. It also uses `@radix-ui/react-label` for the Label component. These should be installed automatically.*
3. **Configure environment variables:**
   - Copy `.env.local.example` to `.env.local` and fill in your keys:
     ```env
     NEXT_PUBLIC_NEWS_API_KEY=your_google_api_key
     O4_MODEL_API_KEY=your_openai_api_key
     O4_MODEL_API_URL= # (leave blank to use default)
     ```
4. **Run the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```
5. **Open the app:**
   - Visit [http://localhost:3000](http://localhost:3000) in your browser.

---

## Project Structure

- `app/`
  - `page.js` — Main layout with sidebar, dashboard, and floating chat
  - `NewsDashboard.js` — Main news grid, search, and summarization UI
  - `api/` — API routes for news, summarization, and chat
- `components/`
  - `NewsExplorer.js` — Floating chat popover for topic deep-dives
  - `NewsCard.js` - Displays individual news articles
  - `TagChip.js` - Clickable tag/category filter chip
  - `ui/` — Modular Shadcn UI components (sidebar, popover, input, button, label, SearchInput, etc.)
- `lib/`
  - `newsFetcher.js` — Fetches news from Google News API
  - `o4ModelClient.js` — Handles OpenAI summarization
  - `youtubeTranscript.js` — Fetches YouTube transcripts

---

## UI/UX Principles

- **Inspired by:** [Horizon AI Boilerplate](https://github.com/kameleonbe/shadcn), [Leniolabs/ai-data-dashboard](https://github.com/Leniolabs/ai-data-dashboard)
- **Sidebar navigation** for clarity and modularity
- **Floating popover chat** for non-blocking, contextual exploration
- **Accessible and responsive** (keyboard, ARIA, mobile)
- **Clean, well-commented code** for maintainability

---

## Deployment

- Deploy on Vercel or any Next.js-compatible platform.
- See [Next.js deployment docs](https://nextjs.org/docs/app/building-your-application/deploying) for details.

---

## Credits
- Built with [Next.js](https://nextjs.org/), [Shadcn UI](https://ui.shadcn.com/), [Tailwind CSS](https://tailwindcss.com/)
- News by Google News API, Summarization by OpenAI
- UI/UX inspired by top open-source AI dashboards

---

## Troubleshooting

- **Build Error: `It looks like you're trying to use tailwindcss directly as a PostCSS plugin`**: 
  - This project uses Tailwind CSS v4. The PostCSS plugin has moved to `@tailwindcss/postcss`.
  - **Fix:** Ensure `@tailwindcss/postcss` is installed (`npm install -D @tailwindcss/postcss`) and update `postcss.config.js`:
    ```javascript
    module.exports = {
      plugins: {
        '@tailwindcss/postcss': {},
        autoprefixer: {},
      },
    };
    ```
- **Build Error: `Module not found: Can't resolve './label'`**: 
  - The `SearchInput` component requires the `Label` component from `components/ui/label.js`.
  - **Fix:** Ensure `components/ui/label.js` exists and contains the standard Shadcn UI Label code (which depends on `@radix-ui/react-label`). Install the dependency if needed (`npm install @radix-ui/react-label`).

---

## License
MIT
