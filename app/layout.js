import { Inter, Space_Grotesk } from 'next/font/google';
import { ThemeProvider } from './ThemeProvider';
import ThemeToggle from '../components/ThemeToggle';
import '../styles.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  weight: ['400', '500', '600', '700'],
  display: 'swap',
});

export const metadata = {
  title: 'AI News Dashboard',
  description: 'Summarize news articles and YouTube videos with AI',
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${spaceGrotesk.variable}`}
    >
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="font-sans antialiased flex flex-col min-h-screen" style={{ backgroundColor: 'var(--bg)', color: 'var(--fg)' }}>
        <ThemeProvider>
          <header
            className="sticky top-0 z-20 backdrop-blur-md border-b"
            style={{ backgroundColor: 'var(--header-bg)', borderColor: 'var(--header-border)' }}
          >
            <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 flex justify-between items-center">
              <a href="/" className="flex items-center gap-2 group">
                <span
                  className="w-7 h-7 rounded-lg flex items-center justify-center text-white text-xs font-bold flex-shrink-0"
                  style={{ background: 'linear-gradient(135deg, var(--primary), #7c3aed)' }}
                >
                  AI
                </span>
                <span className="font-display font-semibold text-base tracking-tight" style={{ color: 'var(--fg)' }}>
                  AI News
                </span>
              </a>

              <div className="flex items-center gap-1 sm:gap-3">
                <nav className="hidden sm:flex items-center gap-1">
                  <a
                    href="/"
                    className="px-3 py-1.5 rounded-md text-sm font-medium transition-colors"
                    style={{ color: 'var(--fg-muted)' }}
                    onMouseEnter={e => { e.target.style.color = 'var(--fg)'; e.target.style.backgroundColor = 'var(--surface-muted)'; }}
                    onMouseLeave={e => { e.target.style.color = 'var(--fg-muted)'; e.target.style.backgroundColor = ''; }}
                  >
                    Home
                  </a>
                  <a
                    href="/dashboard"
                    className="px-3 py-1.5 rounded-md text-sm font-medium transition-colors"
                    style={{ color: 'var(--fg-muted)' }}
                    onMouseEnter={e => { e.target.style.color = 'var(--fg)'; e.target.style.backgroundColor = 'var(--surface-muted)'; }}
                    onMouseLeave={e => { e.target.style.color = 'var(--fg-muted)'; e.target.style.backgroundColor = ''; }}
                  >
                    Dashboard
                  </a>
                </nav>
                <ThemeToggle />
              </div>
            </div>
          </header>

          <main className="flex-1">{children}</main>

          <footer className="border-t" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
            <div className="max-w-6xl mx-auto px-4 py-6 flex flex-col sm:flex-row justify-between items-center gap-2 text-sm" style={{ color: 'var(--fg-muted)' }}>
              <span>2026 AI News Dashboard</span>
              <span className="text-xs opacity-60">Powered by Claude AI</span>
            </div>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}
