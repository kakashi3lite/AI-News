import '../styles.css';

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <title>AI News Dashboard</title>
        <meta name="description" content="Summarize news articles and YouTube videos with AI" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="font-sans antialiased bg-white text-gray-800 flex flex-col min-h-screen">
        <header className="bg-white shadow-md">
          <div className="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center">
            <h1 className="text-xl font-bold">AI News Dashboard</h1>
            <nav className="flex space-x-4">
              <a href="/" className="text-gray-700 hover:text-blue-500">Home</a>
              <a href="/dashboard" className="text-gray-700 hover:text-blue-500">Dashboard</a>
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="bg-gray-100">
          <div className="max-w-6xl mx-auto px-4 py-6 text-center text-sm text-gray-500">
            2025 AI News Dashboard. All rights reserved.
          </div>
        </footer>
      </body>
    </html>
  );
}
