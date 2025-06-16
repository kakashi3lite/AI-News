# AI News Dashboard

**Smart news aggregation platform with AI-powered analysis and social features**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/ai-news-dashboard/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Node.js](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen.svg)](https://nodejs.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14.0.4-black.svg)](https://nextjs.org/)

---

## What is AI News Dashboard?

A modern news platform that uses AI to summarize articles, analyze content, and provide personalized recommendations. Built for journalists, researchers, and news enthusiasts who want intelligent news consumption.

### Key Features

- 🤖 **AI Summarization** - Get concise summaries of any news article
- 🔍 **Smart Search** - Context-aware search with voice support
- 📊 **Content Analysis** - Sentiment analysis and topic categorization
- 👥 **Social Features** - Comments, sharing, and community discussions
- 📱 **Responsive Design** - Works perfectly on all devices
- 🔒 **Privacy First** - GDPR compliant with secure authentication

---

## Quick Start

### Prerequisites

- Node.js 18+ 
- PostgreSQL database
- Git

### Installation

```bash
# Clone and setup
git clone https://github.com/your-username/ai-news-dashboard.git
cd ai-news-dashboard
npm install

# Setup environment
cp .env.example .env.local
# Edit .env.local with your configuration

# Setup database
npm run db:migrate
npm run db:seed

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

### Environment Variables

Create `.env.local` with these required variables:

```env
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/ai_news_dashboard"

# Authentication
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="your-secret-key"

# AI Services (Optional - uses mock data if not provided)
OPENAI_API_KEY="your-openai-key"

# News APIs (Optional - uses mock data if not provided)
NEWS_API_KEY="your-newsapi-key"
GUARDIAN_API_KEY="your-guardian-key"
```

---

## Technology Stack

**Frontend**
- Next.js 14 with App Router
- React 18 with TypeScript
- Tailwind CSS for styling
- Framer Motion for animations

**Backend**
- Next.js API routes
- Prisma ORM with PostgreSQL
- NextAuth.js for authentication

**AI & Analytics**
- OpenAI for summarization
- Custom sentiment analysis
- Vercel Analytics
- Sentry for error tracking

---

## Project Structure

```
ai-news-dashboard/
├── app/                 # Next.js App Router pages
├── components/          # React components
│   ├── ui/             # Base UI components
│   ├── news/           # News-specific components
│   └── social/         # Social features
├── lib/                # Utility libraries
│   ├── ai/             # AI processing
│   ├── news/           # News aggregation
│   └── utils/          # Helper functions
├── prisma/             # Database schema
├── public/             # Static assets
└── tests/              # Test files
```

---

## Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run test         # Run tests
npm run lint         # Check code quality
npm run db:studio    # Open database GUI
```

### Testing

```bash
# Unit tests
npm run test

# End-to-end tests
npm run test:e2e

# Test coverage
npm run test:coverage
```

### Mock Data Mode

The app works without external APIs using realistic mock data:

```bash
# Run with mock data (no API keys needed)
USE_MOCK_DATA=true npm run dev
```

---

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your repo to Vercel
3. Add environment variables in Vercel dashboard
4. Deploy automatically on push

### Manual Deployment

```bash
# Build and start
npm run build
npm run start
```

---

## API Documentation

### Core Endpoints

```
GET  /api/news              # Get latest news
GET  /api/news/[id]         # Get specific article
POST /api/news/search       # Search articles
POST /api/ai/summarize      # Summarize content
GET  /api/social/comments   # Get comments
```

Visit `/api/docs` for interactive API documentation.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `npm run test`
5. Commit: `git commit -m 'feat: add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards

- Use TypeScript for all new code
- Follow ESLint and Prettier rules
- Write tests for new features
- Use conventional commit messages

---

## Roadmap

### ✅ Completed
- Core news aggregation
- AI summarization
- User authentication
- Search functionality
- Social features

### 🔄 In Progress
- Voice search
- Real-time collaboration
- Advanced AI features
- Performance optimization

### 📋 Planned
- Mobile app
- Multi-language support
- Enterprise features
- API marketplace

---

## Support

- 📖 **Documentation**: Check the `/docs` folder
- 🐛 **Issues**: [GitHub Issues](https://github.com/ai-news-dashboard/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ai-news-dashboard/discussions)
- 📧 **Email**: support@ai-news-dashboard.com

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- OpenAI for AI models
- Vercel for hosting platform
- Next.js team for the framework
- All contributors and community members

---

<div align="center">

**Built with ❤️ for the news community**

[⭐ Star on GitHub](https://github.com/ai-news-dashboard) • [🐛 Report Bug](https://github.com/ai-news-dashboard/issues) • [💡 Request Feature](https://github.com/ai-news-dashboard/discussions)

</div>
