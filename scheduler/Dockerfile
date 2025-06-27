# RSE Scheduler Orchestrator Docker Configuration
# Multi-stage build for optimized production image

# Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies for building)
RUN npm ci

# Copy source code
COPY . .

# Run tests to ensure everything works
RUN npm test

# Run linting
RUN npm run lint

# Production stage
FROM node:18-alpine AS production

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init

# Create app user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S rse -u 1001

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application code from builder stage
COPY --from=builder /app/index.js ./
COPY --from=builder /app/jobs.yaml ./
COPY --from=builder /app/README.md ./

# Create logs directory
RUN mkdir -p logs && chown -R rse:nodejs logs

# Create directory for job scripts (if needed)
RUN mkdir -p scripts && chown -R rse:nodejs scripts

# Switch to non-root user
USER rse

# Expose metrics port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD node -e "const http = require('http'); \
                 const options = { hostname: 'localhost', port: 9090, path: '/health', timeout: 5000 }; \
                 const req = http.request(options, (res) => { \
                   if (res.statusCode === 200) process.exit(0); \
                   else process.exit(1); \
                 }); \
                 req.on('error', () => process.exit(1)); \
                 req.on('timeout', () => process.exit(1)); \
                 req.end();"

# Set environment variables
ENV NODE_ENV=production
ENV LOG_LEVEL=info

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the application
CMD ["node", "index.js"]

# Labels for metadata
LABEL maintainer="RSE Team <rse-team@yourcompany.com>"
LABEL version="1.0.0"
LABEL description="RSE Scheduler Orchestrator - Cron-like job scheduling for RSE news updates"
LABEL org.opencontainers.image.title="RSE Scheduler Orchestrator"
LABEL org.opencontainers.image.description="Production-ready job scheduler with dependency management and monitoring"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="RSE Team"
LABEL org.opencontainers.image.url="https://github.com/your-org/ai-news-dashboard"
LABEL org.opencontainers.image.source="https://github.com/your-org/ai-news-dashboard"
LABEL org.opencontainers.image.licenses="MIT"