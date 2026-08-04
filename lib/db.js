import { PrismaClient } from '@prisma/client';

// Prisma client singleton — avoids exhausting connections on hot reload in dev.
const globalForPrisma = globalThis;

export const prisma = globalForPrisma.prisma || new PrismaClient();

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma;

export default prisma;
