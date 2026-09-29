import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    environmentMatchGlobs: [
      ['src/stores/**/*.test.ts', 'jsdom'],
    ],
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
});
