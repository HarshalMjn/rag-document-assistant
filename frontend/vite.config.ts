import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The UI calls /api/...; the dev server forwards it to FastAPI without the prefix.
// The prefix keeps API paths (/documents, /chat) apart from the UI's own routes.
const apiTarget = process.env.API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // File events do not cross a Windows bind mount into Docker, so poll instead.
    watch: process.env.VITE_POLL ? { usePolling: true, interval: 300 } : undefined,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
