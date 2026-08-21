import { defineConfig } from "vite";

// Renderer ถูก serve/บิลด์จากโฟลเดอร์ src
// - dev: Vite dev server ที่ port 5173 (main process โหลด URL นี้)
// - build: ออกไฟล์ static ไปที่ dist/ ให้ Electron โหลดตอน production
export default defineConfig({
  root: "src",
  base: "./",
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
