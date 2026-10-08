// TRACE-5.0 Web Worker Engine
// Executes compute-intensive surveillance pipelines off the main browser UI thread
// ensuring 60 FPS fluid UI rendering, live progress meters, and zero thread locking.

import init, { run_trace50_pipeline_with_progress } from './pkg/trace50_wasm.js';

let wasmReady = false;

self.onmessage = async (e) => {
  const { type, fasta, config } = e.data;

  if (type === 'init') {
    try {
      if (!wasmReady) {
        await init();
        wasmReady = true;
      }
      self.postMessage({ type: 'wasm_ready' });
    } catch (err) {
      self.postMessage({ type: 'error', error: `WASM initialization failed in worker: ${err.message || err}` });
    }
    return;
  }

  if (type === 'run') {
    try {
      if (!wasmReady) {
        self.postMessage({
          type: 'progress',
          stage: 'STAGE_ALIGNMENT',
          pct: 0.02,
          detail: 'Initializing WebAssembly runtime in worker thread...'
        });
        await init();
        wasmReady = true;
      }

      const onProgress = (stage, pct, detail) => {
        self.postMessage({ type: 'progress', stage, pct, detail });
      };

      const configJson = config ? JSON.stringify(config) : null;
      const resultJson = run_trace50_pipeline_with_progress(fasta, configJson, onProgress);

      self.postMessage({ type: 'done', resultJson });
    } catch (err) {
      console.error("[TRACE-5.0 Worker Error]", err);
      self.postMessage({ type: 'error', error: err.message || String(err) });
    }
  }
};
