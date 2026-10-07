# TRACE-5.0: High-Performance Rust & WebAssembly Molecular Surveillance Engine

> **Dynamic Molecular Surveillance of Viral Transmission Networks**  
> Pure Rust Algorithmic Kernel (`trace50-core`), Native Standalone CLI (`trace50-cli`), and WebAssembly Client Engine (`trace50-wasm`).

---

## 1. Architecture Overview

```
trace50-rust/
├── Cargo.toml                     # Workspace root manifest
├── crates/
│   ├── trace50-core/              # Algorithmic kernel (zero C-dependencies, pure Rust, WASM-safe)
│   │   ├── src/
│   │   │   ├── math.rs            # Lanczos ln_gamma, incomplete upper gamma gammaincc, Poisson CDF, Jacobi eigensolver
│   │   │   ├── tn93.rs            # Bitmask IUPAC ambiguous Tamura-Nei 93 pairwise distance & FASTA parser
│   │   │   ├── danno.rs           # Closed-form Kingman coalescent Bayes Factor & Benjamini-Hochberg FDR
│   │   │   ├── info_danno.rs      # Forensic allele-frequency weighting & DRM downweighting
│   │   │   ├── autoclock.rs       # STEVE recursive graph Laplacian spectral deconvolution & Fieller t_MRCA
│   │   │   ├── chin.rs            # Macro-epidemic quadratic scaling law inversion & Borel branching decay
│   │   │   ├── network.rs         # Connected components, giant component tracking & Borel cluster decay
│   │   │   ├── visualization.rs   # Comprehensive Visualization Dossier JSON schema
│   │   │   └── pipeline.rs        # End-to-end execution orchestrator
│   │   └── tests/
│   │       └── test_pipeline.rs   # Automated integration tests
│   ├── trace50-cli/               # Native multi-platform CLI binary (`trace50`)
│   │   └── src/main.rs            # Command-line interface with rich scorecard reporting
│   └── trace50-wasm/              # WebAssembly package for browsers, Node.js, and Observable Framework
│       ├── src/lib.rs             # wasm-bindgen bindings and JS Object serialization
│       ├── pkg/                   # Compiled WASM distribution (190 KB)
│       └── test_wasm.mjs          # Node.js automated verification suite
```

---

## 2. Quickstart: Native CLI Executable

### Build
```bash
cd trace50-rust
cargo build --release --bin trace50
```
The optimized native binary will be generated at `target/release/trace50`.

### Run
```bash
./target/release/trace50 \
  --fasta path/to/alignment.fasta \
  --meta path/to/metadata.csv \
  --id-col accession \
  --date-col collection_date \
  --fdr 0.05 \
  --out-dir results_dir/
```

### Outputs
- `trace50_visualization_dossier.json`: Comprehensive, self-contained JSON dossier ready for Observable Framework, D3.js, Cytoscape, and web dashboards.
- `acute_transmission_dyads.csv`: All screened candidate dyads with distances, time intervals, Bayes Factors, PEPs, $q$-values, and certification status.
- `autoclock_communities.csv`: Resolved STEVE spectral communities with evolutionary velocities $\mu$, $t_{\text{MRCA}}$, and Fieller bounds.
- `patient_surveillance_assignments.csv`: Patient-level cluster and community assignments with operational triage tiers.

---

## 3. Quickstart: WebAssembly (WASM) & Browser Runtime

### Compile to WebAssembly
```bash
cd crates/trace50-wasm
wasm-pack build --target web --out-dir pkg
```

### Run Node.js Verification Test
```bash
node test_wasm.mjs
```

### Web / Browser / Observable Usage
```javascript
import init, { run_trace50_object, compute_tn93, compute_danno_bf } from './pkg/trace50_wasm.js';

await init();

// 1. Direct pairwise distance
const d = compute_tn93("ACGTACGT", "GCGTACGT");

// 2. Continuous Bayes Factor
const bf = compute_danno_bf(0.001, 0.2, 1000, 2e-3, 1.0, 2.0, 20.0);

// 3. Full end-to-end pipeline execution from browser memory:
const fastaContent = ">seq1_2015.0\nACGT...\n>seq2_2015.2\nACGT...";
const config = {
  mu: 2.0e-3,
  tau_bar: 1.0,
  omega: 2.0,
  fdr_threshold: 0.05
};

const dossier = run_trace50_object(fastaContent, config);
console.log("Certified edges:", dossier.kpi.certified_transmission_edges);
console.log("Giant component reduction:", dossier.kpi.giant_component_reduction_pct);
console.log("Estimated active pool N_act:", dossier.kpi.inferred_active_transmitting_pool);
```

---

## 4. Full Visualization JSON Schema

The `VisualizationDossier` JSON schema provides:
- **`metadata`**: Sequence count, site count, temporal range, execution runtime, parameters.
- **`kpi`**: Total comparisons, candidate dyads, clock violations purged, certified transmission links, static vs. DANNO giant component sizes, active transmitting pool $N_{\text{act}}$, and surveillance sampling fraction $\hat{\rho}$.
- **`nodes`**: Node list with dates, static degrees, certified degrees, cluster IDs, and STEVE operational tiers.
- **`edges`**: Dyad list with pairwise distances, $\Delta T$, Bayes Factors, PEPs, $q$-values, adequacy $p$-values, and classification status.
- **`clusters`**: Certified cluster size distributions, member IDs, time spans.
- **`communities`**: STEVE spectral communities with evolutionary velocities $\mu$, $t_{\text{MRCA}}$, Fieller's theorem bounds, and active outbreak flags.
- **`macro_scaling`**: CHIN quadratic collision inversion parameters and Borel branching decay fits.
- **`phase_space_points`**: $(d, \Delta T)$ phase plot points categorized by status ("certified_transmission", "clock_violation", "unsupported_candidate").
- **`borel_spectrum`**: Cluster size frequency distribution for assessing percolation collapse vs. organic branching decay.
