import init, { run_trace50_pipeline, version } from './pkg/trace50_wasm.js';

let wasmReady = false;
let currentDossier = null;
let currentNetworkMode = 'danno'; // 'danno' | 'static' | 'violations'

// DOM Elements
const statusIndicator = document.getElementById('wasm-status');
const fastaInput = document.getElementById('fasta-input');
const dropzone = document.getElementById('dropzone');
const runBtn = document.getElementById('btn-run');
const fastaNameBadge = document.getElementById('fasta-name-badge');

let loadedFastaContent = "";

// Initialize WebAssembly
async function setupWasm() {
  try {
    statusIndicator.textContent = "⏳ Initializing WebAssembly Kernel...";
    await init();
    wasmReady = true;
    statusIndicator.textContent = `⚡ Rust WebAssembly Engine v${version()} Ready`;
    statusIndicator.style.color = "#34d399";
    runBtn.disabled = false;
    // Auto-load Demo 1 on startup
    loadDemoDataset('demo_outbreak.fasta', 'Demo 1: Acute Outbreak vs. Clock Violations (10 seqs)');
  } catch (err) {
    console.error("Failed to initialize WASM:", err);
    statusIndicator.textContent = "❌ Failed to initialize WebAssembly engine";
    statusIndicator.style.color = "#f87171";
  }
}

// Drag & drop handlers
dropzone.addEventListener('click', () => fastaInput.click());
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});
fastaInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
});

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    loadedFastaContent = e.target.result;
    fastaNameBadge.textContent = `Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fastaNameBadge.style.display = 'inline-block';
  };
  reader.readAsText(file);
}

// Quick demo loading
window.loadDemoDataset = async function(filename, label) {
  try {
    statusIndicator.textContent = `⏳ Loading ${label}...`;
    const res = await fetch(`./sample_data/${filename}`);
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    loadedFastaContent = await res.text();
    fastaNameBadge.textContent = label;
    fastaNameBadge.style.display = 'inline-block';
    statusIndicator.textContent = `⚡ Rust WebAssembly Engine v${version()} Ready`;
    // Proactively run analysis on demo load
    runAnalysis();
  } catch (err) {
    console.error("Error loading demo dataset:", err);
    alert(`Failed to load demo dataset: ${err.message}`);
  }
};

// Run TRACE-5.0 Pipeline
runBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!wasmReady) {
    alert("WebAssembly engine is not ready yet.");
    return;
  }
  if (!loadedFastaContent || loadedFastaContent.trim().length === 0) {
    alert("Please select or drop a FASTA sequence file first.");
    return;
  }

  runBtn.disabled = true;
  runBtn.innerHTML = `<span>⏳ Analyzing Alignment...</span>`;

  try {
    // Gather config from DOM inputs
    const config = {
      mu: parseFloat(document.getElementById('param-mu').value) || 0.002,
      tau_bar: parseFloat(document.getElementById('param-tau-bar').value) || 1.0,
      omega: parseFloat(document.getElementById('param-omega').value) || 2.0,
      t_span: parseFloat(document.getElementById('param-t-span').value) || 20.0,
      fdr_threshold: parseFloat(document.getElementById('param-fdr').value) || 0.05,
      alpha_adequacy: parseFloat(document.getElementById('param-adequacy').value) || 0.05,
      static_distance_threshold: parseFloat(document.getElementById('param-static').value) || 0.015,
      r0: parseFloat(document.getElementById('param-r0').value) || 1.5
    };

    const startTime = performance.now();
    const resultJson = run_trace50_pipeline(loadedFastaContent, JSON.stringify(config));
    const elapsedMs = performance.now() - startTime;

    currentDossier = JSON.parse(resultJson);
    currentDossier.client_elapsed_ms = elapsedMs;

    renderScorecard(currentDossier);
    renderNetworkGraph(currentDossier);
    renderPhaseSpace(currentDossier);
    renderCommunities(currentDossier);
    renderMacroScaling(currentDossier);
    renderTable(currentDossier);

  } catch (err) {
    console.error("TRACE-5.0 Execution Error:", err);
    alert(`Error executing TRACE-5.0 pipeline:\n${err}`);
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = `<span>▶ Run Surveillance Analysis</span>`;
  }
}

// 1. Render Scorecard KPIs
function renderScorecard(dossier) {
  const kpi = dossier.kpi;
  const timeMs = dossier.client_elapsed_ms || (dossier.metadata.execution_time_seconds * 1000);

  document.getElementById('kpi-sequences').textContent = kpi.total_sequences;
  document.getElementById('kpi-sequences-sub').textContent = `Latency: ${timeMs.toFixed(1)} ms`;

  document.getElementById('kpi-comparisons').textContent = kpi.total_pairwise_comparisons.toLocaleString();
  document.getElementById('kpi-comparisons-sub').textContent = `${kpi.candidate_pairs_screened} candidate pairs screened`;

  document.getElementById('kpi-edges').textContent = kpi.certified_transmission_edges;
  document.getElementById('kpi-edges-sub').textContent = `FDR q ≤ ${dossier.metadata.parameters.fdr_threshold}`;

  document.getElementById('kpi-violations').textContent = kpi.physical_clock_violations_purged;
  document.getElementById('kpi-violations-sub').textContent = `Clock adequacy p < ${dossier.metadata.parameters.alpha_adequacy} (BF=0)`;

  const redPct = kpi.giant_component_reduction_pct.toFixed(1);
  document.getElementById('kpi-collapse').textContent = `${redPct}%`;
  document.getElementById('kpi-collapse-sub').textContent = `Static max ${kpi.static_network.max_cluster_size} → DANNO max ${kpi.danno_network.max_cluster_size}`;

  const nAct = kpi.inferred_active_transmitting_pool;
  const ci = kpi.inferred_active_pool_ci;
  document.getElementById('kpi-chin').textContent = nAct > 0 ? Math.round(nAct).toLocaleString() : 'N/A';
  document.getElementById('kpi-chin-sub').textContent = ci ? `95% CrI: [${Math.round(ci[0]).toLocaleString()}, ${Math.round(ci[1]).toLocaleString()}]` : 'Incomplete coverage';
}

// 2. Tab switching logic
window.switchTab = function(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  document.querySelector(`.tab-btn[data-tab="${tabId}"]`).classList.add('active');
  document.getElementById(`tab-${tabId}`).classList.add('active');

  // Redraw SVG if needed
  if (currentDossier) {
    if (tabId === 'network') renderNetworkGraph(currentDossier);
    if (tabId === 'phasespace') renderPhaseSpace(currentDossier);
  }
};

window.setNetworkMode = function(mode) {
  currentNetworkMode = mode;
  document.querySelectorAll('.segmented-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.segmented-btn[data-mode="${mode}"]`).classList.add('active');
  if (currentDossier) renderNetworkGraph(currentDossier);
};

// 3. Render D3 Force-Directed Network Graph
function renderNetworkGraph(dossier) {
  const container = document.getElementById('network-viz');
  container.innerHTML = "";
  const width = container.clientWidth || 900;
  const height = 520;

  const svg = d3.select(container)
    .append("svg")
    .attr("class", "viz-svg")
    .attr("viewBox", [0, 0, width, height]);

  const g = svg.append("g");

  // Zoom behavior
  svg.call(d3.zoom()
    .scaleExtent([0.2, 5])
    .on("zoom", (event) => g.attr("transform", event.transform)));

  // Filter edges based on current mode
  let filteredEdges = [];
  if (currentNetworkMode === 'danno') {
    filteredEdges = dossier.edges.filter(e => e.status === 'certified_transmission');
  } else if (currentNetworkMode === 'violations') {
    filteredEdges = dossier.edges.filter(e => e.status === 'clock_violation' || e.p_adequacy < dossier.metadata.parameters.alpha_adequacy);
  } else {
    // static 1.5%
    filteredEdges = dossier.edges.filter(e => e.distance <= dossier.metadata.parameters.static_distance_threshold);
  }

  // Deep copy nodes for D3 simulation
  const nodes = dossier.nodes.map(d => ({ ...d }));
  const nodeById = new Map(nodes.map(n => [n.id, n]));

  const links = [];
  filteredEdges.forEach(e => {
    const s = nodeById.get(e.source);
    const t = nodeById.get(e.target);
    if (s && t) {
      links.push({
        source: s,
        target: t,
        distance: e.distance,
        delta_t: e.delta_t_years,
        bf: e.bayes_factor,
        q: e.q_value,
        status: e.status
      });
    }
  });

  // Color palette for communities
  const colorScale = d3.scaleOrdinal(d3.schemeTableau10);

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(50))
    .force("charge", d3.forceManyBody().strength(-120))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(14));

  // Tooltip
  const tooltip = d3.select("#d3-tooltip");

  // Render Links
  const link = g.append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", d => {
      if (d.status === 'clock_violation') return "#a855f7"; // Purple for clock violation
      if (d.status === 'certified_transmission') return "#10b981"; // Green for certified
      return "#64748b"; // Grey for static noise bridges
    })
    .attr("stroke-width", d => d.status === 'certified_transmission' ? 2.5 : 1.2)
    .attr("stroke-dasharray", d => d.status === 'clock_violation' ? "4,4" : null)
    .attr("stroke-opacity", 0.7);

  // Render Nodes
  const node = g.append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", 7)
    .attr("fill", d => colorScale(d.community_id || d.cluster_id_certified || 0))
    .attr("stroke", "#ffffff")
    .attr("stroke-width", 1.5)
    .call(d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended));

  node.on("mouseover", (event, d) => {
    tooltip.style("display", "block")
      .html(`
        <strong>${d.id}</strong><br/>
        Date: ${d.date ? d.date.toFixed(2) : 'N/A'}<br/>
        Degree (Certified): ${d.degree_certified || 0}<br/>
        Degree (Static): ${d.degree_static || 0}<br/>
        Community: #${d.community_id || 0}
      `)
      .style("left", (event.pageX + 10) + "px")
      .style("top", (event.pageY - 28) + "px");
  }).on("mouseout", () => tooltip.style("display", "none"));

  link.on("mouseover", (event, d) => {
    tooltip.style("display", "block")
      .html(`
        <strong>${d.source.id} ↔ ${d.target.id}</strong><br/>
        Distance: ${(d.distance * 100).toFixed(2)}%<br/>
        Delay: ${d.delta_t.toFixed(2)} yr<br/>
        Bayes Factor: ${d.bf > 0 ? d.bf.toExponential(2) : '0.00'}<br/>
        FDR q: ${d.q !== undefined ? d.q.toFixed(4) : 'N/A'}<br/>
        Status: <span style="font-weight:700;">${d.status}</span>
      `)
      .style("left", (event.pageX + 10) + "px")
      .style("top", (event.pageY - 28) + "px");
  }).on("mouseout", () => tooltip.style("display", "none"));

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);
  });

  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
}

// 4. Render Phase Space Plot (d vs delta_t)
function renderPhaseSpace(dossier) {
  const container = document.getElementById('phasespace-viz');
  container.innerHTML = "";
  const width = container.clientWidth || 900;
  const height = 520;
  const margin = { top: 30, right: 30, bottom: 50, left: 60 };

  const svg = d3.select(container)
    .append("svg")
    .attr("class", "viz-svg")
    .attr("viewBox", [0, 0, width, height]);

  const maxDt = Math.max(4.0, d3.max(dossier.edges, e => e.delta_t_years) || 4.0);
  const maxD = Math.max(0.025, d3.max(dossier.edges, e => e.distance) || 0.025);

  const x = d3.scaleLinear()
    .domain([0, maxDt * 1.05])
    .range([margin.left, width - margin.right]);

  const y = d3.scaleLinear()
    .domain([0, maxD * 1.05])
    .range([height - margin.bottom, margin.top]);

  // Axes
  svg.append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(8))
    .attr("color", "#9ca3af");

  svg.append("text")
    .attr("x", width / 2)
    .attr("y", height - 12)
    .attr("fill", "#9ca3af")
    .attr("text-anchor", "middle")
    .attr("font-size", "12px")
    .text("Elapsed Sampling Separation ΔT (years)");

  svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).tickFormat(d => (d * 100).toFixed(1) + "%"))
    .attr("color", "#9ca3af");

  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -height / 2)
    .attr("y", 18)
    .attr("fill", "#9ca3af")
    .attr("text-anchor", "middle")
    .attr("font-size", "12px")
    .text("Pairwise Genetic Distance d (subs/site)");

  // Static threshold lines (1.5% and 0.5%)
  svg.append("line")
    .attr("x1", margin.left)
    .attr("x2", width - margin.right)
    .attr("y1", y(0.015))
    .attr("y2", y(0.015))
    .attr("stroke", "#f59e0b")
    .attr("stroke-dasharray", "4,4")
    .attr("stroke-width", 1.5);

  svg.append("text")
    .attr("x", width - margin.right - 10)
    .attr("y", y(0.015) - 6)
    .attr("fill", "#f59e0b")
    .attr("font-size", "11px")
    .attr("text-anchor", "end")
    .text("Standard TN93 (1.5%)");

  svg.append("line")
    .attr("x1", margin.left)
    .attr("x2", width - margin.right)
    .attr("y1", y(0.005))
    .attr("y2", y(0.005))
    .attr("stroke", "#ef4444")
    .attr("stroke-dasharray", "3,3")
    .attr("stroke-width", 1.2);

  svg.append("text")
    .attr("x", width - margin.right - 10)
    .attr("y", y(0.005) - 6)
    .attr("fill", "#ef4444")
    .attr("font-size", "11px")
    .attr("text-anchor", "end")
    .text("CDC Priority (0.5%)");

  // Dynamic molecular clock corridor curve: E[d | dt] = 2*mu*tau_bar + mu*dt
  const mu = dossier.metadata.parameters.mu;
  const tau_bar = dossier.metadata.parameters.tau_bar;
  const corridorData = d3.range(0, maxDt, 0.1).map(dt => ({
    dt: dt,
    d: 2 * mu * tau_bar + mu * dt
  }));

  const lineGen = d3.line()
    .x(d => x(d.dt))
    .y(d => y(d.d));

  svg.append("path")
    .datum(corridorData)
    .attr("fill", "none")
    .attr("stroke", "#3b82f6")
    .attr("stroke-width", 2.5)
    .attr("d", lineGen);

  svg.append("text")
    .attr("x", x(maxDt * 0.7))
    .attr("y", y(2 * mu * tau_bar + mu * (maxDt * 0.7)) - 8)
    .attr("fill", "#60a5fa")
    .attr("font-size", "11px")
    .text("Dynamic Clock Corridor E[d | ΔT]");

  // Plot Dyad Points
  const tooltip = d3.select("#d3-tooltip");

  svg.append("g")
    .selectAll("circle")
    .data(dossier.edges)
    .join("circle")
    .attr("cx", d => x(d.delta_t_years))
    .attr("cy", d => y(d.distance))
    .attr("r", 5.5)
    .attr("fill", d => {
      if (d.status === 'clock_violation') return "#a855f7";
      if (d.status === 'certified_transmission') return "#10b981";
      return "#6b7280";
    })
    .attr("stroke", "#ffffff")
    .attr("stroke-width", 1)
    .on("mouseover", (event, d) => {
      tooltip.style("display", "block")
        .html(`
          <strong>${d.source} ↔ ${d.target}</strong><br/>
          Distance: ${(d.distance * 100).toFixed(3)}%<br/>
          Elapsed Delay: ${d.delta_t_years.toFixed(2)} yr<br/>
          Substitutions K: ${d.k_substitutions.toFixed(0)}<br/>
          Bayes Factor: ${d.bayes_factor > 0 ? d.bayes_factor.toExponential(2) : '0.00'}<br/>
          Clock Adequacy p: ${d.p_adequacy.toExponential(3)}<br/>
          Status: <strong>${d.status}</strong>
        `)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 28) + "px");
    })
    .on("mouseout", () => tooltip.style("display", "none"));
}

// 5. Render Communities
function renderCommunities(dossier) {
  const container = document.getElementById('comm-container');
  container.innerHTML = "";

  const comms = dossier.communities || [];
  if (comms.length === 0) {
    container.innerHTML = `<p style="color:var(--text-muted); padding:1rem;">No multi-isolate communities resolved.</p>`;
    return;
  }

  comms.forEach(c => {
    const card = document.createElement('div');
    card.className = "comm-card";

    let tierClass = "tier-endemic";
    if (c.operational_tier.includes("Outbreak")) tierClass = "tier-active";
    else if (c.operational_tier.includes("Emergent")) tierClass = "tier-emergent";
    else if (c.operational_tier.includes("Chronic")) tierClass = "tier-chronic";

    card.innerHTML = `
      <div class="comm-header">
        <span style="font-weight:700; color:var(--text-primary);">Community #${c.community_id} (${c.size} patients)</span>
        <span class="tier-badge ${tierClass}">${c.operational_tier}</span>
      </div>
      <div style="font-size:0.8rem; color:var(--text-secondary); display:flex; flex-direction:column; gap:0.25rem;">
        <div>Evolutionary Velocity μ: <strong style="color:var(--text-primary);">${(c.mu * 1000).toFixed(2)} subs/kb/yr</strong></div>
        <div>Clock Linearity R²: <strong style="color:var(--text-primary);">${c.r_squared.toFixed(3)}</strong></div>
        <div>Emergence Horizon t_MRCA: <strong style="color:var(--text-primary);">${c.tmrca > 0 ? c.tmrca.toFixed(1) : 'Unidentifiable'}</strong></div>
        <div>Fieller Bound: <span style="font-family:var(--font-mono); font-size:0.75rem;">${c.fieller_status}</span></div>
      </div>
    `;
    container.appendChild(card);
  });
}

// 6. Render Macro Scaling (CHIN)
function renderMacroScaling(dossier) {
  const macro = dossier.macro_scaling;
  const container = document.getElementById('macro-container');
  if (!macro) {
    container.innerHTML = `<p style="color:var(--text-muted);">Macro-epidemic scaling not available for this alignment size.</p>`;
    return;
  }

  container.innerHTML = `
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:1rem; margin-bottom:1.5rem;">
      <div class="card" style="background:var(--bg-secondary);">
        <span class="kpi-label">Active Transmitting Pool (N_act)</span>
        <span class="kpi-value" style="color:var(--accent-cyan);">${Math.round(macro.active_population.n_act).toLocaleString()}</span>
        <span class="kpi-sub">95% CrI: [${Math.round(macro.active_population.ci_lower).toLocaleString()}, ${Math.round(macro.active_population.ci_upper).toLocaleString()}]</span>
      </div>
      <div class="card" style="background:var(--bg-secondary);">
        <span class="kpi-label">Surveillance Sampling Fraction (ρ)</span>
        <span class="kpi-value" style="color:var(--accent-green);">${(macro.surveillance_coverage_rho * 100).toFixed(2)}%</span>
        <span class="kpi-sub">Assumed basic R₀ = ${macro.active_population.r0.toFixed(1)}</span>
      </div>
      <div class="card" style="background:var(--bg-secondary);">
        <span class="kpi-label">Borel Branching Decay</span>
        <span class="kpi-value" style="color:var(--accent-amber);">${macro.borel_decay.rejects_uniform_sampling ? 'Supercritical Spread' : 'Controlled Decay'}</span>
        <span class="kpi-sub">Max observed cluster size: ${macro.borel_decay.max_cluster_size}</span>
      </div>
    </div>
    <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6; background:var(--bg-secondary); padding:1.25rem; border-radius:var(--radius-md); border:1px solid var(--border-color);">
      <strong>CHIN Macro-Epidemic Inversion Principle:</strong><br/>
      While sampling individual clinical cases scales linearly with surveillance coverage (E[n] = M · ρ), capturing direct transmission dyads requires sampling both partners independently (E[E] = (M - 1) · ρ²). By inverting this quadratic collision law under Negative Binomial contact heterogeneity, TRACE-5.0 sizes the unobserved iceberg of active community transmission directly from observed cluster collisions in seconds.
    </div>
  `;
}

// 7. Render Data Table
function renderTable(dossier) {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = "";

  const edges = dossier.edges || [];
  edges.forEach(e => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${e.source}</td>
      <td>${e.target}</td>
      <td>${(e.distance * 100).toFixed(2)}%</td>
      <td>${e.delta_t_years.toFixed(2)}</td>
      <td>${e.k_substitutions.toFixed(0)}</td>
      <td>${e.bayes_factor > 0 ? e.bayes_factor.toExponential(2) : '0.00'}</td>
      <td>${e.q_value !== undefined ? e.q_value.toFixed(4) : 'N/A'}</td>
      <td>${e.p_adequacy.toExponential(2)}</td>
      <td><span style="font-weight:600; color:${e.status === 'certified_transmission' ? '#34d399' : (e.status === 'clock_violation' ? '#c084fc' : '#9ca3af')}">${e.status}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// 8. Export Utilities
window.exportEdgesCsv = function() {
  if (!currentDossier || !currentDossier.edges) return;
  const headers = ["source", "target", "distance", "delta_t_years", "k_substitutions", "bayes_factor", "q_value", "p_adequacy", "status"];
  const rows = currentDossier.edges.map(e => [
    e.source, e.target, e.distance, e.delta_t_years, e.k_substitutions, e.bayes_factor, e.q_value, e.p_adequacy, e.status
  ]);
  const csvContent = [headers.join(",")].concat(rows.map(r => r.join(","))).join("\n");
  downloadBlob(csvContent, "trace50_certified_edges.csv", "text/csv");
};

window.exportDossierJson = function() {
  if (!currentDossier) return;
  const jsonContent = JSON.stringify(currentDossier, null, 2);
  downloadBlob(jsonContent, "trace50_visualization_dossier.json", "application/json");
};

function downloadBlob(content, filename, contentType) {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Start WebAssembly engine on page load
setupWasm();
