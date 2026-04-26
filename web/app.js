// ── MiniLLM Web UI ───────────────────────────────────────────────────
// Single-page app that visualizes transformer internals in real time.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let modelInfo = null;
let lastResponse = null;
let activeTab = "pipeline";

// ── Init ─────────────────────────────────────────────────────────
async function init() {
  try {
    const res = await fetch("/api/info");
    modelInfo = await res.json();
    $("#param-badge").textContent = `${modelInfo.parameters.toLocaleString()} params`;
    $("#arch-info").textContent =
      `d=${modelInfo.d_model}  heads=${modelInfo.n_heads}  ` +
      `layers=${modelInfo.n_layers}  ctx=${modelInfo.max_seq_len}  ` +
      `vocab=${modelInfo.vocab_size}`;
    addSystemMsg('MiniLLM ready! Try: "lakers", "bulls wears", "celtics from"');
  } catch (e) {
    addSystemMsg("Failed to connect to server.");
  }

  // Tab switching
  $$(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeTab = tab.dataset.tab;
      if (activeTab === "vocabulary") {
        renderVocabulary($("#viz-content"));
      } else if (lastResponse) {
        renderViz(lastResponse);
      }
    });
  });

  // Form submit
  $("#chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#prompt-input");
    const prompt = input.value.trim();
    if (!prompt) return;
    input.value = "";
    await sendMessage(prompt);
  });
}

// ── Chat ─────────────────────────────────────────────────────────
function addMsg(text, cls) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  $("#messages").appendChild(div);
  $("#messages").scrollTop = $("#messages").scrollHeight;
}
function addSystemMsg(text) { addMsg(text, "system"); }

async function sendMessage(prompt) {
  addMsg(prompt, "user");
  const btn = $("#send-btn");
  btn.disabled = true;
  btn.textContent = "…";

  try {
    const body = {
      prompt,
      temperature: parseFloat($("#temperature").value) || 0.8,
      max_tokens: parseInt($("#max-tokens").value) || 10,
      use_rag: $("#use-rag").checked,
      system_prompt: $("#system-prompt").value.trim(),
    };

    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    lastResponse = data;

    // Show generated text
    if (data.rag_document) {
      addMsg(`📄 RAG: "${data.rag_document}"`, "system");
    }
    addMsg(data.full_text, "bot");

    // Render visualization
    renderViz(data);
  } catch (e) {
    addMsg("Error: " + e.message, "system");
  }

  btn.disabled = false;
  btn.textContent = "Send";
}

// ── Visualization Router ───────────────────────────────────────
function renderViz(data) {
  const container = $("#viz-content");
  container.innerHTML = "";

  switch (activeTab) {
    case "pipeline":  renderPipeline(container, data); break;
    case "attention":  renderAttention(container, data); break;
    case "embeddings": renderEmbeddings(container, data); break;
    case "vocabulary": renderVocabulary(container); break;
  }
}

// ── Pipeline View ────────────────────────────────────────────
function renderPipeline(el, data) {
  const html = [];

  // Step 1: Tokenization
  html.push(pipeSection(1, "Tokenization", `
    <p style="font-size:0.75rem;color:var(--text2);margin-bottom:0.4rem">
      Text → Token IDs (word-level tokenizer, vocab=${modelInfo.vocab_size})
    </p>
    <div class="token-row">
      ${data.input_tokens.map(t =>
        `<div class="token-chip ${t.id < 5 ? 'special' : ''}">
          <span class="tok-word">${esc(t.token)}</span>
          <span class="tok-id">${t.id}</span>
        </div>`
      ).join("")}
    </div>
  `));

  // Step 2: Context Window
  const ctx = data.context_map;
  const sysLen = data.input_tokens.filter(t =>
    data.input_tokens.indexOf(t) < data.input_tokens.findIndex(t2 => t2.token === "<sep>")
  ).length;
  html.push(pipeSection(2, "Context Window", `
    <p style="font-size:0.75rem;color:var(--text2);margin-bottom:0.3rem">
      ${ctx.used}/${ctx.total_slots} slots used
      (${ctx.input_len} input + ${ctx.generated_len} generated)
    </p>
    <div class="ctx-bar">
      ${ctxBar("user", ctx.input_len, ctx.total_slots)}
      ${ctxBar("gen", ctx.generated_len, ctx.total_slots)}
      ${ctxBar("pad", ctx.total_slots - ctx.used, ctx.total_slots)}
    </div>
    <div style="display:flex;gap:0.8rem;margin-top:0.3rem;font-size:0.65rem;color:var(--text2)">
      <span>🟢 Input</span> <span>🟣 Generated</span> <span>⬜ Unused</span>
    </div>
  `));

  // Step 3: Generation Steps
  const stepsHtml = data.steps.map((s, i) => `
    <div class="gen-step">
      <div class="chosen">${esc(s.chosen_token)}</div>
      <div class="prob-bar-container">
        ${s.top_k.slice(0, 5).map(tk => `
          <div class="prob-bar-row">
            <span class="prob-bar-label">${esc(tk.token)}</span>
            <div class="prob-bar" style="width:${Math.max(tk.probability * 200, 2)}px;
              background:${tk.token === s.chosen_token ? 'var(--green)' : 'var(--border)'}"></div>
            <span class="prob-bar-val">${(tk.probability * 100).toFixed(1)}%</span>
          </div>
        `).join("")}
      </div>
    </div>
  `).join("");

  html.push(pipeSection(3, "Generation (next-token prediction)", `
    <div class="gen-steps">${stepsHtml}</div>
  `));

  // Step 4: Output
  html.push(pipeSection(4, "Decoded Output", `
    <div class="token-row">
      ${data.token_labels.map((t, i) => {
        const isGen = i >= data.input_tokens.length;
        const cls = isGen ? "generated" : (i < 5 && ["<bos>","<sep>","<pad>"].includes(t) ? "special" : "");
        return `<div class="token-chip ${cls}">
          <span class="tok-word">${esc(t)}</span>
          <span class="tok-id">${i}</span>
        </div>`;
      }).join("")}
    </div>
    <p style="margin-top:0.5rem;font-size:0.85rem;color:var(--green);font-family:var(--mono)">
      → ${esc(data.full_text)}
    </p>
  `));

  el.innerHTML = `<div class="pipeline">${html.join("")}</div>`;
}

function pipeSection(num, title, body) {
  return `<div class="pipe-section">
    <div class="pipe-header">
      <span class="step-num">${num}</span> ${title}
    </div>
    <div class="pipe-body">${body}</div>
  </div>`;
}

function ctxBar(cls, count, total) {
  if (count <= 0) return "";
  const pct = (count / total * 100).toFixed(1);
  return `<div class="ctx-seg ${cls}" style="width:${pct}%">${count}</div>`;
}

// ── Attention View ───────────────────────────────────────────
function renderAttention(el, data) {
  if (!data.attention || !data.attention.length) {
    el.innerHTML = '<div class="viz-placeholder">No attention data</div>';
    return;
  }

  const labels = data.token_labels;
  const grid = document.createElement("div");
  grid.className = "attn-grid";

  for (const layer of data.attention) {
    for (const head of layer.heads) {
      const card = document.createElement("div");
      card.className = "attn-card";

      const title = document.createElement("h4");
      title.textContent = `Layer ${layer.layer + 1}, Head ${head.head + 1}`;
      card.appendChild(title);

      const canvas = document.createElement("canvas");
      canvas.className = "attn-canvas";
      const T = head.weights.length;
      const cellSize = Math.max(18, Math.min(32, 300 / T));
      const labelSpace = 60;
      const size = T * cellSize + labelSpace;
      canvas.width = size;
      canvas.height = size;
      canvas.style.maxWidth = "100%";
      canvas.style.height = "auto";

      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#1a1d27";
      ctx.fillRect(0, 0, size, size);

      // Draw heatmap
      for (let r = 0; r < T; r++) {
        for (let c = 0; c < T; c++) {
          const v = head.weights[r][c];
          const intensity = Math.floor(v * 255);
          ctx.fillStyle = `rgb(${40 + intensity * 0.4}, ${60 + intensity * 0.5}, ${255})`;
          ctx.fillRect(labelSpace + c * cellSize, labelSpace + r * cellSize,
                       cellSize - 1, cellSize - 1);
        }
      }

      // Draw labels
      ctx.fillStyle = "#8b8fa8";
      ctx.font = `${Math.min(10, cellSize * 0.6)}px monospace`;
      ctx.textAlign = "right";
      for (let i = 0; i < T; i++) {
        const label = labels[i] || "?";
        // Row labels (left)
        ctx.save();
        ctx.textAlign = "right";
        ctx.fillText(label, labelSpace - 4, labelSpace + i * cellSize + cellSize * 0.65);
        ctx.restore();
        // Column labels (top, rotated)
        ctx.save();
        ctx.translate(labelSpace + i * cellSize + cellSize * 0.65, labelSpace - 4);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = "left";
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }

      card.appendChild(canvas);
      grid.appendChild(card);
    }
  }

  el.appendChild(grid);
}

// ── Embeddings View ──────────────────────────────────────────
function renderEmbeddings(el, data) {
  if (!data.embeddings || !data.embeddings.length) {
    el.innerHTML = '<div class="viz-placeholder">No embedding data</div>';
    return;
  }

  const container = document.createElement("div");
  container.className = "embed-container";

  const canvas = document.createElement("canvas");
  canvas.className = "embed-canvas";
  container.appendChild(canvas);
  el.appendChild(container);

  // Wait for layout, then draw
  requestAnimationFrame(() => drawEmbeddings(canvas, data.embeddings));
}

function drawEmbeddings(canvas, points) {
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = rect.width;
  const H = rect.height || 500;
  canvas.width = W * 2;   // retina
  canvas.height = H * 2;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";

  const ctx = canvas.getContext("2d");
  ctx.scale(2, 2);
  ctx.fillStyle = "#0f1117";
  ctx.fillRect(0, 0, W, H);

  // Compute bounds
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const margin = 50;
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  function tx(x) { return margin + (x - xMin) / xRange * (W - 2 * margin); }
  function ty(y) { return margin + (y - yMin) / yRange * (H - 2 * margin); }

  // Draw grid
  ctx.strokeStyle = "#2e3348";
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(margin, H / 2); ctx.lineTo(W - margin, H / 2);
  ctx.moveTo(W / 2, margin); ctx.lineTo(W / 2, H - margin);
  ctx.stroke();

  // Draw reference points
  for (const p of points.filter(p => p.type === "reference")) {
    const px = tx(p.x), py = ty(p.y);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(139,143,168,0.4)";
    ctx.fill();
    ctx.fillStyle = "#8b8fa8";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(p.word, px, py - 7);
  }

  // Draw input points (highlighted)
  for (const p of points.filter(p => p.type === "input")) {
    const px = tx(p.x), py = ty(p.y);
    // Glow
    ctx.beginPath();
    ctx.arc(px, py, 10, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(108,140,255,0.15)";
    ctx.fill();
    // Dot
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#6c8cff";
    ctx.fill();
    // Label
    ctx.fillStyle = "#e4e6f0";
    ctx.font = "bold 11px monospace";
    ctx.textAlign = "center";
    ctx.fillText(p.word, px, py - 10);
  }

  // Axis labels
  ctx.fillStyle = "#8b8fa8";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("PC 1", W / 2, H - 10);
  ctx.save();
  ctx.translate(14, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("PC 2", 0, 0);
  ctx.restore();
}

// ── Vocabulary View ───────────────────────────────────────────
const VOCAB_CATEGORIES = {
  special: { label: "Special", color: "var(--accent)",
    words: new Set(["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]) },
  noun: { label: "Teams", color: "var(--accent2)",
    words: new Set(["lakers","celtics","bulls","warriors"]) },
  verb: { label: "Cities / States", color: "var(--green)",
    words: new Set(["la","boston","chicago","sf","california","massachusetts","illinois"]) },
  adj: { label: "Colors / Conferences", color: "var(--orange)",
    words: new Set(["yellow","green","red","white","east","west"]) },
};

function categorize(word) {
  for (const [cat, info] of Object.entries(VOCAB_CATEGORIES)) {
    if (info.words.has(word)) return cat;
  }
  return "other";
}

function renderVocabulary(el) {
  if (!modelInfo) {
    el.innerHTML = '<div class="viz-placeholder">Loading model info…</div>';
    return;
  }
  el.innerHTML = "";

  const vocab = modelInfo.vocabulary; // array of words, index = token ID

  // Stats
  const stats = document.createElement("div");
  stats.className = "vocab-stats";
  stats.textContent = `${vocab.length} tokens · d_model=${modelInfo.d_model} · each token → ${modelInfo.d_model}-dim vector`;
  el.appendChild(stats);

  // Legend
  const legend = document.createElement("div");
  legend.className = "vocab-legend";
  const cats = [
    ["special", "Special"],
    ["noun", "Teams"],
    ["verb", "Cities / States"],
    ["adj", "Colors / Conferences"],
    ["other", "Grammar"],
  ];
  const catColors = {
    special: "var(--accent)", noun: "var(--accent2)",
    verb: "var(--green)", adj: "var(--orange)", other: "var(--purple)",
  };
  for (const [key, label] of cats) {
    legend.innerHTML += `<span><span class="legend-dot" style="background:${catColors[key]}"></span>${label}</span>`;
  }
  el.appendChild(legend);

  // Group by category
  const groups = { special: [], noun: [], verb: [], adj: [], other: [] };
  vocab.forEach((word, id) => {
    const cat = categorize(word);
    groups[cat].push({ word, id });
  });

  const sectionLabels = {
    special: "Special Tokens",
    noun: "Teams",
    verb: "Cities / States",
    adj: "Colors / Conferences",
    other: "Grammar",
  };

  for (const [cat, tokens] of Object.entries(groups)) {
    if (tokens.length === 0) continue;
    const section = document.createElement("div");
    section.className = "vocab-section";

    const h3 = document.createElement("h3");
    h3.textContent = `${sectionLabels[cat]} (${tokens.length})`;
    section.appendChild(h3);

    const grid = document.createElement("div");
    grid.className = "vocab-grid";
    for (const t of tokens) {
      grid.innerHTML += `<div class="vocab-chip cat-${cat}"><span class="v-id">${t.id}</span>${esc(t.word)}</div>`;
    }
    section.appendChild(grid);
    el.appendChild(section);
  }
}

// ── Utilities ────────────────────────────────────────────────────
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// ── Boot ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
