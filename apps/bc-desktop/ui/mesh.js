// The Mesh page: one model across this account's paired devices, served on this computer.
// All state lives in the Rust side (`src/mesh.rs`); this file renders it. Own scope, like lend.js.

(() => {
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const copyIcon = `<svg class="icon"><use href="#i-copy"/></svg>Copy`;
const PORT = 8484;

let goal = "latency";
let running = null;
let models = [];

function message(text, kind = "bad") {
  $("mesh-msg").className = `banner ${kind}${text ? "" : " hidden"}`;
  $("mesh-msg").textContent = text || "";
}

function busy(text) {
  $("mesh-phase").classList.toggle("hidden", !text);
  $("mesh-phase-text").textContent = text || "";
  $("mesh-run").disabled = !!text;
  $("mesh-measure").disabled = !!text;
}

const fmt = (n, d = 0) => Number(n).toFixed(d);

function renderDevices(devices) {
  if (!devices.length) {
    $("mesh-devices").innerHTML = `<div class="board-empty">This computer is not paired with any device yet. Pair one under Use first.</div>`;
    return;
  }
  $("mesh-devices").innerHTML = devices.map((d) => {
    if (d.error) {
      return `<div class="mesh-dev"><span class="dep-main"><span class="dep-cmd">${esc(d.name)}</span><span class="dep-sub">${esc(d.error)}</span></span><span class="pill bad">Unreachable</span></div>`;
    }
    const pill = d.engine ? `<span class="pill ok">Ready</span>` : `<span class="pill warn">Update needed</span>`;
    return `
      <div class="mesh-dev">
        <span class="dep-main">
          <span class="dep-cmd">${esc(d.name)}</span>
          <span class="dep-sub">
            <span class="kv"><b>Path</b> ${d.relay ? "relay" : "direct"}</span>
            <span class="kv"><b>Round trip</b> <span class="num">${fmt(d.rtt_ms, 1)}<i>ms</i></span></span>
            <span class="kv"><b>Upload</b> <span class="num">${fmt(d.up_mbps, 1)}<i>Mbit/s</i></span></span>
            <span class="kv"><b>Free memory</b> <span class="num">${fmt(d.free_gb, 1)}<i>of ${fmt(d.total_gb)} GB</i></span></span>
            <span class="kv"><b>Memory speed</b> <span class="num">${fmt(d.mem_gbs, 1)}<i>GB/s</i></span></span>
          </span>
        </span>
        ${pill}
      </div>`;
  }).join("");
}

function renderPlan(pipelines, unused = []) {
  $("mesh-plan-block").classList.toggle("hidden", !pipelines.length);
  if (!pipelines.length) return;
  const layers = Math.max(...pipelines.flatMap((p) => p.stages.map((s) => s.end)));
  $("mesh-plan").innerHTML = pipelines.map((p, i) => {
    const head = pipelines.length > 1 ? `<div class="board-head"><span>Copy ${i + 1} of the model</span></div>` : "";
    const stages = p.stages.map((s, k) => `
      <div class="mesh-stage">
        <span class="mesh-step">${k + 1}</span>
        <span class="dep-main">
          <span class="dep-cmd">${esc(s.device)}</span>
          <span class="dep-sub"><span class="kv"><b>Layers</b> <span class="num">${s.start}–${s.end - 1}</span></span>${s.ms ? `<span class="kv"><b>Measured</b> <span class="num">${fmt(s.ms, 1)}<i>ms a token</i></span></span>` : ""}</span>
          <span class="mesh-layers"><span style="left:${(s.start / layers) * 100}%;width:${((s.end - s.start) / layers) * 100}%"></span></span>
        </span>
        <span class="pill">${s.end - s.start} layers</span>
      </div>`).join("");
    const sum = `<div class="mesh-sum">About ${fmt(p.token_ms)} ms a token for one conversation (${fmt(p.network_ms)} ms of it network); ${fmt(p.tokens_per_s, 1)} tokens/s with every device busy.</div>`;
    return head + stages + sum;
  }).join("") + (unused.length ? `<div class="mesh-sum">Not needed: ${unused.map(esc).join(", ")}</div>` : "");
}

function renderRunning(s) {
  running = s.running ? s : null;
  $("mesh-running").classList.toggle("hidden", !running);
  $("mesh-try").classList.toggle("hidden", !running);
  $("mesh-live").classList.toggle("hidden", !running);
  $("mesh-run").classList.toggle("hidden", !!running);
  $("mesh-measure").classList.toggle("hidden", !!running);
  $("mesh-model").disabled = !!running;
  if (!running) return;
  $("mesh-name").textContent = s.model;
  $("mesh-url").innerHTML = `<span class="mono code">${esc(s.url)}</span><button data-copy="${esc(s.url)}">${copyIcon}</button>`;
  renderPlan(s.pipelines);
}

async function measure() {
  message("");
  busy("Measuring your devices at once…");
  try {
    const r = await invoke("mesh_probe", { model: $("mesh-model").value, throughput: goal === "throughput" });
    renderDevices(r.devices);
    renderPlan(r.pipelines, r.unused);
    if (!r.pipelines.length && r.devices.some((d) => d.engine)) message(`${$("mesh-model").value} does not fit in the free memory of your devices together.`, "warn");
  } catch (e) {
    message(String(e));
  } finally {
    busy("");
  }
}

async function start() {
  const m = models.find((x) => x.name === $("mesh-model").value);
  if (m && !m.runnable) {
    message(`${m.name} is published only as split files, which the engine does not read yet. You can still plan it.`, "warn");
    return;
  }
  message("");
  busy("Starting…");
  try {
    renderRunning(await invoke("mesh_start", { model: $("mesh-model").value, throughput: goal === "throughput", port: PORT }));
  } catch (e) {
    message(String(e));
  } finally {
    busy("");
  }
}

listen("mesh-phase", (e) => {
  if (!$("mesh-phase").classList.contains("hidden")) $("mesh-phase-text").textContent = e.payload;
});

$("mesh-measure").addEventListener("click", measure);
$("mesh-run").addEventListener("click", start);
$("mesh-stop").addEventListener("click", async () => {
  await invoke("mesh_stop").catch((e) => message(String(e)));
  renderRunning({ running: false });
  renderPlan([]);
});
document.querySelectorAll("[data-goal]").forEach((b) => b.addEventListener("click", () => {
  goal = b.dataset.goal;
  document.querySelectorAll("[data-goal]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
}));

let answering = false;
listen("mesh-text", (e) => {
  if (answering) $("mesh-answer").textContent += e.payload;
});
$("mesh-chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = $("mesh-prompt").value.trim();
  if (!prompt || answering) return;
  answering = true;
  $("mesh-send").disabled = true;
  $("mesh-answer").textContent = "";
  $("mesh-answer").classList.remove("hidden");
  $("mesh-stats").textContent = "Thinking…";
  try {
    const s = await invoke("mesh_chat", { prompt });
    $("mesh-stats").textContent = `${s.generated} tokens in ${s.passes} trips through the devices; first token after ${fmt(s.first_token_ms)} ms, then ${fmt(s.tokens_per_s, 1)} tokens/s.`;
  } catch (err) {
    $("mesh-stats").textContent = String(err);
  } finally {
    answering = false;
    $("mesh-send").disabled = false;
  }
});

// Stage timings while serving and the page is on screen.
setInterval(async () => {
  if (!running || document.hidden || document.querySelector('[data-page="mesh"]').classList.contains("hidden")) return;
  renderRunning(await invoke("mesh_status").catch(() => ({ running: false })));
}, 3000);

invoke("mesh_models").then((list) => {
  models = list;
  $("mesh-model").innerHTML = list.map((m) => `<option value="${esc(m.name)}">${esc(m.name)} · ${fmt(m.gb, 1)} GB${m.runnable ? "" : " (plan only)"}</option>`).join("");
  $("mesh-model").value = list.find((m) => m.name === "llama-3.1-8b") ? "llama-3.1-8b" : list[0]?.name;
}).catch(() => {});
invoke("mesh_status").then(renderRunning).catch(() => {});
})();
