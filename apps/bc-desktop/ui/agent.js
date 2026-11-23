// The Agent page: conversations with the running model, which works with tools in a
// folder and remembers past conversations. The loop lives in the Rust side
// (`src/agent.rs`, `bc_cli::agent`); this file shows it. Own scope, like mesh.js.

(() => {
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch {} },
};

// The open conversation (null: a new one, made when the first message goes out).
let current = null;
let draftFolder = store.get("agent-folder") || null;
const working = new Set();
let live = null;
let modelReady = false;

$("agent-think").checked = store.get("agent-think") === "1";
$("agent-auto").checked = store.get("agent-auto") === "1";
$("agent-think").addEventListener("change", () => store.set("agent-think", $("agent-think").checked ? "1" : "0"));
$("agent-auto").addEventListener("change", () => store.set("agent-auto", $("agent-auto").checked ? "1" : "0"));

const when = (secs) => {
  const d = new Date(secs * 1000);
  const today = new Date().toDateString() === d.toDateString();
  return today ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : d.toLocaleDateString([], { month: "short", day: "numeric" });
};

// The argument that says most about a call, shown next to its name.
function gist(args) {
  if (!args || typeof args !== "object") return "";
  for (const k of ["command", "path", "url", "query", "text", "note"]) if (args[k]) return String(args[k]);
  return JSON.stringify(args);
}

function callCard(c) {
  const id = c.id ? ` data-call="${esc(c.id)}"` : "";
  if (c.error) {
    return `<div class="a-call"${id}><div class="a-call-head"><span class="a-call-name">unreadable call</span><span class="pill bad">Error</span></div><pre>${esc(c.error)}</pre></div>`;
  }
  return `
    <div class="a-call"${id}>
      <div class="a-call-head"><span class="a-call-name">${esc(c.name)}</span><span class="a-call-arg" title="${esc(gist(c.arguments))}">${esc(gist(c.arguments))}</span><span class="pill a-state">${c.pending ? "Running" : ""}</span></div>
      <details class="a-args"><summary>Arguments</summary><pre>${esc(JSON.stringify(c.arguments, null, 2))}</pre></details>
      <details class="a-result hidden"><summary>Result</summary><pre></pre></details>
      <div class="a-approve hidden"></div>
    </div>`;
}

const DENIED = "The person did not allow this call.";

function setResult(card, ok, text) {
  if (!card) return;
  const state = card.querySelector(".a-state");
  if (state) {
    const denied = text === DENIED;
    state.className = `pill a-state ${ok ? "ok" : denied ? "" : "bad"}`;
    state.textContent = ok ? "Done" : denied ? "Denied" : "Failed";
  }
  const r = card.querySelector(".a-result");
  if (r) {
    r.classList.remove("hidden");
    r.querySelector("pre").textContent = text;
  }
  card.querySelector(".a-approve")?.classList.add("hidden");
}

function assistantHtml(m) {
  const thought = m.thought ? `<details class="a-thought"><summary>Reasoning</summary><p>${esc(m.thought)}</p></details>` : "";
  const text = m.text ? `<div class="a-msg">${esc(m.text)}</div>` : "";
  return thought + text + (m.calls || []).map(callCard).join("");
}

function add(html) {
  const log = $("agent-log");
  const near = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  log.insertAdjacentHTML("beforeend", html);
  if (near) log.scrollTop = log.scrollHeight;
}

function header(folder) {
  const f = folder || (current ? null : draftFolder);
  // The last two folders say which project; the whole path is in the tooltip.
  const parts = (f || "").split(/[\\/]/).filter(Boolean);
  $("agent-folder-text").textContent = f ? (parts.length > 2 ? "…/" : "/") + parts.slice(-2).join("/") : current ? "No folder" : "Choose a folder";
  $("agent-folder").title = f || "The folder its file tools and commands work in";
  // A conversation keeps the folder it started with.
  $("agent-folder").disabled = !!current;
}

function controls() {
  const busy = current && working.has(current);
  $("agent-send").disabled = busy || !modelReady;
  $("agent-stop").classList.toggle("hidden", !busy);
  $("agent-live").classList.toggle("hidden", working.size === 0);
}

async function renderList() {
  const list = await invoke("agent_conversations").catch(() => []);
  $("agent-list").innerHTML = list.map((c) => `
    <div class="a-conv" role="button" tabindex="0" data-conv="${esc(c.id)}" aria-current="${c.id === current}">
      <span class="a-conv-title">${esc(c.title || "New conversation")}</span>
      <span class="a-conv-when">${esc(when(c.updated))}</span>
      <button class="a-conv-del" data-del="${esc(c.id)}" title="Delete" aria-label="Delete conversation">×</button>
    </div>`).join("");
}

async function open(id) {
  current = id;
  live = null;
  $("agent-chat").classList.remove("hidden");
  $("agent-memory").classList.add("hidden");
  $("agent-stats").textContent = "";
  if (!id) {
    $("agent-title").textContent = "New conversation";
    $("agent-log").innerHTML = "";
    header(null);
  } else {
    const c = await invoke("agent_open", { conversation: id }).catch((e) => { add(`<div class="banner bad">${esc(e)}</div>`); return null; });
    if (!c || current !== id) return;
    $("agent-title").textContent = c.title || "New conversation";
    header(c.folder);
    // Tool results follow their calls in order.
    let html = "";
    const waiting = [];
    let n = 0;
    for (const m of c.messages) {
      if (m.role === "user") html += `<div class="a-msg a-user">${esc(m.text)}</div>`;
      else if (m.role === "assistant") {
        const calls = (m.calls || []).map((x) => ({ ...x, id: `old-${n++}` }));
        waiting.push(...calls.filter((x) => !x.error).map((x) => x.id));
        html += assistantHtml({ ...m, calls });
      } else if (m.role === "tool") {
        const cid = waiting.shift();
        if (cid) html = html.replace(`data-call="${cid}">`, `data-call="${cid}" data-result="${esc(m.text)}">`);
      }
    }
    $("agent-log").innerHTML = html;
    $("agent-log").querySelectorAll("[data-result]").forEach((card) => setResult(card, !card.dataset.result.startsWith("Error") && card.dataset.result !== DENIED, card.dataset.result));
    $("agent-log").querySelectorAll('.a-state:empty').forEach((s) => (s.textContent = "Not run"));
    $("agent-log").scrollTop = $("agent-log").scrollHeight;
  }
  controls();
  renderList();
}

async function send(e) {
  e.preventDefault();
  const text = $("agent-input").value.trim();
  if (!text || !modelReady || (current && working.has(current))) return;
  try {
    if (!current) {
      current = await invoke("agent_new", { folder: draftFolder });
      header(draftFolder);
    }
  } catch (err) {
    add(`<div class="banner bad">${esc(err)}</div>`);
    return;
  }
  const id = current;
  $("agent-input").value = "";
  add(`<div class="a-msg a-user">${esc(text)}</div>`);
  if ($("agent-title").textContent === "New conversation") $("agent-title").textContent = text.split("\n")[0].slice(0, 60);
  working.add(id);
  controls();
  renderList();
  try {
    await invoke("agent_send", { conversation: id, text, settings: { think: $("agent-think").checked, auto_approve: $("agent-auto").checked } });
  } catch {
    // The error came as an event too.
  } finally {
    working.delete(id);
    if (current === id) live = null;
    controls();
    renderList();
  }
}

listen("agent-event", ({ payload }) => {
  const { conversation, event: ev } = payload;
  if (conversation !== current) return;
  const log = $("agent-log");
  switch (ev.kind) {
    case "text":
      if (!live) {
        add(`<div class="a-msg a-live"></div>`);
        live = log.lastElementChild;
      }
      live.textContent += ev.text;
      if (log.scrollHeight - log.scrollTop - log.clientHeight < 120) log.scrollTop = log.scrollHeight;
      break;
    case "reply": {
      const html = assistantHtml({ ...ev, calls: ev.calls.map((c) => ({ ...c, pending: true })) });
      if (live) {
        live.insertAdjacentHTML("afterend", html);
        live.remove();
        live = null;
      } else add(html);
      break;
    }
    case "approve": {
      const card = log.querySelector(`[data-call="${CSS.escape(ev.id)}"]`);
      if (!card) break;
      card.querySelector(".a-state").textContent = "Waiting";
      const what = { write: "It wants to change files.", run: "It wants to run a command.", net: "It wants to read a web page." }[ev.risk] || "";
      const box = card.querySelector(".a-approve");
      box.innerHTML = `<span class="hint">${what}</span><button data-decide="deny">Deny</button><button data-decide="allow_always">Always allow ${esc(ev.name)}</button><button class="primary" data-decide="allow">Allow</button>`;
      box.classList.remove("hidden");
      card.querySelector(".a-args").open = true;
      box.querySelector(".primary").focus();
      break;
    }
    case "result":
      setResult(log.querySelector(`[data-call="${CSS.escape(ev.id)}"]`), ev.ok, ev.text);
      break;
    case "done": {
      const reused = ev.prompt_tokens ? Math.round((100 * ev.cached_tokens) / ev.prompt_tokens) : 0;
      const room = ev.context ? ` · ${Math.round((100 * ev.context_used) / ev.context)}% of the context used` : "";
      $("agent-stats").textContent = `${ev.steps} ${ev.steps === 1 ? "step" : "steps"} · ${ev.tokens_per_s.toFixed(1)} tokens/s · ${reused}% of the prompt already on the devices${room}`;
      break;
    }
    case "error":
      if (live) { live.remove(); live = null; }
      add(`<div class="banner bad">${esc(ev.message)}</div>`);
      break;
  }
});

// A reminder the agent set, now due: shown here too, in case notifications are off.
listen("agent-reminder", ({ payload }) => {
  const box = $("agent-reminders");
  box.insertAdjacentHTML("beforeend", `<div class="banner warn agent-reminder"><span><b>Reminder:</b> ${esc(payload.text)}</span><button class="link" data-dismiss>Dismiss</button></div>`);
});
$("agent-reminders").addEventListener("click", (e) => {
  if (e.target.closest("[data-dismiss]")) e.target.closest(".agent-reminder").remove();
});

$("agent-log").addEventListener("click", (e) => {
  const b = e.target.closest("[data-decide]");
  if (!b) return;
  const card = b.closest("[data-call]");
  card.querySelector(".a-approve").classList.add("hidden");
  card.querySelector(".a-state").textContent = b.dataset.decide === "deny" ? "" : "Running";
  invoke("agent_approve", { call: card.dataset.call, decision: b.dataset.decide });
});

$("agent-list").addEventListener("click", async (e) => {
  const del = e.target.closest("[data-del]");
  if (del) {
    e.stopPropagation();
    if (!(await ask("Delete this conversation? The agent can no longer recall it.", "Delete", true))) return;
    await invoke("agent_delete", { conversation: del.dataset.del }).catch(() => {});
    if (current === del.dataset.del) open(null);
    else renderList();
    return;
  }
  const row = e.target.closest("[data-conv]");
  if (row) open(row.dataset.conv);
});
$("agent-list").addEventListener("keydown", (e) => {
  const row = e.target.closest("[data-conv]");
  if (row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); open(row.dataset.conv); }
});

$("agent-new").addEventListener("click", () => open(null));
$("agent-form").addEventListener("submit", send);
$("agent-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) $("agent-form").requestSubmit();
});
$("agent-stop").addEventListener("click", () => current && invoke("agent_stop", { conversation: current }));
$("agent-folder").addEventListener("click", async () => {
  const picked = await invoke("pick_folder").catch(() => null);
  if (!picked) return;
  draftFolder = picked;
  store.set("agent-folder", picked);
  header(null);
});
$("agent-to-mesh").addEventListener("click", () => window.lendUI.showPage("mesh"));

$("agent-memory-open").addEventListener("click", async () => {
  $("agent-notes").value = await invoke("memory_get").catch(() => "");
  $("agent-notes-msg").textContent = "";
  const server = await invoke("model_server_get").catch(() => ({}));
  $("agent-server-url").value = server.url || "";
  $("agent-server-model").value = server.model || "";
  $("agent-server-context").value = server.context || "";
  $("agent-server-msg").textContent = "";
  $("agent-chat").classList.add("hidden");
  $("agent-memory").classList.remove("hidden");
});
$("agent-memory-close").addEventListener("click", () => {
  $("agent-memory").classList.add("hidden");
  $("agent-chat").classList.remove("hidden");
});
$("agent-notes-save").addEventListener("click", async () => {
  try {
    await invoke("memory_set", { text: $("agent-notes").value });
    $("agent-notes-msg").textContent = "Saved.";
  } catch (e) {
    $("agent-notes-msg").textContent = String(e);
  }
});
$("agent-server-save").addEventListener("click", async () => {
  const server = { url: $("agent-server-url").value.trim(), model: $("agent-server-model").value.trim(), context: Number($("agent-server-context").value) || 0 };
  try {
    await invoke("model_server_set", { server });
    $("agent-server-msg").textContent = "Saved.";
    checkModel();
  } catch (e) {
    $("agent-server-msg").textContent = String(e);
  }
});
$("agent-forget").addEventListener("click", async () => {
  if (!(await ask("Delete every conversation on this computer? This cannot be undone.", "Delete all", true))) return;
  await invoke("agent_delete", { conversation: null }).catch(() => {});
  open(null);
});

// Whether a model runs, while the page is on screen.
async function checkModel() {
  modelReady = !!(await invoke("agent_model").catch(() => ""));
  $("agent-nomodel").classList.toggle("hidden", modelReady);
  controls();
}
setInterval(() => {
  if (!document.hidden && !document.querySelector('[data-page="agent"]').classList.contains("hidden")) checkModel();
}, 3000);

checkModel();
open(null);
})();
