// Borrowed Compute window. The Rust side does the work; this file renders state and
// forwards clicks.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Per-viewer conveniences only; the device identity and session live in Rust.
const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage off */ } },
};

let status = null;
let devices = null;
let selectedMac = null;
let running = false;
let verificationUri = null;
let specCommand = null;

function showError(message) {
  $("error").textContent = message || "";
  $("error").classList.toggle("hidden", !message);
}

function ago(secs) {
  if (secs == null) return "never seen";
  if (secs < 120) return `seen ${secs}s ago`;
  if (secs < 7200) return `seen ${Math.floor(secs / 60)} min ago`;
  if (secs < 172800) return `seen ${Math.floor(secs / 3600)} h ago`;
  return `seen ${Math.floor(secs / 86400)} days ago`;
}

async function loadStatus() {
  status = await invoke("get_status");
  $("signin-view").classList.toggle("hidden", status.signed_in);
  $("app-view").classList.toggle("hidden", !status.signed_in);
  if (status.signed_in) {
    selectedMac = selectedMac || status.default_mac;
    invoke("get_account")
      .then((a) => { $("account").textContent = `${a.login}${a.device_name ? ` · ${a.device_name}` : ""}`; })
      .catch(() => {});
    await loadDevices();
  }
}

async function loadDevices() {
  try {
    devices = await invoke("list_devices");
    showError("");
  } catch (e) {
    showError(`Could not reach the cloud: ${e}`);
  }
  renderMacs();
}

function pairingFor(mac) {
  return devices?.pairings.find((p) => p.mac_id === mac.endpoint_id);
}

function renderMacs() {
  const macs = (devices?.devices || []).filter((d) => d.kind === "mac" && d.endpoint_id !== devices.me);
  if (!macs.length) {
    $("macs").innerHTML = `<div class="empty">No Macs on this account yet. Install Borrowed Compute Agent on your Mac and sign in there.</div>`;
    return;
  }
  if (!macs.some((m) => m.name === selectedMac)) {
    const paired = macs.find((m) => pairingFor(m)?.status === "approved");
    selectedMac = (paired || macs[0]).name;
  }
  $("macs").innerHTML = macs.map((m) => {
    const pairing = pairingFor(m);
    const paired = pairing?.status === "approved";
    const dot = m.online ? "ok" : "";
    const state = m.online ? "Online" : ago(m.last_seen_secs_ago);
    const pairLine = paired
      ? `<span class="pill ok">Paired</span>`
      : `<button data-pair="${esc(m.name)}">${pairing ? "Show pairing code" : "Pair"}</button>`;
    return `
      <div class="mac ${m.name === selectedMac ? "selected" : ""}" data-mac="${esc(m.name)}" tabindex="0">
        <div class="top-line"><span class="dot ${dot}"></span><span class="name">${esc(m.name)}</span></div>
        <div class="sub">${state}</div>
        <div class="pair-line">${pairLine}</div>
      </div>`;
  }).join("");
  updateRunButton();
}

function updateRunButton() {
  const mac = devices?.devices.find((d) => d.name === selectedMac);
  const paired = mac && pairingFor(mac)?.status === "approved";
  $("run").disabled = running || !selectedMac || !$("folder").value.trim();
  $("run").textContent = selectedMac ? `Run on ${selectedMac}` : "Run on Mac";
  if (selectedMac && mac && !paired) {
    $("run").title = `Pair with ${selectedMac} first`;
  } else {
    $("run").title = "";
  }
}

async function selectMac(name) {
  selectedMac = name;
  await invoke("choose_mac", { mac: name }).catch(() => {});
  renderMacs();
}

// ---- Sign-in ----

$("signin-btn").addEventListener("click", async () => {
  $("signin-btn").disabled = true;
  $("signin-error").classList.add("hidden");
  try {
    await invoke("login", { name: $("pc-name").value });
    await loadStatus();
  } catch (e) {
    $("signin-error").textContent = `Sign-in failed: ${e}`;
    $("signin-error").classList.remove("hidden");
  } finally {
    $("signin-btn").disabled = false;
    $("signin-code").classList.add("hidden");
  }
});

$("reopen").addEventListener("click", () => {
  if (verificationUri) invoke("open_url", { url: verificationUri });
});

listen("login-code", (event) => {
  verificationUri = event.payload.verification_uri;
  $("user-code").textContent = event.payload.user_code;
  $("signin-code").classList.remove("hidden");
});

$("signout").addEventListener("click", async () => {
  if (!confirm("Sign out of Borrowed Compute on this computer?")) return;
  await invoke("logout");
  devices = null;
  await loadStatus();
});

// ---- Macs and pairing ----

$("refresh").addEventListener("click", loadDevices);

$("macs").addEventListener("click", (event) => {
  const pair = event.target.closest("[data-pair]");
  if (pair) { startPairing(pair.dataset.pair); return; }
  const mac = event.target.closest("[data-mac]");
  if (mac) selectMac(mac.dataset.mac);
});
$("macs").addEventListener("keydown", (event) => {
  const mac = event.target.closest("[data-mac]");
  if (mac && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); selectMac(mac.dataset.mac); }
});

let pairingOpen = false;

async function startPairing(macName) {
  $("pair-mac").textContent = macName;
  $("pair-code").textContent = "······";
  $("pair-msg").className = "banner hidden";
  $("pair-wait").classList.remove("hidden");
  $("pair-dialog").classList.remove("hidden");
  pairingOpen = true;
  try {
    const request = await invoke("request_pairing", { mac: macName });
    if (request.status === "approved") return pairDone(true, `Already paired with ${macName}.`);
    $("pair-code").textContent = request.code;
    const approved = await invoke("wait_pairing", { id: request.id, expiresInSecs: request.expires_in_secs ?? 600 });
    pairDone(approved, approved ? `Paired with ${macName}. You can run jobs on it now.` : "The code expired. Close this and click Pair again.");
  } catch (e) {
    pairDone(false, String(e));
  }
}

function pairDone(ok, message) {
  if (!pairingOpen) return;
  $("pair-wait").classList.add("hidden");
  $("pair-msg").className = `banner ${ok ? "ok" : "bad"}`;
  $("pair-msg").textContent = message;
  loadDevices();
}

$("pair-close").addEventListener("click", () => {
  pairingOpen = false;
  $("pair-dialog").classList.add("hidden");
  loadDevices();
});

// ---- Project and command ----

async function setFolder(path) {
  $("folder").value = path || "";
  store.set("folder", path || "");
  specCommand = null;
  $("hint").textContent = "";
  if (path) {
    const info = await invoke("project_info", { folder: path });
    if (info.spec_error) {
      $("hint").textContent = `bc.toml has a problem: ${info.spec_error}`;
    } else if (info.has_spec) {
      specCommand = info.command;
      $("hint").textContent = "Using bc.toml from this folder. A command typed here replaces its command.";
      if (!$("command").value.trim()) $("command").placeholder = info.command;
    } else {
      $("command").placeholder = "cargo test";
    }
  }
  updateRunButton();
}

$("choose").addEventListener("click", async () => {
  const picked = await invoke("pick_folder").catch((e) => showError(String(e)));
  if (picked) setFolder(picked);
});
$("folder").addEventListener("change", () => setFolder($("folder").value.trim()));
$("folder").addEventListener("input", updateRunButton);
$("command").addEventListener("input", () => store.set("command", $("command").value));
$("command").addEventListener("keydown", (e) => { if (e.key === "Enter" && !$("run").disabled) runJob(); });

// ---- Running ----

const consoleEl = $("console");
let atBottom = true;
consoleEl.addEventListener("scroll", () => {
  atBottom = consoleEl.scrollHeight - consoleEl.scrollTop - consoleEl.clientHeight < 24;
});

function appendOutput(stream, text) {
  const last = consoleEl.lastChild;
  if (last && last.dataset.stream === stream) {
    last.textContent += text;
  } else {
    const span = document.createElement("span");
    span.dataset.stream = stream;
    if (stream !== "stdout") span.className = stream;
    span.textContent = text;
    consoleEl.appendChild(span);
  }
  if (atBottom) consoleEl.scrollTop = consoleEl.scrollHeight;
}

let sawOutput = false;
listen("job-output", (event) => {
  // The first output means the upload is over and the job is running on the Mac.
  if (running && !sawOutput && event.payload.stream !== "notice") {
    sawOutput = true;
    setPhase(`Running on ${selectedMac}…`, true);
  }
  appendOutput(event.payload.stream, event.payload.text);
});
listen("job-phase", (event) => setPhase(event.payload, true));

function setPhase(text, busy) {
  $("phase").innerHTML = `${busy ? '<span class="spinner"></span>' : ""}<span>${esc(text)}</span>`;
}

function ms(v) { return v == null ? "–" : v < 1000 ? `${v} ms` : `${(v / 1000).toFixed(1)} s`; }

async function runJob() {
  if (running) return;
  const command = $("command").value.trim();
  if (!command && !specCommand) { showError("Type a command to run, or choose a folder with a bc.toml."); return; }
  showError("");
  running = true;
  sawOutput = false;
  consoleEl.textContent = "";
  atBottom = true;
  $("result").classList.add("hidden");
  $("run").classList.add("hidden");
  $("cancel").classList.remove("hidden");
  $("cancel").disabled = false;
  setPhase("Starting…", true);
  try {
    const r = await invoke("run_job", {
      mac: selectedMac,
      folder: $("folder").value.trim(),
      command,
      relayOnly: $("relay-only").checked,
    });
    const cls = r.outcome === "success" ? "ok" : r.outcome === "cancelled" ? "" : "bad";
    const t = r.timings;
    const path = t.via_relay ? "relay" : "direct";
    $("result").innerHTML = `
      <span class="pill ${cls}"><span class="dot ${cls}"></span>${esc(r.message)}</span>
      <span class="timings">connect ${ms(t.connect_ms)} · upload ${ms(t.upload_ms)} · first output ${ms(t.first_log_ms)} · total ${ms(t.total_ms)} · ${path}</span>`;
    $("result").classList.remove("hidden");
    setPhase(r.outcome === "rejected" ? "Not paired with this Mac. Click Pair next to it." : "Done.", false);
    if (r.outcome === "rejected") loadDevices();
  } catch (e) {
    setPhase("Could not run the job.", false);
    showError(String(e));
  } finally {
    running = false;
    $("run").classList.remove("hidden");
    $("cancel").classList.add("hidden");
    updateRunButton();
  }
}

$("run").addEventListener("click", runJob);
$("cancel").addEventListener("click", () => { $("cancel").disabled = true; invoke("cancel_job"); });
$("clear").addEventListener("click", () => {
  consoleEl.textContent = "";
  $("result").classList.add("hidden");
  setPhase("Output appears here as the Mac produces it.", false);
});

// ---- Start ----

$("command").value = store.get("command") || "";
loadStatus()
  .then(() => setFolder(store.get("folder") || ""))
  .catch((e) => showError(String(e)));
// Keep online states fresh without hammering the cloud.
setInterval(() => { if (status?.signed_in && !running && document.visibilityState === "visible") loadDevices(); }, 15000);
