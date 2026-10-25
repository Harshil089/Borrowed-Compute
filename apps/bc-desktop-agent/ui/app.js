// Borrowed Compute Agent window. All state lives in the Rust side; this file renders it.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

let status = null;
let account = null;
let pairings = [];
let jobs = [];
let verificationUri = null;
let signingIn = false;

function showError(message) {
  const box = $("error");
  box.textContent = message || "";
  box.classList.toggle("hidden", !message);
}

function ago(ms) {
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ms).toLocaleDateString();
}

function clientName(id) {
  const p = pairings.find((p) => p.client_id === id);
  return p ? p.client_name : `${id.slice(0, 10)}…`;
}

function renderStatus() {
  const pill = $("status");
  let cls = "", label = "Starting";
  if (!status) {
    label = "Starting";
  } else if (status.start_error) {
    cls = "bad"; label = "Not running";
  } else if (!status.serving) {
    label = "Starting";
  } else if (!status.signed_in) {
    cls = "warn"; label = "Signed out";
  } else if (status.cloud.ok === false) {
    cls = "warn"; label = "Cloud unreachable";
  } else if (status.cloud.ok === true) {
    cls = "ok"; label = "Ready for jobs";
  } else {
    label = "Connecting";
  }
  pill.className = `pill ${cls}`;
  pill.innerHTML = `<span class="dot ${cls}"></span><span>${label}</span>`;
  pill.title = status?.cloud?.error || status?.start_error || "";
}

function render() {
  renderStatus();
  const signedIn = !!status?.signed_in;
  $("signin").classList.toggle("hidden", signedIn);
  $("home").classList.toggle("hidden", !signedIn);
  $("footer").classList.toggle("hidden", !signedIn);
  $("account").textContent = signedIn && account
    ? `${account.login}${account.device_name ? ` · ${account.device_name}` : ""}`
    : "";
  $("device-id").textContent = status?.device_id ? `Device ${status.device_id.slice(0, 16)}…` : "";
  if (status?.start_error) showError(`The agent could not start: ${status.start_error}`);
  if (!signedIn) return;

  const pending = pairings.filter((p) => p.status === "pending");
  $("requests").innerHTML = pending.length
    ? pending.map((p) => `
        <div class="item request">
          <div class="main">
            <span class="name">${esc(p.client_name)}</span>
            <span class="sub">Approve only if ${esc(p.client_name)} shows this code</span>
          </div>
          <span class="code">${esc(p.code)}</span>
          <div class="actions">
            <button class="primary" data-approve="${p.id}" data-code="${esc(p.code)}">Approve</button>
          </div>
        </div>`).join("")
    : `<div class="empty">No requests. On your PC, choose this Mac and click Pair.</div>`;

  const approved = pairings.filter((p) => p.status === "approved");
  $("paired").innerHTML = approved.length
    ? approved.map((p) => `
        <div class="item">
          <span class="dot ok"></span>
          <div class="main">
            <span class="name">${esc(p.client_name)}</span>
            <span class="sub mono">${esc(p.client_id.slice(0, 16))}…</span>
          </div>
          <button class="danger" data-unpair="${p.id}" data-name="${esc(p.client_name)}">Remove</button>
        </div>`).join("")
    : `<div class="empty">No paired devices yet.</div>`;

  $("jobs").innerHTML = jobs.length
    ? jobs.map((j) => {
        const [cls, label] = jobState(j);
        const what = j.rejected ? "Rejected: device is not paired" : esc(j.argv.join(" "));
        return `
          <div class="item">
            <span class="dot ${cls}"></span>
            <div class="main">
              <span class="name ${j.rejected ? "" : "argv"}">${what}</span>
              <span class="sub">${esc(clientName(j.client))} · ${ago(j.started_ms)}</span>
            </div>
            <span class="pill ${cls}">${label}</span>
          </div>`;
      }).join("")
    : `<div class="empty">Jobs from paired devices appear here.</div>`;
}

function jobState(j) {
  if (j.rejected) return ["bad", "Rejected"];
  if (!j.end) return ["warn", "Running"];
  switch (j.end.kind) {
    case "exited":
      if (j.end.timed_out) return ["bad", "Timed out"];
      return j.end.code === 0 ? ["ok", "Exit 0"] : ["bad", `Exit ${j.end.code ?? "?"}`];
    case "failed": return ["bad", "Failed"];
    case "cancelled": return ["", "Cancelled"];
    default: return ["", "Disconnected"];
  }
}

async function refresh() {
  try {
    status = await invoke("get_status");
    if (status.signed_in) {
      if (!account) account = await invoke("get_account").catch(() => null);
      pairings = await invoke("list_pairings");
      jobs = await invoke("recent_jobs");
      showError(status.start_error ? `The agent could not start: ${status.start_error}` : "");
    }
  } catch (e) {
    showError(String(e));
  }
  render();
}

$("signin-btn").addEventListener("click", async () => {
  if (signingIn) return;
  signingIn = true;
  showError("");
  $("signin-btn").disabled = true;
  try {
    account = await invoke("login", { name: $("mac-name").value });
    $("signin-code").classList.add("hidden");
  } catch (e) {
    showError(`Sign-in failed: ${e}`);
    $("signin-code").classList.add("hidden");
  } finally {
    signingIn = false;
    $("signin-btn").disabled = false;
    await refresh();
  }
});

$("reopen").addEventListener("click", () => {
  if (verificationUri) invoke("open_url", { url: verificationUri }).catch((e) => showError(String(e)));
});

$("signout").addEventListener("click", async () => {
  if (!confirm("Sign out? Paired devices will not be able to send jobs until you sign in again.")) return;
  await invoke("logout").catch((e) => showError(String(e)));
  account = null; pairings = []; jobs = [];
  await refresh();
});

document.addEventListener("click", async (event) => {
  const approve = event.target.closest("[data-approve]");
  const remove = event.target.closest("[data-unpair]");
  try {
    if (approve) {
      approve.disabled = true;
      await invoke("approve", { id: Number(approve.dataset.approve), code: approve.dataset.code });
      await refresh();
    } else if (remove) {
      if (!confirm(`Remove ${remove.dataset.name}? It will stop being able to send jobs within 15 seconds.`)) return;
      remove.disabled = true;
      await invoke("unpair", { id: Number(remove.dataset.unpair) });
      await refresh();
    }
  } catch (e) {
    showError(String(e));
    await refresh();
  }
});

listen("login-code", (event) => {
  verificationUri = event.payload.verification_uri;
  $("user-code").textContent = event.payload.user_code;
  $("signin-code").classList.remove("hidden");
});
listen("agent-event", refresh);
listen("status-changed", refresh);

// Pairing requests come from the cloud, so poll while the window is open.
setInterval(() => { if (document.visibilityState === "visible") refresh(); }, 5000);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refresh(); });
refresh();
