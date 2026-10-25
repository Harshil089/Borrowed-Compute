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
let tailscale = null;
let sharing = false;
let shareMessage = null;
let approvalUrl = null;

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
  } else if (status.host.hosting && status.host.error) {
    cls = "bad"; label = "Cloud stopped";
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
  else if (status?.host.hosting && status.host.error) showError(`The built-in cloud is not running: ${status.host.error}`);
  if (!signedIn) return;

  renderConnect();

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

const copyIcon = "Copy";

function renderConnect() {
  const box = $("connect");
  if (!status.host.hosting) {
    box.innerHTML = `
      <p>This Mac uses the cloud at</p>
      <div class="address"><span>${esc(status.host.url)}</span></div>
      <p class="muted">Enter the same address in Borrowed Compute on your PC.</p>`;
    return;
  }
  const t = tailscale;
  let body;
  if (!t) {
    body = `<div class="row"><span class="spinner"></span><span class="muted">Checking Tailscale…</span></div>`;
  } else if (t.state === "sharing") {
    body = `
      <p>Your PC reaches this Mac's cloud at:</p>
      <div class="address"><span id="share-url">${esc(t.url)}</span><button data-copy="${esc(t.url)}">${copyIcon}</button></div>
      <p class="muted">In Borrowed Compute on your PC, enter this address. Only devices in your Tailscale network can open it.</p>
      <div class="actions"><button class="link" data-ts="stop">Stop sharing</button></div>`;
  } else if (t.state === "ready") {
    body = `
      <p>Tailscale is connected as <span class="mono">${esc(t.host)}</span>. Share this Mac's cloud with your other devices:</p>
      <div class="actions"><button class="primary" data-ts="share" ${sharing ? "disabled" : ""}>${sharing ? "Sharing…" : "Share with my devices"}</button></div>`;
  } else if (t.state === "https_disabled") {
    body = `
      <p>Tailscale is connected, but your tailnet has not turned on HTTPS certificates, which sharing needs.</p>
      <ol class="steps"><li>Open the Tailscale admin page, DNS tab.</li><li>Turn on MagicDNS and HTTPS Certificates.</li><li>Come back and click Check again.</li></ol>
      <div class="actions">
        <button class="primary" data-open="https://login.tailscale.com/admin/dns">Open Tailscale DNS settings</button>
        <button data-ts="check">Check again</button>
      </div>`;
  } else if (t.state === "not_running") {
    body = `
      <p>Tailscale is installed but not connected. Open Tailscale from the menu bar and sign in.</p>
      <div class="actions"><button data-ts="check">Check again</button></div>`;
  } else if (t.state === "not_installed") {
    body = `
      <p>Your PC reaches this Mac through Tailscale, a free private network between your own devices.</p>
      <ol class="steps">
        <li>Install Tailscale on this Mac and on your PC.</li>
        <li>Sign in to Tailscale on both with the same account.</li>
        <li>Come back here and click Check again.</li>
      </ol>
      <div class="actions">
        <button class="primary" data-open="https://tailscale.com/download">Download Tailscale</button>
        <button data-ts="check">Check again</button>
      </div>`;
  } else {
    body = `<p>Tailscale reported a problem: ${esc(t.message)}</p><div class="actions"><button data-ts="check">Check again</button></div>`;
  }
  if (approvalUrl) {
    body += `
      <div class="banner warn">Tailscale needs you to allow sharing for your tailnet once.</div>
      <div class="actions">
        <button class="primary" data-open="${esc(approvalUrl)}">Allow in Tailscale</button>
        <button data-ts="share">Try again</button>
      </div>`;
  }
  if (shareMessage) body += `<div class="banner bad">${esc(shareMessage)}</div>`;
  box.innerHTML = body;
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
      if (!account) {
        try {
          account = await invoke("get_account");
        } catch (e) {
          // The session belonged to another cloud; the app signed out, so show sign-in.
          if (String(e) === "signed-out") { status = await invoke("get_status"); render(); return; }
        }
      }
      if (status.host.hosting) tailscale = await invoke("tailscale_status");
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
  const copy = event.target.closest("[data-copy]");
  const open = event.target.closest("[data-open]");
  const ts = event.target.closest("[data-ts]");
  if (copy) {
    await navigator.clipboard.writeText(copy.dataset.copy).catch(() => {});
    copy.textContent = "Copied";
    setTimeout(() => { copy.textContent = copyIcon; }, 1500);
    return;
  }
  if (open) {
    invoke("open_url", { url: open.dataset.open }).catch((e) => showError(String(e)));
    return;
  }
  if (ts) {
    shareMessage = null;
    if (ts.dataset.ts === "share") {
      sharing = true; approvalUrl = null; renderConnect();
      try {
        const r = await invoke("tailscale_share");
        if (r.result === "needs_approval") approvalUrl = r.url;
      } catch (e) {
        shareMessage = String(e);
      }
      sharing = false;
    } else if (ts.dataset.ts === "stop") {
      if (!confirm("Stop sharing? Your PC will not reach this Mac's cloud until you share again.")) return;
      await invoke("tailscale_stop").catch((e) => { shareMessage = String(e); });
    }
    tailscale = null; renderConnect();
    await refresh();
    return;
  }
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
