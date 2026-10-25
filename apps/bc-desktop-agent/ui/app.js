// Borrowed Compute Agent window. All state lives in the Rust side; this file renders it.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// The web view has no native confirm() dialog, so ask inside the window instead.
function ask(message, okLabel = "OK", danger = false) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "ask-overlay";
    overlay.setAttribute("role", "alertdialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.innerHTML = `
      <div class="ask-box">
        <p></p>
        <div class="ask-actions">
          <button data-ask="no">Cancel</button>
          <button data-ask="yes" class="primary ${danger ? "danger" : ""}"></button>
        </div>
      </div>`;
    overlay.querySelector("p").textContent = message;
    overlay.querySelector('[data-ask="yes"]').textContent = okLabel;
    const done = (answer) => {
      overlay.remove();
      document.removeEventListener("keydown", onKey, true);
      resolve(answer);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); done(false); }
    };
    overlay.addEventListener("click", (e) => {
      const button = e.target.closest("[data-ask]");
      if (button) { e.stopPropagation(); done(button.dataset.ask === "yes"); }
    });
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(overlay);
    overlay.querySelector('[data-ask="no"]').focus();
  });
}

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
// Requests already on screen, so only new ones animate in.
const seenRequests = new Set();

// ---- Arrivals: the latest run, flown live ----

const ARR_LEGS = ["upload", "run", "return"];
let arrivalsKey = null;
let liveSince = null;

function setArrivalLegs(at, failed) {
  const legs = $("arr-legs");
  legs.querySelectorAll("li").forEach((li, i) => {
    li.classList.toggle("done", i < at);
    li.classList.toggle("active", i === at);
  });
  legs.classList.toggle("idle", at < 0);
  legs.classList.toggle("failed", failed);
  // The jet sits at the start of the active leg, or at this Mac once the run is home.
  const p = at < 0 ? 0 : at >= ARR_LEGS.length ? 1 : (at + 0.5) / ARR_LEGS.length;
  legs.style.setProperty("--p", String(p));
}

function renderArrivals() {
  const job = jobs.find((j) => !j.rejected);
  const me = account?.device_name || "This Mac";
  $("arr-to").textContent = deviceCode(me);
  $("arr-to-name").textContent = me;
  $("arr-date").textContent = new Date().toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const key = job ? `${job.job_id}:${job.end ? job.end.kind : "live"}` : "none";
  if (key === arrivalsKey) return;
  const firstRender = arrivalsKey === null;
  arrivalsKey = key;

  if (!job) {
    $("arr-tab").textContent = "Waiting";
    $("arr-from").textContent = "PC";
    $("arr-from-name").textContent = "Your PC";
    $("arr-cmd").className = "arr-cmd";
    $("arr-cmd").textContent = "No runs yet. When your PC sends one, it lands here live.";
    $("arr-clock").textContent = "";
    $("arr-stub").classList.add("hidden");
    liveSince = null;
    setArrivalLegs(-1, false);
    return;
  }
  const from = clientName(job.client);
  $("arr-from").textContent = deviceCode(from);
  $("arr-from-name").textContent = from;
  $("arr-cmd").className = "arr-cmd argv";
  $("arr-cmd").textContent = job.argv.join(" ");

  if (!job.end) {
    // The agent reports a run once its upload is in, so the upload leg is already flown.
    $("arr-tab").textContent = "In flight";
    $("arr-stub").classList.add("hidden");
    liveSince = job.started_ms;
    tickClock();
    setArrivalLegs(1, false);
    return;
  }

  liveSince = null;
  const [cls, word] = jobState(job);
  $("arr-tab").textContent = word;
  $("arr-clock").textContent = job.finished_ms ? ago(job.finished_ms) : "";
  const landed = cls === "ok";
  setArrivalLegs(landed ? ARR_LEGS.length : 1, !landed);
  const u = job.end.usage;
  const cores = u && u.wall_ms > 0 ? ((u.user_ms + u.system_ms) / u.wall_ms).toFixed(1) : null;
  const stampCls = landed ? "ok" : word === "Void" ? "void" : "bad";
  const stub = $("arr-stub");
  stub.innerHTML = `
    <div class="arr-time"><span class="label">Ran here</span><strong>${u ? fig(u.wall_ms) : "–"}</strong></div>
    <div class="arr-figs">
      ${u ? `<b>CPU</b><span>${fig(u.user_ms + u.system_ms)}</span>
      <b>Cores</b><span>${cores}</span>
      <b>Memory</b><span>${Math.round(u.max_rss_bytes / 1048576)}<i>MB</i></span>` : `<b>Usage</b><span>not reported</span>`}
    </div>
    <span class="stamp ${stampCls}">${esc(word)}</span>`;
  stub.classList.remove("hidden");
  // Tear the copy off only for a run that lands while the window is open.
  stub.classList.toggle("arrived", !firstRender);
}

function tickClock() {
  if (liveSince == null) return;
  const s = Math.max(0, Math.round((Date.now() - liveSince) / 1000));
  $("arr-clock").innerHTML = s < 60 ? `${s}<i>s</i>` : `${Math.floor(s / 60)}<i>m</i> ${s % 60}<i>s</i>`;
}
setInterval(tickClock, 1000);

// Writes HTML only when it differs, so a poll never resets focus or replays animation.
function setHtml(id, html) {
  const el = $(id);
  if (el.dataset.html === html) return;
  el.dataset.html = html;
  el.innerHTML = html;
}

// Line breaks only after "//" and dots, so the address never splits mid-word.
function breakableUrl(url) {
  return esc(url).replace(/(\/\/|\.)/g, "$1<wbr>");
}

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

// Three-letter codes, like airport codes: initials of a multi-word name, else its first letters.
function deviceCode(name) {
  const words = String(name || "").split(/[^A-Za-z0-9]+/).filter(Boolean);
  let code;
  if (words.length >= 3) code = words.slice(0, 3).map((w) => w[0]).join("");
  else if (words.length === 2) code = words[0][0] + words[1].slice(0, 2);
  else code = (words[0] || "MAC").slice(0, 3);
  return code.toUpperCase().padEnd(3, "X");
}

function clock(ms) {
  const d = new Date(ms);
  const sameDay = d.toDateString() === new Date().toDateString();
  return sameDay
    ? d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false })
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// A measured figure: the number in carbon, the unit set small beside it.
function fig(v) {
  return v < 1000 ? `${v}<i>ms</i>` : `${(v / 1000).toFixed(1)}<i>s</i>`;
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
    cls = "bad"; label = "Closed";
  } else if (status.host.hosting && status.host.error) {
    cls = "bad"; label = "Cloud stopped";
  } else if (!status.serving) {
    label = "Starting";
  } else if (!status.signed_in) {
    cls = "warn"; label = "Signed out";
  } else if (status.cloud.ok === false) {
    cls = "warn"; label = "Cloud unreachable";
  } else if (status.cloud.ok === true) {
    cls = "ok"; label = "Open for runs";
  } else {
    label = "Connecting";
  }
  pill.className = `pill ${cls}`;
  pill.innerHTML = `<span>${label}</span>`;
  pill.title = status?.cloud?.error || status?.start_error || "";
}

function render() {
  renderStatus();
  const signedIn = !!status?.signed_in;
  $("signin").classList.toggle("hidden", signedIn);
  $("home").classList.toggle("hidden", !signedIn);
  $("footer").classList.toggle("hidden", !signedIn);
  $("account").textContent = signedIn && account ? `Signed in as ${account.login}` : "";
  $("station-name").textContent = account?.device_name || "This Mac";
  $("station-code").textContent = deviceCode(account?.device_name || "Mac");
  $("device-id").textContent = status?.device_id ? `Device ${status.device_id.slice(0, 16)}…` : "";
  if (status?.start_error) showError(`The agent could not start: ${status.start_error}`);
  else if (status?.host.hosting && status.host.error) showError(`The built-in cloud is not running: ${status.host.error}`);
  if (!signedIn) return;

  renderConnect();

  const pending = pairings.filter((p) => p.status === "pending");
  $("requests-block").classList.toggle("hidden", !pending.length);
  const pendingHtml = pending.map((p) => `
    <div class="pass die-cut ${seenRequests.has(p.id) ? "" : "arrived"}" data-request="${p.id}">
      <div class="pass-main">
        <span class="pass-from">${esc(p.client_name)}</span>
        <p>Asks to run jobs on this Mac. Approve only if ${esc(p.client_name)} shows the same code.</p>
      </div>
      <div class="pass-stub">
        <span class="code">${esc(p.code)}</span>
        <button class="primary" data-approve="${p.id}" data-code="${esc(p.code)}">Approve</button>
      </div>
    </div>`).join("");
  // Compare without the one-time arrival class, so a repeat poll is a no-op.
  if ($("requests").dataset.key !== pending.map((p) => `${p.id}:${p.code}`).join()) {
    $("requests").dataset.key = pending.map((p) => `${p.id}:${p.code}`).join();
    $("requests").innerHTML = pendingHtml;
  }
  pending.forEach((p) => seenRequests.add(p.id));

  const approved = pairings.filter((p) => p.status === "approved");
  setHtml("paired", approved.length
    ? approved.map((p) => `
        <div class="cleared">
          <span class="cleared-code">${esc(deviceCode(p.client_name))}</span>
          <span class="cleared-meta">
            <span class="cleared-name">${esc(p.client_name)}</span>
            <span class="cleared-id">${esc(p.client_id.slice(0, 16))}…</span>
          </span>
          <button class="danger" data-unpair="${p.id}" data-name="${esc(p.client_name)}">Remove</button>
        </div>`).join("")
    : `<div class="board-empty">No devices cleared yet. On your PC, choose this Mac and press Pair; the request appears here to approve.</div>`);

  renderArrivals();

  setHtml("jobs", jobs.length
    ? `<div class="board-head"><span>Time</span><span>Run</span><span>Status</span></div>` +
      jobs.map((j) => {
        const [cls, label] = jobState(j);
        const what = j.rejected ? "Refused: device is not paired" : esc(j.argv.join(" "));
        return `
          <div class="dep ${!j.end && !j.rejected ? "live" : ""}">
            <span class="dep-time">${clock(j.started_ms)}</span>
            <span class="dep-main">
              <span class="dep-cmd ${j.rejected ? "refused" : ""}">${what}</span>
              <span class="dep-sub"><span class="kv"><b>From</b> ${esc(clientName(j.client))}</span>${usageText(j)}</span>
            </span>
            <span class="pill ${cls}">${label}</span>
          </div>`;
      }).join("")
    : `<div class="board-empty">Runs from your paired PC depart from here, with what each one used on this Mac.</div>`);
}

const copyIcon = `<svg class="icon"><use href="#i-copy"/></svg>Copy`;

function renderConnect() {
  const box = $("connect");
  // Until the PC has a way in, the route is the first thing this window asks for.
  $("home").classList.toggle("needs-route", status.host.hosting && tailscale?.state !== "sharing");
  if (!status.host.hosting) {
    const html = `
      <p>This Mac uses the cloud at</p>
      <div class="address"><span>${esc(status.host.url)}</span></div>
      <p class="muted">Enter the same address in Borrowed Compute on your PC.</p>`;
    if (box.dataset.html !== html) { box.dataset.html = html; box.innerHTML = html; }
    return;
  }
  const t = tailscale;
  let body;
  if (!t) {
    body = `<div class="row"><span class="spinner"></span><span class="muted">Checking Tailscale…</span></div>`;
  } else if (t.state === "sharing") {
    body = `
      <p>Your PC reaches this Mac's cloud at:</p>
      <div class="address"><span id="share-url">${breakableUrl(t.url)}</span><button data-copy="${esc(t.url)}">${copyIcon}</button></div>
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
  if (box.dataset.html !== body) { box.dataset.html = body; box.innerHTML = body; }
}

// CPU time and peak memory the job used on this Mac, when the job reported them.
function usageText(j) {
  const u = j.end && j.end.usage;
  if (!u) return "";
  const mb = Math.round(u.max_rss_bytes / 1048576);
  return ` <span class="kv"><b>CPU</b> <span class="num">${fig(u.user_ms + u.system_ms)}</span></span>` +
    ` <span class="kv"><b>Mem</b> <span class="num">${mb}<i>MB</i></span></span>` +
    ` <span class="kv"><b>Ran</b> <span class="num">${fig(u.wall_ms)}</span></span>`;
}

function jobState(j) {
  if (j.rejected) return ["bad", "Refused"];
  if (!j.end) return ["live", "In flight"];
  switch (j.end.kind) {
    case "exited":
      if (j.end.timed_out) return ["bad", "Timed out"];
      return j.end.code === 0 ? ["ok", "Landed"] : ["bad", `Exit ${j.end.code ?? "?"}`];
    case "failed": return ["bad", "Failed"];
    case "cancelled": return ["", "Void"];
    default: return ["", "Lost contact"];
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

async function loadSettings() {
  try {
    const st = await invoke("get_settings");
    $("run-bg").checked = st.run_in_background;
    $("version").textContent = `Version ${st.version}`;
  } catch (e) {
    showError(String(e));
  }
}

$("run-bg").addEventListener("change", async (event) => {
  try {
    await invoke("set_run_in_background", { enabled: event.target.checked });
  } catch (e) {
    event.target.checked = !event.target.checked;
    showError(String(e));
  }
});

function updateStatus(html, cls = "") {
  const box = $("update-status");
  box.className = cls ? `banner ${cls}` : "muted";
  box.innerHTML = html;
}

$("check-update").addEventListener("click", async () => {
  const button = $("check-update");
  button.disabled = true;
  updateStatus("Checking…");
  try {
    const u = await invoke("check_update");
    if (!u.available) {
      updateStatus(`You have the latest version (${esc(u.current)}).`, "ok");
    } else {
      updateStatus(`Version ${esc(u.available)} is ready. <button id="install-update" class="primary">Install and restart</button>`, "warn");
    }
  } catch (e) {
    updateStatus(`Could not check for updates: ${esc(String(e))}`, "bad");
  } finally {
    button.disabled = false;
  }
});

$("update-status").addEventListener("click", async (event) => {
  if (!event.target.closest("#install-update")) return;
  if (!(await ask("Install the update and restart? The agent stops accepting jobs for a few seconds.", "Install and restart"))) return;
  updateStatus("Downloading and installing… The app restarts when it is done.");
  try {
    await invoke("install_update");
  } catch (e) {
    updateStatus(`Could not install the update: ${esc(String(e))}`, "bad");
  }
});

$("open-logs").addEventListener("click", () => {
  invoke("open_logs").catch((e) => showError(String(e)));
});

$("signout").addEventListener("click", async () => {
  if (!(await ask("Sign out? Paired devices will not be able to send jobs until you sign in again.", "Sign out", true))) return;
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
    setTimeout(() => { copy.innerHTML = copyIcon; }, 1500);
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
      if (!(await ask("Stop sharing? Your PC will not reach this Mac's cloud until you share again.", "Stop sharing", true))) return;
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
      if (!(await ask(`Remove ${remove.dataset.name}? It will stop being able to send jobs within 15 seconds.`, "Remove", true))) return;
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
loadSettings();
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refresh(); });
refresh();
