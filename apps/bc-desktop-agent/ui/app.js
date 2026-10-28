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
let services = [];
// Sharing with people on other accounts: the switch, open invites, guests and requests.
let guestsView = { sharing: false, invites: [], guests: [], requests: [] };
let verificationUri = null;
let signingIn = false;
// Requests already on screen, so only new ones animate in.
const seenRequests = new Set();

// ---- The latest run, live ----

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
    $("arr-cmd").textContent = "No runs yet. When your PC sends one, it shows here live.";
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
    $("arr-tab").textContent = "Running";
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
  const stampCls = landed ? "ok" : word === "Cancelled" ? "void" : "bad";
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
  // Animate the result only for a run that ends while the window is open.
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

// How long a service has been up, as "4m" or "2h 10m".
function uptime(startedAt) {
  const s = Math.max(0, Math.floor(Date.now() / 1000) - startedAt);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d`;
}

function clientName(id) {
  const p = pairings.find((p) => p.client_id === id);
  if (p) return p.client_name;
  const g = guestsView.guests.find((g) => g.device === id);
  return g ? `${g.name} (guest)` : `${id.slice(0, 10)}…`;
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

  // Guest requests share the pairing passes; their ids start with "g" to keep them apart.
  const pending = pairings.filter((p) => p.status === "pending")
    .map((p) => ({ ...p, about: `Asks to run jobs on this Mac. Type the code ${p.client_name} shows to approve it.` }))
    .concat(guestsView.requests.map((r) => ({
      id: `g${r.id}`,
      client_name: r.device_name,
      about: `${r.name}, invited as a guest, asks to run jobs on this Mac from ${r.device_name}. Type the code it shows to approve it.`,
    })));
  $("requests-block").classList.toggle("hidden", !pending.length);
  const pendingHtml = pending.map((p) => `
    <div class="pass die-cut ${seenRequests.has(p.id) ? "" : "arrived"}" data-request="${p.id}">
      <div class="pass-main">
        <span class="pass-from">${esc(p.client_name)}</span>
        <p>${esc(p.about)}</p>
      </div>
      <div class="pass-stub">
        <label class="label" for="code-${p.id}">Code on ${esc(p.client_name)}</label>
        <input id="code-${p.id}" class="code-input" type="text" maxlength="9" placeholder="ABCD-EFGH"
               autocomplete="off" spellcheck="false" autocapitalize="characters" data-code-for="${p.id}">
        <button class="primary" data-approve="${p.id}">Approve</button>
        <span class="pass-msg" id="msg-${p.id}"></span>
      </div>
    </div>`).join("");
  // Compare without the one-time arrival class, so a repeat poll is a no-op.
  // Keyed by request id only, so a poll never clears a code being typed.
  if ($("requests").dataset.key !== pending.map((p) => p.id).join()) {
    $("requests").dataset.key = pending.map((p) => p.id).join();
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
    : `<div class="board-empty">No devices paired yet. On your PC, choose this Mac and press Pair; the request appears here to approve.</div>`);

  renderGuests();
  renderArrivals();

  $("services-block").classList.toggle("hidden", !services.length);
  setHtml("services", services.map((s) => `
    <div class="dep svc live">
      <span class="dep-time">${esc(uptime(s.started_at))}</span>
      <span class="dep-main">
        <span class="dep-cmd">${esc(s.argv.join(" "))}</span>
        <span class="dep-sub"><span class="kv"><b>From</b> ${esc(clientName(s.owner))}</span><span class="kv"><b>Up since</b> ${clock(s.started_at * 1000)}</span>${s.ports.length ? `<span class="kv"><b>Port</b> <span class="num">${s.ports.join(", ")}</span></span>` : ""}</span>
      </span>
      <button class="danger" data-stop="${esc(s.job_id)}" data-cmd="${esc(s.argv.join(" "))}">Stop</button>
    </div>`).join(""));

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
    : `<div class="board-empty">Runs from your paired PCs show here, with what each one used on this Mac.</div>`);
}

const copyIcon = `<svg class="icon"><use href="#i-copy"/></svg>Copy`;

function renderConnect() {
  const box = $("connect");
  // Until the PC has a way in, the route is the first thing this window asks for.
  $("home").classList.toggle("needs-route", status.host.hosting && pairings.length === 0);
  if (!status.host.hosting) {
    const html = `
      <p>This Mac uses the cloud at</p>
      <div class="address"><span>${esc(status.host.url)}</span></div>
      <p class="muted">Enter the same address in Borrowed Compute on your PC.</p>`;
    if (box.dataset.html !== html) { box.dataset.html = html; box.innerHTML = html; }
    return;
  }
  const code = status.device_id || "";
  const body = code ? `
      <p>Your PC reaches this Mac with its code:</p>
      <div class="address"><span class="mono code">${esc(code)}</span><button data-copy="${esc(code)}">${copyIcon}</button></div>
      <p class="muted">In Borrowed Compute on your PC, paste this code where it asks for your Mac. The connection is encrypted end to end, and nothing else needs to be installed. You still approve each device before it can send jobs.</p>`
    : `<div class="row"><span class="spinner"></span><span class="muted">Starting…</span></div>`;
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
  if (!j.end) return ["live", "Running"];
  switch (j.end.kind) {
    case "exited":
      if (j.end.timed_out) return ["bad", "Timed out"];
      return j.end.code === 0 ? ["ok", "Done"] : ["bad", `Exit ${j.end.code ?? "?"}`];
    case "failed": return ["bad", "Failed"];
    case "cancelled": return ["", "Cancelled"];
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
      pairings = await invoke("list_pairings");
      jobs = await invoke("recent_jobs");
      services = await invoke("list_services");
      guestsView = await invoke("guests").catch(() => guestsView);
      showError(status.start_error ? `The agent could not start: ${status.start_error}` : "");
    }
  } catch (e) {
    showError(String(e));
  }
  if (status && !status.signed_in) loadProviders();
  render();
}

async function signIn(provider) {
  if (signingIn) return;
  signingIn = true;
  showError("");
  $("signin-btn").disabled = true;
  $("signin-google").disabled = true;
  try {
    account = await invoke("login", { name: $("mac-name").value, provider });
    $("signin-code").classList.add("hidden");
  } catch (e) {
    showError(`Sign-in failed: ${e}`);
    $("signin-code").classList.add("hidden");
  } finally {
    signingIn = false;
    $("signin-btn").disabled = false;
    $("signin-google").disabled = false;
    await refresh();
  }
}

$("signin-btn").addEventListener("click", () => signIn("github"));
$("signin-google").addEventListener("click", () => signIn("google"));

// Google sign-in shows only when this Mac's cloud has a Google client. Asked on each
// refresh while signed out, since the built-in cloud may still be starting.
function loadProviders() {
  invoke("providers").then((p) => $("signin-google").classList.toggle("hidden", !p.google)).catch(() => {});
}

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

// "Downloading 1.9 of 6.9 MB", plus the attempt once a download has started again.
function updateProgressText(p) {
  const mb = (b) => (b / 1e6).toFixed(1);
  let text = p.total ? `Downloading ${mb(p.downloaded)} of ${mb(p.total)} MB` : `Downloading ${mb(p.downloaded)} MB`;
  if (p.attempt > 1) text += ` (connection dropped, try ${p.attempt} of ${p.attempts})`;
  return `${text}… The app restarts when it is done.`;
}

function meterScale(p) {
  return p.total ? `scaleX(${Math.min(1, p.downloaded / p.total).toFixed(3)})` : "";
}

listen("update-progress", (event) => {
  const p = event.payload;
  updateStatus(`${esc(updateProgressText(p))}<div class="meter${p.total ? "" : " unknown"}"><span style="transform: ${meterScale(p)}"></span></div>`);
});

$("update-status").addEventListener("click", async (event) => {
  if (!event.target.closest("#install-update")) return;
  if (!(await ask("Install the update and restart? The agent stops accepting jobs for a few seconds.", "Install and restart"))) return;
  installUpdate();
});

async function installUpdate() {
  updateStatus("Downloading and installing… The app restarts when it is done.");
  try {
    await invoke("install_update");
  } catch (e) {
    updateStatus(`Could not install the update: ${esc(String(e))}`, "bad");
  }
}

// Whatever disk image put this app here, it moves to the latest release on start, so an
// old download never leaves the Mac on an old version. No job can be running yet.
async function updateOnStart() {
  try {
    const u = await invoke("check_update");
    // Local builds are 0.1.0; replacing one with a release would undo the build under test.
    if (u.available && u.current !== "0.1.0") installUpdate();
  } catch (_) {
    // Offline or GitHub unreachable: Check for updates still works later.
  }
}

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
  const approve = event.target.closest("[data-approve]");
  const remove = event.target.closest("[data-unpair]");
  const removeGuest = event.target.closest("[data-guest-remove]");
  const dropInvite = event.target.closest("[data-invite-delete]");
  if (removeGuest) {
    if (!(await ask(`Remove ${removeGuest.dataset.name}? Their device can no longer send jobs, and their servers stop.`, "Remove", true))) return;
    removeGuest.disabled = true;
    await invoke("remove_guest", { device: removeGuest.dataset.guestRemove }).catch((e) => showError(String(e)));
    await refresh();
    return;
  }
  if (dropInvite) {
    dropInvite.disabled = true;
    await invoke("delete_invite", { id: dropInvite.dataset.inviteDelete }).catch((e) => showError(String(e)));
    $("invite-new").classList.add("hidden");
    await refresh();
    return;
  }
  const stop = event.target.closest("[data-stop]");
  if (stop) {
    if (!(await ask(`Stop ${stop.dataset.cmd}? Its process ends and its workspace is deleted.`, "Stop", true))) return;
    stop.disabled = true;
    try { await invoke("stop_service", { jobId: stop.dataset.stop }); } catch (e) { showError(String(e)); }
    await refresh();
    return;
  }
  if (approve) { approvePairing(approve.dataset.approve); return; }
  try {
    if (remove) {
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

// Approves a request with the code typed on its pass. A wrong code keeps the pass open
// with the cloud's answer (tries left, or cancelled after three).
async function approvePairing(id) {
  const input = $(`code-${id}`);
  const button = document.querySelector(`[data-approve="${id}"]`);
  const msg = $(`msg-${id}`);
  const code = input.value.trim();
  if (!code) { msg.textContent = "Type the code shown on the other device."; input.focus(); return; }
  button.disabled = true;
  msg.textContent = "";
  try {
    if (String(id).startsWith("g")) await invoke("approve_guest", { id: Number(id.slice(1)), code });
    else await invoke("approve", { id: Number(id), code });
    await refresh();
  } catch (e) {
    msg.textContent = String(e).replace(/ \(400 Bad Request\)$/, "");
    button.disabled = false;
    input.select();
    // A cancelled request disappears on the next refresh.
    if (/cancelled/.test(String(e))) setTimeout(refresh, 2500);
  }
}

// ---- Guests ----

// The product site. Its domain is not final yet: change it here and every invite follows.
const SITE_URL = "https://borrowed-compute.vercel.app";
const RELEASES = "https://github.com/Harshil089/Borrowed-Compute-releases/releases/latest/download";
const WINDOWS_APP = `${RELEASES}/BorrowedCompute-setup.exe`;

function renderGuests() {
  const g = guestsView;
  if (document.activeElement !== $("sharing")) $("sharing").checked = g.sharing;
  $("guests-body").classList.toggle("hidden", !g.sharing);
  const day = (secs) => new Date(secs * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  setHtml("guests",
    g.guests.map((x) => `
      <div class="cleared">
        <span class="cleared-code">${esc(deviceCode(x.name))}</span>
        <span class="cleared-meta">
          <span class="cleared-name">${esc(x.name)}</span>
          <span class="cleared-id">${esc(x.device_name)} · guest since ${day(x.added)}</span>
        </span>
        <button class="danger" data-guest-remove="${esc(x.device)}" data-name="${esc(x.name)}">Remove</button>
      </div>`).join("") +
    g.invites.map((i) => `
      <div class="cleared invited">
        <span class="cleared-code">${esc(deviceCode(i.name))}</span>
        <span class="cleared-meta">
          <span class="cleared-name">${esc(i.name)}</span>
          <span class="cleared-id">Invited · not used yet · expires ${day(i.expires)}</span>
        </span>
        <button data-invite-delete="${esc(i.id)}">Delete</button>
      </div>`).join(""));
}

$("sharing").addEventListener("change", async (event) => {
  const on = event.target.checked;
  if (!on && guestsView.guests.length &&
      !(await ask("Stop sharing this Mac? Guests can no longer send jobs, and their servers stop. They stay on the list for when you turn it on again.", "Stop sharing", true))) {
    event.target.checked = true;
    return;
  }
  try {
    await invoke("set_sharing", { on });
  } catch (e) {
    event.target.checked = !on;
    showError(String(e));
  }
  if (!on) $("invite-new").classList.add("hidden");
  await refresh();
});

$("invite-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = $("invite-name").value.trim();
  if (!name) { $("invite-name").focus(); return; }
  $("invite-btn").disabled = true;
  try {
    const invite = await invoke("create_invite", { name });
    const host = account?.device_name || "my Mac";
    // A message ready to paste into any chat or email; the invite itself is never shown again.
    const message = `You're invited to run code on ${host} with Borrowed Compute (${SITE_URL}).\n\n` +
      `1. Install the app: ${WINDOWS_APP}\n   (all downloads: ${SITE_URL}/download)\n` +
      `2. Open it, choose "Use a shared Mac" and paste this invite:\n${invite}\n\n` +
      `It works once, within 7 days. Read me the code it shows so I can let you in.\n` +
      `How it works: ${SITE_URL}/how-it-works#guests`;
    $("invite-new").innerHTML = `
      <p>Send this to ${esc(name)} by chat or email. It is shown only now.</p>
      <div class="address"><span>${esc(message)}</span><button data-copy="${esc(message)}">${copyIcon}</button></div>`;
    $("invite-new").classList.remove("hidden");
    $("invite-name").value = "";
  } catch (e) {
    showError(String(e));
  } finally {
    $("invite-btn").disabled = false;
  }
  await refresh();
});

document.addEventListener("keydown", (event) => {
  const input = event.target.closest?.("[data-code-for]");
  if (input && event.key === "Enter") approvePairing(input.dataset.codeFor);
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
updateOnStart();
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refresh(); });
refresh();
