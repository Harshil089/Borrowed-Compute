// This device as a lender: requests to approve, runs from other devices, paired devices,
// sharing and settings. All state lives in the Rust side (`lend.rs`); this file renders it.
// It runs beside app.js in its own scope; `window.lendUI` lets app.js switch views.

(() => {
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

let status = null;
let account = null;
let pairings = [];
// Model parts this device runs for other devices' meshes (`mesh_hosted`).
let hosted = { stages: [], held_bytes: 0 };
let jobs = [];
let services = [];
// Sharing with people on other accounts: the switch, open invites, guests and requests.
let guestsView = { sharing: false, invites: [], guests: [], requests: [] };
// Requests already on screen, so only new ones animate in.
const seenRequests = new Set();

// ---- The latest run, live ----

const ARR_LEGS = ["upload", "run", "return"];
let arrivalsKey = null;
let liveSince = null;
let endedAt = null; // when the last run on the card ended

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
  const me = account?.device_name || "This computer";
  $("arr-to").textContent = deviceCode(me);
  $("arr-to-name").textContent = me;
  $("arr-date").textContent = new Date().toLocaleDateString(undefined, { month: "short", day: "numeric" });
  // A server has no finish line: with no batch run live, the card shows it until it stops.
  const svc = !(job && !job.end) && services[0];
  if (svc) { renderServing(svc); return; }
  $("live-heading").textContent = "Current run";
  const key = job ? `${job.job_id}:${job.end ? job.end.kind : "live"}` : "none";
  if (key === arrivalsKey) return;
  const firstRender = arrivalsKey === null;
  arrivalsKey = key;

  if (!job) {
    $("arr-tab").textContent = "Waiting";
    $("arr-from").textContent = "DEV";
    $("arr-from-name").textContent = "Another device";
    $("arr-cmd").className = "arr-cmd";
    $("arr-cmd").textContent = "No runs yet. When a paired device sends one, it shows here live.";
    $("arr-clock").textContent = "";
    $("arr-stub").classList.add("hidden");
    liveSince = null;
    endedAt = null;
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
    endedAt = null;
    tickClock();
    setArrivalLegs(1, false);
    return;
  }

  liveSince = null;
  endedAt = job.finished_ms || null;
  const [cls, word] = jobState(job);
  $("arr-tab").textContent = word;
  if (!endedAt) $("arr-clock").textContent = "";
  tickClock();
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

function renderServing(svc) {
  liveSince = svc.started_at * 1000;
  endedAt = null;
  tickClock();
  const key = `svc:${svc.job_id}:${svc.ports.join(",")}`;
  if (key === arrivalsKey) return;
  arrivalsKey = key;
  const from = clientName(svc.owner);
  $("live-heading").textContent = "Serving";
  $("arr-tab").textContent = "Serving";
  $("arr-from").textContent = deviceCode(from);
  $("arr-from-name").textContent = from;
  $("arr-cmd").className = "arr-cmd argv";
  $("arr-cmd").textContent = svc.argv.join(" ");
  setArrivalLegs(1, false);
  const stub = $("arr-stub");
  stub.innerHTML = `
    <div class="arr-time"><span class="label">${svc.ports.length > 1 ? "Ports" : "Port"}</span><strong class="num">${svc.ports.length ? esc(svc.ports.join(", ")) : "–"}</strong></div>
    <div class="arr-figs"><span class="plain"><b>Runs until stopped</b>here or on ${esc(from)}, keeping this computer awake</span></div>
    <button class="danger stop" data-lend-stop="${esc(svc.job_id)}" data-cmd="${esc(svc.argv.join(" "))}">Stop</button>`;
  stub.classList.remove("hidden", "arrived");
}

// Once a run ends, the figure counts the time since, so it never sits at "just now".
function tickClock() {
  if (liveSince == null && endedAt != null) {
    const s = (Date.now() - endedAt) / 1000;
    $("arr-clock").innerHTML = s < 86400
      ? `${esc(window.motion.since(s))}<i>ago</i>`
      : esc(new Date(endedAt).toLocaleDateString());
    return;
  }
  if (liveSince == null) return;
  const s = Math.max(0, Math.round((Date.now() - liveSince) / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  $("arr-clock").innerHTML = s < 60 ? `${s}<i>s</i>`
    : h ? `${h}<i>h</i> ${m}<i>m</i>` : `${m}<i>m</i> ${s % 60}<i>s</i>`;
}
setInterval(tickClock, 1000);

// ---- This computer's cores, measured once a second while a run or server is up ----
// Each cell's opacity is that core's measured use; the whole grid's colour shows the
// average: ink when light, yellow over 40%, red over 66%.
let coresIdle = true;
function sizeCores(n) {
  const grid = $("cores");
  if (grid.children.length === n) return;
  grid.innerHTML = "<div class=\"core\"><i></i></div>".repeat(n);
  grid.style.setProperty("--cols", String(n <= 8 ? n : n <= 12 ? 6 : 8));
}
function paintCores(loads) {
  sizeCores(loads.length);
  const cells = $("cores").children;
  let sum = 0;
  loads.forEach((l, i) => { sum += l; cells[i].style.setProperty("--l", l > 0 ? Math.max(0.06, l).toFixed(2) : "0"); });
  const avg = loads.length ? sum / loads.length : 0;
  const tier = avg > 0.66 ? "high" : avg > 0.4 ? "mid" : "low";
  const card = $("live-run");
  card.dataset.heat = tier;
  card.style.setProperty("--heat", (avg * 1.4).toFixed(2));
  document.querySelectorAll("#heat-legend span").forEach((s) => s.classList.toggle("on", s.dataset.t === tier));
  $("cores-busy").textContent = sum > 0 ? `${sum.toFixed(1)} of ${loads.length} busy` : "idle";
}
sizeCores(navigator.hardwareConcurrency || 8);
$("cores-title").textContent = /Mac/.test(navigator.platform || navigator.userAgent) ? "Cores on this Mac" : "Cores on this PC";
setInterval(async () => {
  if (document.visibilityState !== "visible" || !status?.signed_in) return;
  const busy = status.lending && (jobs.some((j) => !j.rejected && !j.end) || services.length > 0 || hosted.stages.length > 0);
  if (!busy) {
    if (!coresIdle) { paintCores(new Array($("cores").children.length).fill(0)); coresIdle = true; }
    return;
  }
  try { paintCores(await invoke("cpu_load")); coresIdle = false; } catch { /* next tick tries again */ }
}, 1000);

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
  const box = $("lend-error");
  box.textContent = message || "";
  box.classList.toggle("hidden", !message);
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

const size = (b) => (b >= 1 << 30 ? `${(b / (1 << 30)).toFixed(1)} GB` : `${Math.round(b / (1 << 20))} MB`);

// The same rows on Overview and on the Mesh page: lending to a mesh is never invisible.
function renderHosted() {
  const rows = hosted.stages.map((h) => {
    const where = h.gpu_layers > 0
      ? `${esc(h.gpu)}${h.gpu_layers < h.layers.end - h.layers.start ? `, ${h.gpu_layers} of ${h.layers.end - h.layers.start} layers` : ""}`
      : "the CPU";
    const state = h.running
      ? `<span class="pill ok">${h.idle_secs != null && h.idle_secs < 10 ? "Working" : "Ready"}</span>`
      : `<span class="pill warn">Downloading ${h.total ? Math.floor((h.loaded / h.total) * 100) : 0}%</span>`;
    const sub = h.running
      ? `<span class="kv"><b>On</b> ${where}</span><span class="kv"><b>Memory</b> <span class="num">${size(h.total)}</span></span><span class="kv"><b>Batches</b> <span class="num">${h.batches}</span></span>`
      : `<span class="kv"><b>Fetching</b> <span class="num">${size(h.loaded)}<i>of ${size(h.total)}</i></span></span>`;
    return `
      <div class="mesh-stage mesh-hosted-row">
        <span class="dep-main">
          <span class="dep-cmd">${esc(h.model)}, layers ${h.layers.start}–${h.layers.end - 1} of ${h.of}</span>
          <span class="dep-sub"><span class="kv"><b>For</b> ${h.coordinator === status?.device_id ? "this computer's own mesh" : esc(clientName(h.coordinator))}</span>${sub}</span>
          <span class="mesh-layers"><span style="left:${(h.layers.start / h.of) * 100}%;width:${((h.layers.end - h.layers.start) / h.of) * 100}%"></span></span>
        </span>
        ${state}
      </div>`;
  }).join("");
  for (const id of ["lend-mesh", "mesh-hosted"]) {
    $(`${id}-block`).classList.toggle("hidden", !rows);
    setHtml(id, rows);
  }
}

// Downloads and batches move fast; follow them closely while there are any.
setInterval(async () => {
  if (document.visibilityState !== "visible" || !hosted.stages.length && !status?.lending) return;
  try {
    hosted = await invoke("mesh_hosted");
    renderHosted();
  } catch { /* the next tick tries again */ }
}, 1500);

function clientName(id) {
  const p = pairings.find((p) => p.client_id === id);
  if (p) return p.client_name;
  const g = guestsView.guests.find((g) => g.device === id);
  return g ? `${g.name} (guest)` : `${id.slice(0, 10)}…`;
}

function renderStatus() {
  const pill = $("lend-pill");
  let cls = "", label = "Starting";
  if (!status) {
    label = "Starting";
  } else if (status.start_error) {
    cls = "bad"; label = "Closed";
  } else if (status.host.hosting && status.host.error) {
    cls = "bad"; label = "Cloud stopped";
  } else if (!status.serving) {
    label = "Starting";
  } else if (!status.lending) {
    label = "Not lending";
  } else if (status.cloud.ok === false) {
    cls = "warn"; label = "Cloud unreachable";
  } else if (status.cloud.ok === true) {
    cls = "ok"; label = "Lending";
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
  $("lend-nav").classList.toggle("hidden", !signedIn);
  if (!signedIn && !$("lend-view").classList.contains("hidden")) showRun();
  if (document.activeElement !== $("lending")) $("lending").checked = !!status?.lending;
  $("live-run").classList.toggle("paused", !!status && !status.lending);
  if (status?.start_error) showError(`The agent could not start: ${status.start_error}`);
  else if (status?.host.hosting && status.host.error) showError(`The built-in cloud is not running: ${status.host.error}`);
  if (!signedIn) return;

  renderConnect();

  // Guest requests share the pairing passes; their ids start with "g" to keep them apart.
  const pending = pairings.filter((p) => p.status === "pending")
    .map((p) => ({ ...p, about: `Asks to run jobs on this computer. Type the code ${p.client_name} shows to approve it.` }))
    .concat(guestsView.requests.map((r) => ({
      id: `g${r.id}`,
      client_name: r.device_name,
      about: `${r.name}, invited as a guest, asks to run jobs on this computer from ${r.device_name}. Type the code it shows to approve it.`,
    })));
  $("requests-block").classList.toggle("hidden", !pending.length);
  const pendingHtml = pending.map((p) => `
    <div class="fold" data-fold="${p.id}"><div>
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
    </div>
    </div></div>`).join("");
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
    : `<div class="board-empty">No devices paired yet. On another computer, choose this one and press Pair; the request appears on Overview to approve.</div>`);

  renderGuests();
  renderArrivals();
  renderHosted();

  $("lend-services-block").classList.toggle("hidden", !services.length);
  setHtml("lend-services", services.map((s) => `
    <div class="dep svc live">
      <span class="dep-time">${esc(uptime(s.started_at))}</span>
      <span class="dep-main">
        <span class="dep-cmd">${esc(s.argv.join(" "))}</span>
        <span class="dep-sub"><span class="kv"><b>From</b> ${esc(clientName(s.owner))}</span><span class="kv"><b>Up since</b> ${clock(s.started_at * 1000)}</span>${s.ports.length ? `<span class="kv"><b>Port</b> <span class="num">${s.ports.join(", ")}</span></span>` : ""}</span>
      </span>
      <button class="danger" data-lend-stop="${esc(s.job_id)}" data-cmd="${esc(s.argv.join(" "))}">Stop</button>
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
    : `<div class="board-empty">Runs from your paired devices show here, with what each one used on this computer.</div>`);
}

const copyIcon = `<svg class="icon"><use href="#i-copy"/></svg>Copy`;

function renderConnect() {
  const box = $("connect");
  if (!status.host.hosting) {
    const html = `<p class="muted">Your account lives on another computer, so other devices sign in through that one. Its code is under Paired devices there. Here, approve the devices that may send jobs.</p>`;
    if (box.dataset.html !== html) { box.dataset.html = html; box.innerHTML = html; }
    return;
  }
  const code = status.device_id || "";
  const body = code ? `
      <p>Your account lives on this computer. Your other devices sign in through it with its code:</p>
      <div class="address"><span class="mono code">${esc(code)}</span><button data-copy="${esc(code)}">${copyIcon}</button></div>
      <p class="muted">In Borrowed Compute on another computer, paste this code where it asks where your account lives, or pick this one from its network list. The connection is encrypted end to end. You still approve each device before it can send jobs.</p>`
    : `<div class="row"><span class="spinner"></span><span class="muted">Starting…</span></div>`;
  if (box.dataset.html !== body) { box.dataset.html = body; box.innerHTML = body; }
}

// CPU time and peak memory the job used here, when the job reported them.
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
    const st = await invoke("get_status");
    status = { ...st.lend, signed_in: st.signed_in, device_id: st.device_id };
    if (!status.signed_in) account = null;
    if (status.signed_in) {
      // app.js handles a session another cloud refused; here the name is all that matters.
      if (!account) account = await invoke("get_account").catch(() => null);
      pairings = await invoke("list_pairings");
      jobs = await invoke("recent_jobs");
      services = await invoke("list_services");
      guestsView = await invoke("guests").catch(() => guestsView);
      hosted = await invoke("mesh_hosted").catch(() => hosted);
      showError(status.start_error ? `The agent could not start: ${status.start_error}` : "");
    }
  } catch (e) {
    showError(String(e));
  }
  render();
}

async function loadSettings() {
  try {
    const st = await invoke("get_settings");
    $("run-bg").checked = st.run_in_background;
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

$("open-logs").addEventListener("click", () => {
  invoke("open_logs").catch((e) => showError(String(e)));
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
  const stop = event.target.closest("[data-lend-stop]");
  if (stop) {
    if (!(await ask(`Stop ${stop.dataset.cmd}? Its process ends and its workspace is deleted.`, "Stop", true))) return;
    stop.disabled = true;
    try { await invoke("stop_service", { jobId: stop.dataset.lendStop }); } catch (e) { showError(String(e)); }
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
    // Confirm on the request itself, then fold it away so the page below glides up.
    const fold = document.querySelector(`[data-fold="${id}"]`);
    if (fold && !window.motion?.reduced()) {
      fold.querySelector(".pass-stub").innerHTML = `<span class="pass-done">Paired</span>`;
      await new Promise((r) => setTimeout(r, 650));
      fold.classList.add("closing");
      await new Promise((r) => setTimeout(r, 300));
    }
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
      !(await ask("Stop sharing this device? Guests can no longer send jobs, and their servers stop. They stay on the list for when you turn it on again.", "Stop sharing", true))) {
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
    const host = account?.device_name || "my computer";
    // A message ready to paste into any chat or email; the invite itself is never shown again.
    const message = `You're invited to run code on ${host} with Borrowed Compute (${SITE_URL}).\n\n` +
      `1. Install the app: ${SITE_URL}/download\n` +
      `2. Open it, choose "Use a shared device" and paste this invite:\n${invite}\n\n` +
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

listen("agent-event", refresh);
listen("status-changed", refresh);

// Pairing requests come from the cloud, so poll while the window is open.
setInterval(() => { if (document.visibilityState === "visible") refresh(); }, 5000);
loadSettings();
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refresh(); });
refresh();

// ---- Lending switch ----
$("lending").addEventListener("change", async (event) => {
  const on = event.target.checked;
  if (!on && !(await ask("Stop lending this computer? Paired devices can no longer start runs here; runs in progress finish.", "Stop lending", true))) {
    event.target.checked = true;
    return;
  }
  try {
    await invoke("set_lending", { on });
  } catch (e) {
    event.target.checked = !on;
    showError(String(e));
  }
  await refresh();
});

// ---- Views: the run page, or one of this device's pages, remembered between launches ----
function showRun() {
  try { localStorage.removeItem("page"); } catch { /* storage off */ }
  const wasHidden = $("run-view").classList.contains("hidden");
  $("lend-view").classList.add("hidden");
  $("run-view").classList.remove("hidden");
  document.querySelectorAll("[data-nav]").forEach((b) => b.setAttribute("aria-current", "false"));
  const dir = window.motion?.syncNav() ?? 0;
  if (wasHidden) window.motion?.enter($("run-view"), dir);
}

function showPage(name) {
  const before = $("run-view").classList.contains("hidden") ? document.querySelector('[data-nav][aria-current="page"]')?.dataset.nav : "run";
  $("run-view").classList.add("hidden");
  $("lend-view").classList.remove("hidden");
  document.querySelectorAll("[data-page]").forEach((p) => p.classList.toggle("hidden", p.dataset.page !== name));
  document.querySelectorAll("[data-nav]").forEach((b) => b.setAttribute("aria-current", b.dataset.nav === name ? "page" : "false"));
  $("page-title").textContent = document.querySelector(`[data-nav="${name}"]`).dataset.title;
  try { localStorage.setItem("page", name); } catch { /* storage off */ }
  $("lend-view").scrollTop = 0;
  $("lend-view").classList.remove("scrolled");
  const dir = window.motion?.syncNav() ?? 0;
  if (before !== name) window.motion?.enter(document.querySelector(`[data-page="${name}"]`), dir);
}
document.querySelectorAll("[data-nav]").forEach((b) => b.addEventListener("click", () => showPage(b.dataset.nav)));
window.lendUI = { showRun, showPage, refresh };
// The command palette's This device entries.
window.motion?.addCommands(() => {
  if (!status?.signed_in) return [];
  const pages = [...document.querySelectorAll("[data-nav]")].map((b) => ({
    group: "This device",
    label: b.dataset.title,
    icon: b.querySelector("use")?.getAttribute("href").slice(1),
    run: () => showPage(b.dataset.nav),
  }));
  pages.push({ group: "This device", label: status.lending ? "Stop lending this computer" : "Lend this computer", icon: "i-gear", run: () => $("lending").click() });
  return pages;
});
let savedPage = null;
try { savedPage = localStorage.getItem("page"); } catch { /* storage off */ }
if (savedPage) showPage(savedPage);

// Requests waiting show as a count on Overview, whichever page is open.
new MutationObserver(() => {
  const n = $("requests").querySelectorAll("[data-request]").length;
  $("nav-count").textContent = String(n);
  $("nav-count").classList.toggle("hidden", n === 0);
}).observe($("requests"), { childList: true });
})();
