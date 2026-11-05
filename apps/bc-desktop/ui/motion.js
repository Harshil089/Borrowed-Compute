// Motion shared by app.js (the Use side) and lend.js (This device): the sliding sidebar
// highlight, page entrances, the large title that hands off to the bar, count-ups and the
// command palette. Each motion answers "what just changed?". Anything started from the
// keyboard switches instantly, and reduced motion turns all of it off (base.css).

(() => {
const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const isMac = /Mac/.test(navigator.platform || navigator.userAgent);
const mod = isMac ? "⌘" : "Ctrl+";
// True while a keyboard command runs, so whatever it changes switches without motion.
let fromKeyboard = false;

// ---- Sidebar highlight: one pill slides to where you are ----
let pillTop = null;
function moveNav(el, instant) {
  const pill = $("nav-pill"), side = $("sidebar");
  if (!pill || !side) return 0;
  if (!el || !el.offsetParent) { pill.classList.remove("on"); pillTop = null; return 0; }
  const top = el.getBoundingClientRect().top - side.getBoundingClientRect().top + side.scrollTop;
  const dir = pillTop == null ? 0 : Math.sign(top - pillTop);
  const jump = instant || fromKeyboard || !pill.classList.contains("on");
  if (jump) pill.classList.add("no-anim");
  pill.style.height = `${el.offsetHeight}px`;
  pill.style.transform = `translateY(${top}px)`;
  pill.classList.add("on");
  if (jump) { void pill.offsetWidth; pill.classList.remove("no-anim"); }
  pillTop = top;
  return dir;
}
// The highlighted item is the selected device on the run page, else the open device page.
function current() {
  const runShown = !$("run-view")?.classList.contains("hidden");
  return runShown ? document.querySelector(".mac.selected") : document.querySelector('[data-nav][aria-current="page"]');
}
function syncNav(instant) { return moveNav(current(), instant); }
addEventListener("resize", () => syncNav(true));
// Lists above the highlight grow and shrink (nearby devices, servers): follow without motion,
// but leave a move already under way alone.
document.addEventListener("DOMContentLoaded", () => {
  const side = $("sidebar");
  if (!side) return;
  new MutationObserver(() => {
    const el = current(), s = $("sidebar");
    if (!el || !el.offsetParent) return syncNav(true);
    const top = el.getBoundingClientRect().top - s.getBoundingClientRect().top + s.scrollTop;
    if (top !== pillTop) syncNav(true);
  }).observe(side, { childList: true, subtree: true });
  side.addEventListener("scroll", () => syncNav(true), { passive: true });
});

// ---- Page entrances: from the direction you travelled ----
function enter(el, dir) {
  if (!el || reduced() || fromKeyboard || !dir) return;
  el.style.setProperty("--dy", `${dir * 8}px`);
  el.classList.remove("entering");
  void el.offsetWidth;
  el.classList.add("entering");
}

// ---- The large title hands off to the frosted bar once you scroll under it ----
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".view").forEach((v) =>
    v.addEventListener("scroll", () => v.classList.toggle("scrolled", v.scrollTop > 24), { passive: true }));
});

// ---- Count-up: a measured figure settles into place once, when a run ends ----
function countUp(root) {
  if (reduced()) return;
  root.querySelectorAll("[data-count]").forEach((el, i) => {
    const m = el.innerHTML.match(/^(\d+(?:\.\d+)?)([\s\S]*)$/);
    if (!m) return;
    const target = parseFloat(m[1]), decimals = (m[1].split(".")[1] || "").length, rest = m[2];
    const start = performance.now() + i * 50, dur = 700;
    const step = (now) => {
      const p = Math.max(0, Math.min(1, (now - start) / dur));
      el.innerHTML = (target * (1 - Math.pow(1 - p, 3))).toFixed(decimals) + rest;
      if (p < 1) requestAnimationFrame(step);
    };
    el.innerHTML = (0).toFixed(decimals) + rest;
    requestAnimationFrame(step);
  });
}

// ---- Command palette: every page, device and action, from the keyboard ----
const sources = [];
let overlay = null, items = [], sel = 0;

function render(text) {
  const t = text.trim().toLowerCase();
  items = sources.flatMap((f) => f()).filter((c) => !t || `${c.group} ${c.label} ${c.sub || ""}`.toLowerCase().includes(t));
  sel = Math.min(sel, Math.max(0, items.length - 1));
  const list = overlay.querySelector("ul");
  if (!items.length) { list.innerHTML = `<li class="none">Nothing matches.</li>`; return; }
  let group = null, html = "";
  items.forEach((c, i) => {
    if (c.group !== group) { group = c.group; html += `<li class="group" role="presentation">${esc(group)}</li>`; }
    html += `<li role="option" data-i="${i}" aria-selected="${i === sel}"><svg class="icon"><use href="#${c.icon || "i-plane"}"/></svg><span>${esc(c.label)}</span>${c.sub ? `<span class="hint">${esc(c.sub)}</span>` : ""}</li>`;
  });
  list.innerHTML = html;
  list.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
}

function choose(i, viaKeys) {
  const c = items[i];
  if (!c) return;
  closePalette();
  fromKeyboard = viaKeys;
  try { c.run(); } finally { fromKeyboard = false; }
}

function openPalette(viaKeys) {
  if (overlay) return;
  overlay = document.createElement("div");
  overlay.className = `overlay${viaKeys ? " instant" : ""}`;
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  overlay.setAttribute("aria-label", "Command palette");
  overlay.innerHTML = `<div class="palette">
    <input type="text" placeholder="Go to a device, a page or an action" autocomplete="off" spellcheck="false" aria-label="Search">
    <ul role="listbox"></ul></div>`;
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) closePalette(); });
  overlay.addEventListener("mousemove", (e) => {
    const li = e.target.closest("[data-i]");
    if (!li || +li.dataset.i === sel) return;
    sel = +li.dataset.i;
    overlay.querySelectorAll("[data-i]").forEach((x) => x.setAttribute("aria-selected", String(+x.dataset.i === sel)));
  });
  overlay.addEventListener("click", (e) => { const li = e.target.closest("[data-i]"); if (li) choose(+li.dataset.i, false); });
  const q = overlay.querySelector("input");
  q.addEventListener("input", () => { sel = 0; render(q.value); });
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(items.length - 1, sel + 1); render(q.value); }
    else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(0, sel - 1); render(q.value); }
    else if (e.key === "Enter") { e.preventDefault(); choose(sel, true); }
    else if (e.key === "Escape") { e.preventDefault(); closePalette(); }
  });
  document.body.appendChild(overlay);
  sel = 0;
  render("");
  q.focus();
}

function closePalette() {
  overlay?.remove();
  overlay = null;
}

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") {
    // Only once signed in: the palette's commands all act on the signed-in app.
    if ($("app-view")?.classList.contains("hidden")) return;
    e.preventDefault();
    if (overlay) closePalette(); else openPalette(true);
  }
});

window.motion = {
  mod,
  reduced,
  moveNav,
  syncNav,
  enter,
  countUp,
  openPalette,
  addCommands: (fn) => sources.push(fn),
  keyboard: () => fromKeyboard,
};
})();
