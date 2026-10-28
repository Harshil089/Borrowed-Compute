// Borrowed Compute website: theme, menu, reveals, the silicon die and the run it replays.
(() => {
  const root = document.documentElement;
  root.classList.add("js");
  // ?still shows every part settled, for screenshots and reviews.
  const still = location.search.includes("still");
  const reduced = still || matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  // ---- Theme: dark is the die's own light; daylight is optional and remembered ----
  $$(".theme-btn").forEach((btn) => btn.addEventListener("click", () => {
    const next = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = next;
    try { localStorage.setItem("bc-theme", next); } catch (_) {}
    window.dispatchEvent(new Event("themechange"));
  }));

  // ---- Menu on narrow screens ----
  const bar = $(".bar");
  const menu = $(".menu-btn");
  if (bar && menu) {
    const set = (open) => { bar.classList.toggle("open", open); menu.setAttribute("aria-expanded", String(open)); };
    menu.addEventListener("click", () => set(!bar.classList.contains("open")));
    $$(".nav a").forEach((a) => a.addEventListener("click", () => set(false)));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") set(false); });
  }

  // ---- Reveals: parts settle in once, then stay ----
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
  $$(".reveal").forEach((el) => (reduced ? el.classList.add("in") : io.observe(el)));

  // ---- Copy buttons ----
  $$("[data-copy]").forEach((btn) => btn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(btn.dataset.copy); } catch (_) { return; }
    const label = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = label; }, 1600);
  }));

  // ---- Docs index follows the reader ----
  const toc = $$(".toc a");
  if (toc.length) {
    const map = new Map(toc.map((a) => [a.getAttribute("href").slice(1), a]));
    const seen = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        toc.forEach((a) => a.classList.remove("on"));
        map.get(e.target.id)?.classList.add("on");
      }
    }, { rootMargin: "-20% 0px -70% 0px" });
    map.forEach((_, id) => { const s = document.getElementById(id); if (s) seen.observe(s); });
  }

  // ---- Releases: versions on the download page and the What's new timeline ----
  // Served by /api/releases on this site, which reads the public feed server side.
  const versionSlots = $$("[data-version]");
  const timeline = $("#timeline");
  if (versionSlots.length || timeline) {
    const day = (d) => new Date(d).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
    const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
    fetch("/api/releases")
      .then((r) => r.json())
      .then(({ releases = [] }) => {
        for (const slot of versionSlots) {
          const rel = releases.find((r) => r.key === slot.dataset.version);
          if (!rel) continue;
          slot.textContent = rel.version;
          const when = slot.parentElement.querySelector("[data-released]");
          if (when) when.textContent = day(rel.date);
        }
        if (!timeline) return;
        if (!releases.length) { timeline.innerHTML = `<li class="timeline-empty">Release history is unavailable right now. Try again in a minute.</li>`; return; }
        timeline.innerHTML = releases.map((r, i) => `
          <li class="rel" data-app="${r.key}">
            <div class="rel-when"><span class="marking">${day(r.date)}</span></div>
            <div class="rel-body pkg flat">
              <div class="rel-head"><span class="chip-tag${i < 2 ? " on" : ""}">${r.app}</span><span class="fig">${esc(r.version)}</span></div>
              <ul>${r.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
            </div>
          </li>`).join("");
      })
      .catch(() => { if (timeline) timeline.innerHTML = `<li class="timeline-empty">Release history is unavailable right now. Try again in a minute.</li>`; });
    $$("[data-filter]").forEach((btn) => btn.addEventListener("click", () => {
      $$("[data-filter]").forEach((b) => { b.classList.toggle("on", b === btn); b.setAttribute("aria-pressed", String(b === btn)); });
      $$(".rel").forEach((li) => { li.hidden = btn.dataset.filter !== "all" && li.dataset.app !== btn.dataset.filter; });
    }));
  }

  // ---- Hero: the Windows app replays one real, measured run ----
  // The usage figures are the real run's; the 61 seconds play back in about six.
  const win = $(".app-window");
  if (win) {
    const typed = $(".typed", win);
    const phases = $$(".win-phases li", win);
    const clock = $("[data-clock]", win);
    const log = $("[data-log]", win);
    const COMMAND = "python3 heavy_benchmark.py";
    const RUN = 61.0;
    const LINES = [
      [0.0, '<span class="d">connected   studio · direct · 41 ms</span>'],
      [0.0, '<span class="d">uploaded    18 KB in 0.1 s</span>'],
      [0.0, '<span class="g">$ python3 heavy_benchmark.py</span>'],
      [1.2, "workers 8 · matrix 2048 x 2048"],
      [10.4, "round 1/6   10.2 s"],
      [20.6, "round 2/6   10.1 s"],
      [30.8, "round 3/6   10.2 s"],
      [41.0, "round 4/6   10.1 s"],
      [51.1, "round 5/6   10.1 s"],
      [61.0, "round 6/6    9.9 s"],
      [61.0, '<span class="ok">done · 6 rounds in 61.0 s</span>'],
    ];
    const setPhase = (at) => phases.forEach((li, i) => { li.classList.toggle("done", i < at); li.classList.toggle("active", i === at); });
    const showLog = (secs) => { log.innerHTML = LINES.filter(([t]) => t <= secs).map(([, l]) => l).join("\n"); };
    const settle = () => { win.dataset.state = "landed"; typed.textContent = COMMAND; setPhase(4); clock.textContent = `${RUN.toFixed(1)} s`; showLog(RUN); };

    if (reduced) settle();
    else {
      let t = 0, step = "typing", visible = false, last = 0, raf = 0;
      const start = () => { step = "typing"; t = 0; win.dataset.state = "typing"; typed.textContent = ""; setPhase(-1); clock.textContent = "0.0 s"; log.innerHTML = ""; };
      start();
      const tick = (now) => {
        const dt = Math.min(64, now - last); last = now; t += dt;
        if (step === "typing") {
          const n = Math.min(COMMAND.length, Math.floor((t - 500) / 55));
          typed.textContent = n > 0 ? COMMAND.slice(0, n) : "";
          if (n >= COMMAND.length && t > 500 + COMMAND.length * 55 + 450) { step = "pressed"; t = 0; win.dataset.state = "pressed"; }
        } else if (step === "pressed" && t > 260) { step = "pack"; t = 0; win.dataset.state = "running"; setPhase(0); }
        else if (step === "pack" && t > 500) { step = "upload"; t = 0; setPhase(1); }
        else if (step === "upload" && t > 650) { step = "run"; t = 0; setPhase(2); }
        else if (step === "run") {
          const secs = Math.min(RUN, t / 100);
          clock.textContent = `${secs.toFixed(1)} s`;
          showLog(secs);
          if (secs >= RUN) { step = "return"; t = 0; setPhase(3); }
        } else if (step === "return" && t > 500) { step = "landed"; t = 0; settle(); }
        else if (step === "landed" && t > 7000) start();
        raf = visible ? requestAnimationFrame(tick) : 0;
      };
      new IntersectionObserver(([e]) => {
        visible = e.isIntersecting;
        if (visible && !raf) { last = performance.now(); raf = requestAnimationFrame(tick); }
      }).observe(win);
    }
  }

  // ---- Steps: a gold wire drawn from the PC pad to the Mac die as the reader scrolls ----
  const steps = $(".wire-steps");
  if (steps) {
    const pads = $$(".step", steps);
    const update = () => {
      const r = steps.getBoundingClientRect();
      const vh = window.innerHeight;
      const p = reduced ? 1 : Math.max(0, Math.min(1, (vh * 0.85 - r.top) / (vh * 0.6)));
      steps.style.setProperty("--p", p.toFixed(3));
      pads.forEach((el, i) => el.classList.toggle("lit", p >= i / Math.max(1, pads.length - 1) - 0.02));
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
  }
})();
