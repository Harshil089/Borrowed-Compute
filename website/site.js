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

  // ---- The die ----
  function rng(seed) {
    return () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  }
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  // The die is vector art built here, so it stays sharp at any zoom. Floorplan in die units
  // (0 to 1000), loosely after an Apple Silicon layout: memory PHYs along two edges, a
  // performance and an efficiency cluster with their L2, the neural engine, the system
  // cache, sixteen GPU cores and the media blocks. Cores of one kind are copies of one
  // design, as on a real die. Illustrative, not to scale.
  const NS = "http://www.w3.org/2000/svg";
  const HUE = {
    pcore: ["#8a4a98", "#4b2459"], ecore: ["#b8703f", "#6b3a1c"], l2: ["#4a9a86", "#245347"],
    slc: ["#4580b4", "#203f62"], gpu: ["#c39a4c", "#6f5121"], npu: ["#6466bf", "#30337a"],
    media: ["#8f9548", "#4f5325"], misc: ["#6d4590", "#382150"], phy: ["#348790", "#17444a"],
  };
  // Deterministic "random" so every load draws the same die.
  function seq(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

  function dieSVG(labels) {
    const parts = [];
    const r = (x, y, w, h, a = "") => parts.push(`<rect x="${+x.toFixed(2)}" y="${+y.toFixed(2)}" width="${+w.toFixed(2)}" height="${+h.toFixed(2)}" ${a}/>`);
    // One functional unit: metallic gradient, its circuitry as a fine pattern, a crisp edge.
    const unit = (x, y, w, h, hue, tex, tone = 0) => {
      r(x, y, w, h, `fill="url(#g-${hue})"` + (tone ? ` opacity="${1 - tone}"` : ""));
      r(x, y, w, h, `fill="url(#p-${tex})"`);
      r(x + 0.5, y + 0.5, w - 1, h - 1, `fill="none" stroke="rgba(255,255,255,0.28)" stroke-width="0.9"`);
    };
    // A block split into the macros a placed design has, with routing gutters between.
    const macros = (x, y, w, h, hue, seed, depth) => {
      const rnd = seq(seed);
      const split = (x, y, w, h, d) => {
        if (d === 0 || w < 40 || h < 40) { unit(x, y, w, h, hue, rnd() < 0.28 ? "sram" : "logic", rnd() * 0.25); return; }
        const g = 4, vert = w >= h, k = 0.32 + rnd() * 0.36;
        if (vert) { const a = Math.round((w - g) * k); split(x, y, a, h, d - 1); split(x + a + g, y, w - a - g, h, d - 1); }
        else { const a = Math.round((h - g) * k); split(x, y, w, a, d - 1); split(x, y + a + g, w, h - a - g, d - 1); }
      };
      split(x, y, w, h, depth);
    };
    const pcore = (x, y) => {
      unit(x, y, 36, 170, "l2", "sram");
      unit(x + 40, y, 100, 38, "pcore", "logic");
      for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) unit(x + 40 + i * 34, y + 42 + j * 36, 32, 34, "pcore", "logic", 0.12 * ((i + j) % 2));
      unit(x + 40, y + 118, 62, 52, "pcore", "logic", 0.2);
      unit(x + 106, y + 118, 34, 52, "l2", "sram");
    };
    const ecore = (x, y) => {
      unit(x, y, 66, 20, "l2", "sram");
      unit(x, y + 24, 66, 34, "ecore", "logic");
      unit(x, y + 62, 31, 23, "ecore", "logic", 0.2);
      unit(x + 35, y + 62, 31, 23, "ecore", "logic", 0.1);
    };
    const gpu = (x, y) => {
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) unit(x + i * 52, y + j * 60, 48, 40, "gpu", "logic", 0.1 * j);
      unit(x, y + 42, 100, 14, "l2", "sram");
      unit(x, y + 102, 100, 11, "gpu", "sram", 0.25);
    };

    // Package around the die: substrate vias and straight traces from each pad.
    parts.push(`<g class="pkg-art">`);
    const pads = 30;
    for (let k = 0; k < pads; k++) {
      const t = (k + 0.5) / pads * 1000;
      r(t - 0.6, -400, 1.2, 380, `fill="rgba(217,164,65,0.22)"`); r(t - 0.6, 1020, 1.2, 380, `fill="rgba(217,164,65,0.22)"`);
      r(-400, t - 0.6, 380, 1.2, `fill="rgba(217,164,65,0.22)"`); r(1020, t - 0.6, 380, 1.2, `fill="rgba(217,164,65,0.22)"`);
    }
    parts.push(`</g>`);
    // Die body, seal ring, pad ring.
    r(8, 14, 1000, 1000, `fill="rgba(0,0,0,0.55)"`);
    r(0, 0, 1000, 1000, `fill="#0d0f13"`);
    r(0, 0, 1000, 1000, `fill="url(#p-fill)"`);
    r(3, 3, 994, 994, `fill="none" stroke="#c99a45" stroke-width="2.5"`);
    r(12, 12, 976, 976, `fill="none" stroke="rgba(201,154,69,0.4)" stroke-width="1.2"`);
    for (let k = 0; k < pads; k++) {
      const t = (k + 0.5) / pads * 1000;
      for (const [x, y, w, h] of [[t - 9, 18, 18, 16], [t - 9, 966, 18, 16], [18, t - 9, 16, 18], [966, t - 9, 16, 18]]) {
        r(x, y, w, h, `fill="url(#g-pad)"`);
        r(x + 0.5, y + 0.5, w - 1, h - 1, `fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="0.8"`);
      }
    }

    // Memory PHYs top and bottom: repeated driver lanes.
    for (const y of [46, 906]) for (let i = 0; i < 16; i++) unit(60 + i * 55.5, y, 51, 48, "phy", "phy");
    // Performance cluster, its L2, efficiency cluster, its L2, neural engine.
    for (let i = 0; i < 4; i++) parts.push(`<g data-core="${i}">`), pcore(60 + (i % 2) * 148, 112 + Math.floor(i / 2) * 178), parts.push(`</g>`);
    unit(60, 470, 288, 60, "l2", "sram");
    for (let i = 0; i < 4; i++) parts.push(`<g data-core="${4 + i}">`), ecore(60 + i * 74, 540), parts.push(`</g>`);
    unit(60, 630, 288, 34, "l2", "sram");
    for (let i = 0; i < 8; i++) for (let j = 0; j < 4; j++) unit(60 + i * 36.3, 672 + j * 56, 32.3, 52, "npu", j % 2 ? "sram" : "logic", 0.08 * ((i + j) % 2));
    // System cache: banks of SRAM with a controller down the middle.
    for (let j = 0; j < 8; j++) { unit(362, 112 + j * 98, 58, 94, "slc", "sram"); unit(428, 112 + j * 98, 58, 94, "slc", "sram", 0.1); }
    unit(420, 112, 8, 782, "slc", "logic", 0.3);
    // GPU: sixteen identical cores.
    for (let i = 0; i < 16; i++) gpu(502 + (i % 4) * 110, 112 + Math.floor(i / 4) * 121);
    // Media, display and the rest.
    macros(502, 600, 214, 294, "media", 11, 3);
    macros(724, 600, 216, 294, "misc", 29, 3);

    // Light raking across the die.
    r(0, 0, 1000, 1000, `fill="url(#g-sheen)" style="mix-blend-mode:screen"`);
    // Load overlays for the eight cores, drawn above the sheen.
    const cores = [];
    for (let i = 0; i < 4; i++) cores.push([60 + (i % 2) * 148, 112 + Math.floor(i / 2) * 178, 140, 170]);
    for (let i = 0; i < 4; i++) cores.push([60 + i * 74, 540, 66, 85]);
    cores.forEach(([x, y, w, h], i) => {
      parts.push(`<g class="hot" data-hot="${i}" opacity="0">`);
      r(x, y, w, h, `fill="rgba(240,170,60,0.32)"`);
      r(x + 1, y + 1, w - 2, h - 2, `fill="none" stroke="#f0c66b" stroke-width="2.2"`);
      r(x + 6, y + h - 9, w - 12, 3, `fill="#f0c66b" class="bar" data-w="${w - 12}"`);
      parts.push(`</g>`);
    });

    const callouts = labels ? [
      ["Performance cores", -1, 204, 290], ["Efficiency cores", -1, 204, 582], ["Neural Engine", -1, 204, 780],
      ["System cache", 1, 424, 300], ["GPU", 1, 720, 360], ["Media and display", 1, 720, 746],
    ].map(([t, side, cx, cy]) => {
      const ex = side < 0 ? -40 : 1040;
      return `<g class="callout"><line x1="${cx}" y1="${cy}" x2="${ex}" y2="${cy}"/><circle cx="${cx}" cy="${cy}" r="6"/><text x="${ex + side * 14}" y="${cy}" text-anchor="${side < 0 ? "end" : "start"}" dominant-baseline="middle">${t}</text></g>`;
    }).join("") : "";

    const grad = (id, [a, b]) => `<linearGradient id="g-${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${a}"/><stop offset="1" stop-color="${b}"/></linearGradient>`;
    // Circuit textures. SRAM: dense bit-line rows. Logic: standard-cell rows of fixed widths.
    let logic = `<rect width="60" height="0.7" fill="rgba(255,255,255,0.22)"/>`;
    const rl = seq(5);
    for (let row = 0; row < 2; row++) for (let x = 0; x < 60;) {
      const w = 2 + Math.floor(rl() * 6);
      logic += `<rect x="${x}" y="${1.4 + row * 3}" width="${Math.max(1, w - 0.8)}" height="2.2" fill="rgba(255,255,255,${(0.06 + rl() * 0.16).toFixed(2)})"/>`;
      x += w;
    }
    const defs = `<defs>
      ${Object.entries(HUE).map(([k, v]) => grad(k, v)).join("")}
      <linearGradient id="g-pad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e7e2d6"/><stop offset="1" stop-color="#8d8778"/></linearGradient>
      <linearGradient id="g-sheen" x1="0" y1="0" x2="1" y2="1"><stop offset="0.25" stop-color="#fff" stop-opacity="0"/><stop offset="0.45" stop-color="#fff" stop-opacity="0.16"/><stop offset="0.62" stop-color="#fff" stop-opacity="0"/></linearGradient>
      <pattern id="p-sram" width="6" height="2.4" patternUnits="userSpaceOnUse"><rect width="6" height="1" fill="rgba(255,255,255,0.2)"/><rect width="0.6" height="2.4" fill="rgba(0,0,0,0.18)"/></pattern>
      <pattern id="p-logic" width="60" height="6.4" patternUnits="userSpaceOnUse">${logic}</pattern>
      <pattern id="p-phy" width="8.5" height="48" patternUnits="userSpaceOnUse"><rect x="1" y="4" width="4" height="40" fill="rgba(255,255,255,0.14)"/><rect x="1" y="4" width="4" height="6" fill="rgba(255,255,255,0.3)"/></pattern>
      <pattern id="p-fill" width="8" height="8" patternUnits="userSpaceOnUse"><rect x="2" y="2" width="3" height="3" fill="rgba(255,255,255,0.05)"/></pattern>
    </defs>`;
    const vb = labels ? "-430 -60 1860 1120" : "-80 -80 1160 1160";
    return `<svg viewBox="${vb}" xmlns="${NS}" preserveAspectRatio="xMidYMid meet" class="die-art" role="presentation">${defs}${parts.join("")}${callouts}</svg>`;
  }

  // Same interface the hero and proof sections drive: set() a load per core, frame() to apply.
  class Die {
    constructor(host, { labels = true } = {}) {
      this.canvas = host;
      host.innerHTML = dieSVG(labels);
      this.hot = $$(".hot", host).map((g) => ({ g, bar: $(".bar", g) }));
      this.heat = new Array(8).fill(0);
      this.target = new Array(8).fill(0);
      this.frame();
    }
    set(load) { this.target = load.slice(0, 8); }
    frame(dt = 16) {
      this.hot.forEach(({ g, bar }, i) => {
        this.heat[i] += (this.target[i] - this.heat[i]) * Math.min(1, dt / 160);
        const v = this.heat[i];
        g.setAttribute("opacity", Math.min(1, v * 1.4).toFixed(3));
        bar.setAttribute("width", (bar.dataset.w * v).toFixed(1));
      });
    }
  }

  function animate(die, tick) {
    if (reduced) { tick?.(0); die.frame(1000); return; }
    let visible = false, last = performance.now(), raf = 0;
    const loop = (now) => {
      const dt = Math.min(64, now - last); last = now;
      tick?.(dt);
      die.frame(dt);
      raf = visible ? requestAnimationFrame(loop) : 0;
    };
    new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible && !raf) { last = performance.now(); raf = requestAnimationFrame(loop); }
    }).observe(die.canvas);
  }

  // Proof section: the die at rest with every core warm, as at the end of the real run.
  const proofCanvas = $("#proof-die");
  if (proofCanvas) {
    const d = new Die(proofCanvas, { labels: false });
    d.set([0.9, 0.85, 0.8, 0.75, 0.55, 0.5, 0.5, 0.45]);
    d.heat = d.target.slice();
    animate(d);
  }

  // ---- Hero: replays one real, measured run on the die ----
  const heroCanvas = $("#hero-die");
  const panel = $(".run-panel");
  if (heroCanvas && panel) {
    const die = new Die(heroCanvas, { labels: heroCanvas.clientWidth > 520 });
    const typed = $(".typed", panel);
    const phases = $$(".phase", panel);
    const clock = $("[data-clock]", panel);
    const figs = $$("[data-fig]", panel);
    const COMMAND = "python3 heavy_benchmark.py";
    const RUN = 61.0;
    const finalFigs = figs.map((f) => f.textContent);
    let t = 0, step = "typing";

    const show = (state) => {
      panel.dataset.state = state;
      phases.forEach((p, i) => {
        const on = { connect: 0, upload: 1, run: 2, landed: 3 }[state] ?? -1;
        p.classList.toggle("on", i <= on);
        p.classList.toggle("done", i < on || state === "landed");
      });
      figs.forEach((f, i) => { f.textContent = state === "landed" ? finalFigs[i] : "—"; f.classList.toggle("pending", state !== "landed"); });
    };

    if (reduced) {
      typed.textContent = COMMAND;
      clock.textContent = `${RUN.toFixed(1)} s`;
      show("landed");
      die.set([0.9, 0.85, 0.8, 0.75, 0.55, 0.5, 0.5, 0.45]);
      die.heat = die.target.slice();
      animate(die);
    } else {
      show("typing");
      typed.textContent = "";
      const rand = rng(3);
      let flicker = 0;
      animate(die, (dt) => {
        t += dt;
        if (step === "typing") {
          const n = Math.min(COMMAND.length, Math.floor((t - 600) / 55));
          typed.textContent = n > 0 ? COMMAND.slice(0, n) : "";
          if (n >= COMMAND.length && t > 600 + COMMAND.length * 55 + 500) { step = "connect"; t = 0; show("connect"); }
        } else if (step === "connect" && t > 700) { step = "upload"; t = 0; show("upload"); }
        else if (step === "upload" && t > 900) { step = "run"; t = 0; show("run"); }
        else if (step === "run") {
          // 61 seconds replayed in about six: the clock says so beside the figures.
          const s = Math.min(RUN, t / 100);
          clock.textContent = `${s.toFixed(1)} s`;
          flicker += dt;
          if (flicker > 120) {
            flicker = 0;
            die.set([0, 1, 2, 3, 4, 5, 6, 7].map((i) => (i < 4 ? 0.7 + rand() * 0.3 : 0.35 + rand() * 0.45)));
          }
          if (s >= RUN) { step = "landed"; t = 0; show("landed"); die.set([0.35, 0.3, 0.3, 0.25, 0.15, 0.15, 0.1, 0.1]); }
        } else if (step === "landed" && t > 6500) {
          step = "typing"; t = 0; show("typing"); clock.textContent = "0.0 s"; die.set(new Array(8).fill(0));
        }
      });
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
