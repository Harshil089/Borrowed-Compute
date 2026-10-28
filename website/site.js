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

  // ---- Live versions on the download page, from the public releases feed ----
  const versionSlots = $$("[data-version]");
  if (versionSlots.length) {
    fetch("https://api.github.com/repos/Harshil089/Borrowed-Compute-releases/releases?per_page=50")
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => {
        for (const slot of versionSlots) {
          // The newest by version number, whatever order the feed lists them in.
          const ver = (r) => r.tag_name.replace(/^.*-v/, "").split(".").map(Number);
          const newer = (a, b) => { const x = ver(a), y = ver(b); for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] > y[i]; return false; };
          const rel = list.filter((r) => r.tag_name.startsWith(`${slot.dataset.version}-v`) && !r.draft && !r.prerelease)
            .reduce((best, r) => (!best || newer(r, best) ? r : best), null);
          if (!rel) continue;
          slot.textContent = rel.tag_name.replace(/^.*-v/, "");
          const when = slot.parentElement.querySelector("[data-released]");
          if (when) when.textContent = new Date(rel.published_at).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
        }
      })
      .catch(() => {});
  }

  // ---- The die ----
  function rng(seed) {
    return () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  }
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  // The die is rendered once as a photographic micrograph by die-worker.js and shared by
  // every canvas on the page. Floorplan in die units (0 to 1), loosely after an Apple Silicon
  // layout: memory PHYs along two edges, a performance and an efficiency cluster with their
  // L2, the neural engine, the system cache, the GPU and the media blocks. Not to scale.
  const GOLD = [217, 164, 65];
  function plan() {
    const b = [];
    b.push({ m: "phy", f: 0.36, x: 0.06, y: 0.045, w: 0.88, h: 0.05 });
    b.push({ m: "phy", f: 0.36, x: 0.06, y: 0.905, w: 0.88, h: 0.05 });
    for (let i = 0; i < 4; i++) b.push({ m: "pcore", f: 0.02, core: i, x: 0.06 + (i % 2) * 0.148, y: 0.115 + Math.floor(i / 2) * 0.178, w: 0.14, h: 0.17 });
    b.push({ m: "sram", f: 0.48, x: 0.06, y: 0.475, w: 0.288, h: 0.06 });
    for (let i = 0; i < 4; i++) b.push({ m: "ecore", f: 0.12, core: 4 + i, x: 0.06 + i * 0.0735, y: 0.545, w: 0.066, h: 0.085 });
    b.push({ m: "sram", f: 0.52, x: 0.06, y: 0.64, w: 0.288, h: 0.035 });
    b.push({ m: "npu", f: 0.78, x: 0.06, y: 0.685, w: 0.288, h: 0.205 });
    b.push({ m: "sram", f: 0.62, x: 0.36, y: 0.115, w: 0.13, h: 0.775 });
    for (let i = 0; i < 16; i++) b.push({ m: "gpu", f: 0.24, x: 0.505 + (i % 4) * 0.1095, y: 0.115 + Math.floor(i / 4) * 0.122, w: 0.1, h: 0.113 });
    b.push({ m: "logic", f: 0.4, x: 0.505, y: 0.605, w: 0.215, h: 0.285 });
    b.push({ m: "logic", f: 0.9, x: 0.73, y: 0.605, w: 0.21, h: 0.285 });
    return b;
  }
  const CALLOUTS = [
    { text: "Performance cores", side: -1, x: 0.06, y: 0.2, w: 0.288, h: 0.27 },
    { text: "Efficiency cores", side: -1, x: 0.06, y: 0.545, w: 0.288, h: 0.085 },
    { text: "Neural Engine", side: -1, x: 0.06, y: 0.685, w: 0.288, h: 0.205 },
    { text: "System cache", side: 1, x: 0.36, y: 0.115, w: 0.13, h: 0.3 },
    { text: "GPU", side: 1, x: 0.505, y: 0.237, w: 0.43, h: 0.24 },
    { text: "Media and display", side: 1, x: 0.505, y: 0.605, w: 0.435, h: 0.285 },
  ];
  const BLOCKS = plan();

  // One render for the whole page, at a size that stays sharp on a large retina screen.
  let micrograph = null;
  function getMicrograph() {
    if (!micrograph) micrograph = new Promise((resolve) => {
      const size = 1400;
      const done = (px) => {
        const c = document.createElement("canvas");
        c.width = c.height = size;
        c.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(px), size, size), 0, 0);
        resolve(c);
      };
      try {
        const w = new Worker("/die-worker.js");
        w.onmessage = (e) => { done(e.data.px); w.terminate(); };
        w.postMessage({ size, blocks: BLOCKS });
      } catch (_) { resolve(null); }
    });
    return micrograph;
  }

  class Die {
    constructor(canvas, { labels = true } = {}) {
      this.canvas = canvas;
      this.labels = labels;
      this.heat = new Array(8).fill(0);
      this.target = new Array(8).fill(0);
      this.img = null;
      this.fade = 0;
      getMicrograph().then((img) => { this.img = img; if (reduced) this.fade = 1; this.frame(reduced ? 0 : 16); });
      this.resize();
      new ResizeObserver(() => this.resize()).observe(canvas);
    }
    resize() {
      const r = this.canvas.getBoundingClientRect();
      if (!r.width) return;
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.W = this.canvas.width = Math.round(r.width * this.dpr);
      this.H = this.canvas.height = Math.round(r.height * this.dpr);
      const room = this.labels && r.width > 560 ? 0.56 : 0.84;
      const s = Math.round(Math.min(this.W * room, this.H * 0.84));
      this.die = { x: Math.round((this.W - s) / 2), y: Math.round((this.H - s) / 2), s };
      this.frame();
    }
    box(b) {
      const { die } = this;
      const x = Math.round(die.x + b.x * die.s), y = Math.round(die.y + b.y * die.s);
      return [x, y, Math.round(die.x + (b.x + b.w) * die.s) - x, Math.round(die.y + (b.y + b.h) * die.s) - y];
    }
    set(load) { this.target = load.slice(0, 8); }
    frame(dt = 16) {
      if (!this.die) return;
      const g = this.canvas.getContext("2d");
      const { W, H, die, dpr } = this;
      const u = Math.max(1, Math.round(dpr));
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.clearRect(0, 0, W, H);

      // Package substrate, and straight traces from each pad out to the package edge.
      g.fillStyle = "rgba(236,233,226,0.06)";
      const pitch = Math.round(16 * dpr);
      for (let x = pitch / 2; x < W; x += pitch) for (let y = pitch / 2; y < H; y += pitch) g.fillRect(Math.round(x), Math.round(y), u, u);
      g.fillStyle = "rgba(217,164,65,0.2)";
      for (let k = 0; k < 34; k++) {
        const t = Math.round(die.x + ((k + 0.5) / 34) * die.s), v = Math.round(die.y + ((k + 0.5) / 34) * die.s);
        g.fillRect(t, 0, u, die.y); g.fillRect(t, die.y + die.s, u, H - die.y - die.s);
        g.fillRect(0, v, die.x, u); g.fillRect(die.x + die.s, v, W - die.x - die.s, u);
      }
      // Shadow under the die, then the die.
      g.fillStyle = "rgba(0,0,0,0.6)";
      g.fillRect(die.x + 6 * u, die.y + 10 * u, die.s, die.s);
      if (this.img) {
        this.fade = Math.min(1, this.fade + dt / 500);
        g.fillStyle = "#0c0d10"; g.fillRect(die.x, die.y, die.s, die.s);
        g.globalAlpha = this.fade;
        g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
        g.drawImage(this.img, die.x, die.y, die.s, die.s);
        g.globalAlpha = 1;
      } else {
        g.fillStyle = "#0c0d10"; g.fillRect(die.x, die.y, die.s, die.s);
      }

      // A working core warms to gold: a lit fill, a crisp outline and a load bar.
      for (let i = 0; i < 8; i++) this.heat[i] += (this.target[i] - this.heat[i]) * Math.min(1, dt / 160);
      for (const b of BLOCKS) {
        if (b.core === undefined) continue;
        const v = this.heat[b.core];
        if (v < 0.02) continue;
        const [x, y, w, h] = this.box(b);
        g.globalCompositeOperation = "screen";
        g.fillStyle = `rgba(217,150,50,${0.38 * v})`;
        g.fillRect(x, y, w, h);
        g.globalCompositeOperation = "source-over";
        g.fillStyle = `rgba(240,198,107,${Math.min(1, 0.35 + v)})`;
        g.fillRect(x, y, w, u); g.fillRect(x, y + h - u, w, u); g.fillRect(x, y, u, h); g.fillRect(x + w - u, y, u, h);
        g.fillStyle = "rgba(240,198,107,0.95)";
        g.fillRect(x + 3 * u, y + h - 5 * u, Math.round((w - 6 * u) * v), 2 * u);
      }

      // Callouts, like an annotated die shot in a keynote: a dot on the block, a hairline
      // out past the die edge, and the name.
      if (this.labels && W - die.s > 250 * dpr) {
        const size = Math.round(11 * dpr);
        g.font = `600 ${size}px Archivo, sans-serif`;
        if ("letterSpacing" in g) g.letterSpacing = `${(0.6 * dpr).toFixed(1)}px`;
        for (const c of CALLOUTS) {
          const cx = Math.round(die.x + (c.x + c.w / 2) * die.s), cy = Math.round(die.y + (c.y + c.h / 2) * die.s);
          const ex = c.side < 0 ? die.x - 14 * dpr : die.x + die.s + 14 * dpr;
          g.fillStyle = "rgba(236,233,226,0.55)";
          g.fillRect(Math.min(cx, ex), cy, Math.abs(ex - cx), u);
          g.fillStyle = "#f0c66b";
          g.beginPath(); g.arc(cx, cy + u / 2, 3 * dpr, 0, Math.PI * 2); g.fill();
          g.fillStyle = "rgba(236,233,226,0.92)";
          g.textBaseline = "middle";
          g.textAlign = c.side < 0 ? "right" : "left";
          g.fillText(c.text, ex + c.side * 8 * dpr, cy + u);
        }
        g.textAlign = "left";
      }
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
    const die = new Die(heroCanvas);
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
    window.addEventListener("themechange", () => die.frame());
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
