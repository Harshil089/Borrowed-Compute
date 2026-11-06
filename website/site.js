// Borrowed Compute website: theme, header, sliding pills, reveals, the jump-to palette, release
// history, and the hero, which replays one real measured run in a replica of the app.
(() => {
  const root = document.documentElement;
  root.classList.add("js");
  // ?still shows every part settled, for screenshots and reviews.
  const still = location.search.includes("still");
  const reduced = still || matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (still) root.classList.add("still");
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  // ---- Which computer is reading: the hero window wears its frame, Download leads with it ----
  const plat = (navigator.userAgentData?.platform || navigator.platform || navigator.userAgent || "").toLowerCase();
  const os = /mac|iphone|ipad/.test(plat) ? "mac" : /win/.test(plat) ? "windows" : "other";
  root.dataset.os = os;
  const isMac = os === "mac";
  $$("[data-mod]").forEach((k) => { k.textContent = isMac ? "⌘K" : "Ctrl K"; });
  $$("[data-mod-enter]").forEach((k) => { k.textContent = os === "windows" ? "Ctrl ↵" : "⌘↵"; });
  $(`.dl-card[data-os="${os}"]`)?.classList.add("mine");

  // ---- Theme: follows the system until you choose, then remembers ----
  $$(".theme-btn").forEach((btn) => btn.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("bc-theme", next); } catch (_) {}
  }));

  // ---- Header: frosted once the page moves under it ----
  const bar = $("#bar");
  const onScroll = () => bar?.classList.toggle("scrolled", scrollY > 8);
  onScroll();
  addEventListener("scroll", onScroll, { passive: true });

  // ---- One pill slides to the link you point at, and rests on the page you're on ----
  function pill(container, pillEl, { vertical = false } = {}) {
    let shown = false;
    const place = (el, instant) => {
      if (!el || !el.offsetParent) { pillEl.classList.remove("on"); shown = false; return; }
      const c = container.getBoundingClientRect(), r = el.getBoundingClientRect();
      if (instant || !shown || reduced) pillEl.classList.add("jump");
      if (vertical) pillEl.style.transform = `translateY(${r.top - c.top}px)`;
      else { pillEl.style.transform = `translateX(${r.left - c.left}px)`; pillEl.style.width = `${r.width}px`; }
      pillEl.classList.add("on");
      shown = true;
      requestAnimationFrame(() => requestAnimationFrame(() => pillEl.classList.remove("jump")));
    };
    return place;
  }
  const nav = $("#nav"), navPill = $(".nav-pill");
  if (nav && navPill) {
    const place = pill(nav, navPill);
    const current = () => $('a[aria-current="page"]', nav);
    const rest = () => place(current(), false);
    $$("a", nav).forEach((a) => a.addEventListener("pointerenter", () => place(a)));
    nav.addEventListener("pointerleave", rest);
    if (document.fonts?.ready) document.fonts.ready.then(() => place(current(), true)); else place(current(), true);
    addEventListener("resize", () => place(current(), true));
  }

  // ---- Menu on narrow screens ----
  const menu = $(".menu-btn");
  if (bar && menu) {
    const set = (open) => { bar.classList.toggle("open", open); menu.setAttribute("aria-expanded", String(open)); };
    menu.addEventListener("click", () => set(!bar.classList.contains("open")));
    $$("#nav a").forEach((a) => a.addEventListener("click", () => set(false)));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") set(false); });
  }

  // ---- Count-ups: a measured figure rises to its value once ----
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const dp = el.dataset.dp != null ? +el.dataset.dp : (el.dataset.count.split(".")[1] || "").length;
    if (reduced) { el.textContent = target.toFixed(dp); return; }
    const t0 = performance.now(), dur = 1100;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 4);
      el.textContent = (target * e).toFixed(dp);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ---- Reveals: parts settle in once, then stay ----
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add("in");
      $$("[data-on-view]", e.target).forEach(countUp);
      io.unobserve(e.target);
    }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
  $$(".reveal").forEach((el) => (reduced ? el.classList.add("in") : io.observe(el)));

  // ---- Copy buttons ----
  $$("[data-copy]").forEach((btn) => btn.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(btn.dataset.copy); } catch (_) { return; }
    const label = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = label; }, 1600);
  }));

  // ---- Guide index: a pill follows the section you're reading ----
  const toc = $(".toc");
  if (toc) {
    const links = $$("a", toc);
    const place = pill(toc, $(".toc-pill", toc), { vertical: true });
    const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const seen = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        links.forEach((a) => a.classList.remove("on"));
        const a = map.get(e.target.id);
        a?.classList.add("on");
        place(a);
      }
    }, { rootMargin: "-20% 0px -70% 0px" });
    map.forEach((_, id) => { const s = document.getElementById(id); if (s) seen.observe(s); });
  }

  // ---- Releases: versions on Download and the What's new timeline ----
  // Served by /api/releases on this site, which reads the public feed server side.
  const versionSlots = $$("[data-version]");
  const timeline = $("#timeline");
  if (versionSlots.length || timeline) {
    const day = (d) => new Date(d).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
    const unavailable = `<li class="timeline-empty">Release history is unavailable right now. Try again in a minute.</li>`;
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
        if (!releases.length) { timeline.innerHTML = unavailable; return; }
        const latest = new Set();
        timeline.innerHTML = releases.map((r, i) => {
          const first = !latest.has(r.key);
          latest.add(r.key);
          return `
          <li class="rel${first ? " latest" : ""}" data-app="${r.key}" style="animation-delay:${Math.min(i, 8) * 40}ms">
            <div class="rel-when">${day(r.date)}</div>
            <div class="rel-body">
              <div class="rel-head"><span class="chip">${esc(r.app)}</span><b>${esc(r.version)}</b>${first ? '<span class="chip">Latest</span>' : ""}</div>
              <ul>${r.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
            </div>
          </li>`;
        }).join("");
      })
      .catch(() => { if (timeline) timeline.innerHTML = unavailable; });
    const seg = $(".segmented");
    if (seg) {
      const segPill = $(".seg-pill", seg);
      const placeSeg = (btn, instant) => {
        if (instant || reduced) segPill.classList.add("jump");
        segPill.style.width = `${btn.offsetWidth}px`;
        segPill.style.transform = `translateX(${btn.offsetLeft - 3}px)`;
        requestAnimationFrame(() => requestAnimationFrame(() => segPill.classList.remove("jump")));
      };
      const pressed = () => $('[aria-pressed="true"]', seg);
      (document.fonts?.ready || Promise.resolve()).then(() => placeSeg(pressed(), true));
      $$("[data-filter]", seg).forEach((btn) => btn.addEventListener("click", () => {
        $$("[data-filter]", seg).forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        placeSeg(btn);
        $$(".rel").forEach((li) => { li.hidden = btn.dataset.filter !== "all" && li.dataset.app !== btn.dataset.filter; });
      }));
    }
  }

  // ---- Steps: an ink line runs through them as you scroll ----
  const steps = $("[data-steps]");
  if (steps) {
    const items = $$(".step", steps);
    const update = () => {
      const r = steps.getBoundingClientRect(), vh = innerHeight;
      const p = reduced ? 1 : Math.max(0, Math.min(1, (vh * 0.8 - r.top) / (vh * 0.5)));
      steps.style.setProperty("--p", p.toFixed(3));
      items.forEach((el, i) => el.classList.toggle("lit", p >= i / Math.max(1, items.length - 1) - 0.01));
    };
    update();
    addEventListener("scroll", update, { passive: true });
    addEventListener("resize", update);
  }

  // ---- Jump to: ⌘K / Ctrl K, or "/" ----
  const COMMANDS = [
    ...[["Home", "/"], ["How it works", "/how-it-works"], ["Pricing", "/pricing"], ["Download", "/download"], ["What's new", "/changelog"], ["About", "/about"]]
      .map(([label, href]) => ({ group: "Pages", label, href, icon: "i-page" })),
    ...[["Set up", "setup"], ["Run a job", "run"], ["bc.toml", "bc-toml"], ["Dev servers", "servers"], ["Sharing", "guests"], ["Command line", "cli"], ["Security", "security"], ["Updates", "updates"], ["Troubleshooting", "troubleshooting"]]
      .map(([label, id]) => ({ group: "Guide", label, href: `/how-it-works#${id}`, icon: "i-hash" })),
    { group: "Get the app", label: "Download for Mac", href: "/download/mac", icon: "i-apple", sub: "Apple Silicon" },
    { group: "Get the app", label: "Download for Windows", href: "/download/windows", icon: "i-windows", sub: "Windows 10 and 11" },
    { group: "Get the app", label: "Switch between light and dark", icon: "i-sun", run: () => $(".theme-btn")?.click() },
  ];
  let overlay = null, items = [], sel = 0, returnFocus = null;
  function renderPalette(q) {
    const t = q.trim().toLowerCase();
    items = COMMANDS.filter((c) => !t || `${c.group} ${c.label}`.toLowerCase().includes(t));
    sel = Math.min(sel, Math.max(0, items.length - 1));
    let group = null, html = "";
    items.forEach((c, i) => {
      if (c.group !== group) { group = c.group; html += `<li class="group" role="presentation">${esc(group)}</li>`; }
      html += `<li role="option" id="pal-${i}" data-i="${i}" aria-selected="${i === sel}"><svg class="icon" aria-hidden="true"><use href="#${c.icon}"/></svg><span>${esc(c.label)}</span>${c.sub ? `<small>${esc(c.sub)}</small>` : ""}</li>`;
    });
    const list = $("ul", overlay);
    list.innerHTML = html || `<li class="none">Nothing matches.</li>`;
    $("input", overlay).setAttribute("aria-activedescendant", items.length ? `pal-${sel}` : "");
    $('[aria-selected="true"]', list)?.scrollIntoView({ block: "nearest" });
  }
  function choose(c) {
    if (!c) return;
    closePalette();
    if (c.run) c.run(); else location.href = c.href;
  }
  function openPalette(fromKeyboard) {
    if (overlay) return;
    returnFocus = document.activeElement;
    overlay = document.createElement("div");
    overlay.className = `overlay${fromKeyboard ? " instant" : ""}`;
    overlay.innerHTML = `<div class="palette" role="dialog" aria-modal="true" aria-label="Jump to"><input type="text" role="combobox" aria-expanded="true" aria-controls="pal-list" placeholder="Jump to a page or section…" autocomplete="off" spellcheck="false"><ul id="pal-list" role="listbox"></ul></div>`;
    document.body.appendChild(overlay);
    const input = $("input", overlay);
    sel = 0;
    renderPalette("");
    input.focus();
    input.addEventListener("input", () => { sel = 0; renderPalette(input.value); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!items.length) return;
        sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
        renderPalette(input.value);
      } else if (e.key === "Enter") { e.preventDefault(); choose(items[sel]); }
      else if (e.key === "Escape") { e.preventDefault(); closePalette(); }
      else if (e.key === "Tab") e.preventDefault();
    });
    overlay.addEventListener("pointerdown", (e) => { if (e.target === overlay) closePalette(); });
    $("ul", overlay).addEventListener("click", (e) => { const li = e.target.closest("[data-i]"); if (li) choose(items[+li.dataset.i]); });
  }
  function closePalette() {
    overlay?.remove();
    overlay = null;
    returnFocus?.focus?.();
  }
  $$("[data-palette]").forEach((b) => b.addEventListener("click", (e) => openPalette(e.detail === 0)));
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
    if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") {
      e.preventDefault();
      overlay ? closePalette() : openPalette(true);
    } else if (e.key === "/" && !typing && !overlay) {
      e.preventDefault();
      openPalette(true);
    }
  });

  // ---- Hero: one real run, replayed ----
  // The figures are a real CPU-heavy Python benchmark's (61.0 s, 324.5 s of CPU, 5.3 cores,
  // 618 MB). Its 61 seconds play back in about six; the lending computer's cores are shaded
  // the way the app shades them: ink when light, yellow at medium load, red when heavy.
  const stage = $("[data-replay]");
  if (stage) {
    const win = $(".main-win", stage);
    const lend = $(".lend-win", stage);
    const typed = $("[data-typed]", win);
    const legs = $$("[data-leg]", win);
    const clock = $("[data-clock]", win);
    const status = $("[data-status]", win);
    const log = $("[data-log]", win);
    const wire = $(".wire", win);
    const from = $(".node.from", win), to = $(".node.to", win);
    const busy = $("[data-busy]", lend);
    const coreBox = $("[data-cores]", lend);
    const N = 8;
    coreBox.innerHTML = Array.from({ length: N }, () => `<span class="core"><i></i></span>`).join("");
    const cells = $$(".core i", coreBox);
    const sidePill = $(".w-pill", win), onDev = $(".w-dev.on", win);
    const placeSide = () => { if (onDev.offsetParent) sidePill.style.setProperty("--y", `${onDev.offsetTop}px`); };
    placeSide();
    addEventListener("resize", placeSide);

    const COMMAND = "python3 heavy_benchmark.py";
    const RUN = 61.0;
    const LINES = [
      [0.0, '<span class="d">connected   studio · direct · 41 ms</span>'],
      [0.0, '<span class="d">uploaded    18 KB in 0.1 s</span>'],
      [0.0, '<span class="c">$ python3 heavy_benchmark.py</span>'],
      [1.2, "workers 8 · matrix 2048 x 2048"],
      [10.4, "round 1/6   10.2 s"],
      [20.6, "round 2/6   10.1 s"],
      [30.8, "round 3/6   10.2 s"],
      [41.0, "round 4/6   10.1 s"],
      [51.1, "round 5/6   10.1 s"],
      [61.0, "round 6/6    9.9 s"],
      [61.0, '<span class="ok">done · 6 rounds in 61.0 s</span>'],
    ];
    let shownLines = 0;
    const showLog = (secs, animate) => {
      const n = LINES.filter(([t]) => t <= secs).length;
      if (n === shownLines) return;
      if (n < shownLines) { log.innerHTML = ""; shownLines = 0; }
      for (let i = shownLines; i < n; i++) {
        const span = document.createElement("span");
        span.className = `log-line${animate ? " new" : ""}`;
        span.style.display = "block";
        span.innerHTML = LINES[i][1];
        log.appendChild(span);
      }
      shownLines = n;
    };
    const setLeg = (at) => legs.forEach((el, i) => { el.classList.toggle("done", i < at); el.classList.toggle("active", i === at); });
    const heat = (loads) => {
      cells.forEach((c, i) => c.style.setProperty("--l", (0.12 + 0.88 * loads[i]).toFixed(2)));
      const avg = loads.reduce((a, b) => a + b, 0) / loads.length;
      lend.dataset.heat = avg > 0.66 ? "high" : avg > 0.4 ? "mid" : "low";
      lend.toggleAttribute("data-busy", true);
      busy.textContent = `${Math.round(avg * N)} of ${N} busy`;
    };
    const cool = () => {
      cells.forEach((c) => c.style.setProperty("--l", "0"));
      lend.dataset.heat = "low";
      lend.removeAttribute("data-busy");
      busy.textContent = "idle";
    };
    // A believable spread around the real 5.3-core average: most cores pinned during a
    // round, a short dip as each round hands off to the next.
    const loadsAt = (secs) => {
      const warm = Math.min(1, secs / 2.5);
      const inRound = (secs % 10.2) / 10.2;
      const dip = inRound > 0.94 || inRound < 0.03 ? 0.45 : 1;
      return Array.from({ length: N }, (_, i) => {
        const base = i < 6 ? 0.9 : 0.42;
        const wobble = Math.sin(secs * 1.7 + i * 1.3) * 0.08 + (Math.random() - 0.5) * 0.08;
        return Math.max(0, Math.min(1, (base + wobble) * warm * dip));
      });
    };
    const lit = (a, b) => { from.classList.toggle("lit", a); to.classList.toggle("lit", b); };

    const settle = () => {
      win.dataset.state = "landed";
      typed.textContent = COMMAND;
      setLeg(4);
      wire.style.setProperty("--w", "1");
      lit(true, true);
      showLog(RUN, false);
      status.textContent = "Done · exit 0";
      cool();
    };
    const landed = () => {
      settle();
      $$("[data-count]", win).forEach(countUp);
    };

    if (reduced) settle();
    else {
      let t = 0, step = "typing", visible = false, last = 0, raf = 0, lastHeat = 0;
      const start = () => {
        step = "typing"; t = 0; win.dataset.state = "typing"; typed.textContent = "";
        setLeg(-1); clock.textContent = "0.0 s"; log.innerHTML = ""; shownLines = 0;
        status.textContent = "Waiting"; wire.style.setProperty("--w", "0"); wire.classList.remove("up", "back");
        lit(false, false); cool();
      };
      start();
      const tick = (now) => {
        const dt = Math.min(64, now - last); last = now; t += dt;
        if (step === "typing") {
          const n = Math.min(COMMAND.length, Math.floor((t - 600) / 55));
          typed.textContent = n > 0 ? COMMAND.slice(0, n) : "";
          if (n >= COMMAND.length && t > 600 + COMMAND.length * 55 + 500) { step = "pressed"; t = 0; win.dataset.state = "pressed"; }
        } else if (step === "pressed" && t > 180) {
          step = "pack"; t = 0; win.dataset.state = "running"; setLeg(0); lit(true, false); status.textContent = "Packing";
        } else if (step === "pack" && t > 500) {
          step = "upload"; t = 0; setLeg(1); wire.classList.add("up"); wire.style.setProperty("--w", "0.5"); status.textContent = "Uploading";
        } else if (step === "upload" && t > 650) {
          step = "run"; t = 0; setLeg(2); lit(true, true); wire.style.setProperty("--w", "1"); status.textContent = "Running on studio";
        } else if (step === "run") {
          const secs = Math.min(RUN, t / 100);
          clock.textContent = `${secs.toFixed(1)} s`;
          showLog(secs, true);
          if (now - lastHeat > 380) { lastHeat = now; heat(loadsAt(secs)); }
          if (secs >= RUN) { step = "return"; t = 0; setLeg(3); wire.classList.remove("up"); void wire.offsetWidth; wire.classList.add("back"); status.textContent = "Returning"; cool(); }
        } else if (step === "return" && t > 520) { step = "landed"; t = 0; landed(); }
        else if (step === "landed" && t > 7000) start();
        raf = visible ? requestAnimationFrame(tick) : 0;
      };
      new IntersectionObserver(([e]) => {
        visible = e.isIntersecting;
        if (visible && !raf) { last = performance.now(); raf = requestAnimationFrame(tick); }
      }).observe(stage);
    }
  }
})();
