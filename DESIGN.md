---
name: Borrowed Compute
description: Porcelain and ink by day, midnight lacquer by night; Geist type, one ink accent, motion that shows what is happening.
colors:
  bg: "#f4f4f2"
  sidebar: "#ececea"
  surface: "#ffffff"
  subtle: "#f7f7f5"
  ink: "#101116"
  ink-2: "#4a4e58"
  ink-3: "#6b707b"
  accent: "#111216"
  on-accent: "#ffffff"
  ok: "#1d7f49"
  warn: "#9a5a00"
  bad: "#b8302f"
  load-low: "#101116"
  load-mid: "#d9900f"
  load-high: "#d92d24"
  dark-bg: "#0d0e11"
  dark-sidebar: "#15161a"
  dark-surface: "#16171b"
  dark-ink: "#f2f2f4"
  dark-ink-2: "#a8acb6"
  dark-ink-3: "#8a8f9c"
  dark-accent: "#f2f2f4"
  dark-on-accent: "#0d0e11"
typography:
  body:
    fontFamily: "Geist, -apple-system, Segoe UI, system-ui, sans-serif"
    fontSize: "13.5px"
    lineHeight: 1.5
  page-title:
    fontFamily: "Geist"
    fontSize: "28px"
    fontWeight: 500
    letterSpacing: "-0.03em"
  heading:
    fontFamily: "Geist"
    fontSize: "14px"
    fontWeight: 500
  label:
    fontFamily: "Geist"
    fontSize: "12px"
    fontWeight: 500
  figure:
    fontFamily: "Geist"
    fontSize: "26px"
    fontWeight: 300
    letterSpacing: "-0.03em"
  mono:
    fontFamily: "Geist Mono, ui-monospace, SF Mono, Cascadia Mono, monospace"
    fontSize: "12.5px"
rounded:
  card: "14px"
  control: "8px"
  pill: "999px"
spacing:
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "18px"
  xl: "22px"
  2xl: "32px"
---

# Design System: Borrowed Compute

## Overview

The owner chose a precise, quiet language (owner decision, 2026-11-05, replacing the earlier Stripe-indigo look): porcelain surfaces and black ink in light mode, midnight surfaces and near-white ink in dark mode, Geist type, and the structure of native desktop apps (a sidebar, native window frames). Light is the default; dark matches it role for role.

## Colors

Warm-grey window, white cards, ink text in three steps. The accent is ink itself: black in light mode, near-white in dark mode. It marks the primary button, the selected item, progress, switches and focus. Anything filled with the accent carries `--on-accent` for its text or knob, so it stays visible in both modes. There is no purple or indigo anywhere.

Green, amber and red appear only as status: outlined pills and banners.

The live core grid is the one place colour carries load. Each cell's opacity is that core's measured use. The whole grid shifts from ink (light load) to a yellow gradient (medium, average over 40%) to a red gradient (heavy, over 66%), gliding between bands.

## Typography

Geist and Geist Mono, bundled in `ui/fonts/` (SIL OFL, `OFL.txt`), with the system fonts as fallbacks:

- Page titles: 28px medium, tight tracking.
- Section headings: 14px medium.
- Labels: 12px.
- Body: 13.5px.
- Measured figures (run time, CPU time, memory): large and light (300), with a small unit beside them.
- Commands, paths, output and codes: Geist Mono.

## Layout

Both apps are a sidebar and a main area.

**Sidebar:**
- **Use:** devices to run on, with "On this network" below them.
- **Running:** servers you started.
- **This device:** Overview, Activity, Paired devices, Sharing, Settings, with the lending status beside the heading.
- **Foot:** the account and appearance.
- One sliding highlight marks where you are.

**Main area:**
- **New run:** the page header, then the run card. The run card holds the folder and the command with suggestions, a four-step progress bar, then options and the Run button. Below it sits the output card, with the log on the left and the measured summary on the right.
- **Overview:** the lending switch, requests to approve, the current run (with the live core grid) and running servers.

## Components

- **Cards:** 14px corners, a layered soft shadow and a one-pixel top highlight.
- **Buttons:**
  - Quiet grey fills.
  - Primary is ink with a light top edge and a soft glow.
  - Danger is red text.
  - Buttons press in to 97%.
- **Fields:** a soft fill that turns white with an ink focus ring.
- **Badges:** outlined pills; running is a grey fill.
- **Progress:** four thin bars that fill as each step completes.
- **Switches, checkboxes and segmented controls:** follow macOS, filled with the accent.
- **Dialogs and the command palette:** frosted sheets that grow from the top.

## Motion

Short and functional. Every motion answers "what just changed?":

- **The sidebar highlight slides** to the item you picked (220ms).
- **Pages enter** with a fade, an 8px move in the direction you travelled and a slight blur (260ms). The old page fades out in 90ms.
- **The page title** shrinks into the frosted bar once you scroll under it.
- **The Run button** turns into the live run timer, or "Serving" with uptime for a server.
- **Results count up** once when a run ends.
- **Requests rise in** when they arrive. After approval they show "Paired" and fold away.
- **Core cells** fade between measured loads, and the load colour glides between bands.
- **The command palette** (⌘K or Ctrl+K) grows in when clicked.

Anything started from the keyboard switches instantly. Nothing loops for attention. Reduced motion turns all of it off.
