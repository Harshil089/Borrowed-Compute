---
name: Borrowed Compute
description: Silicon Die. The product borrows a chip, so the apps and the website are cut from one.
colors:
  ground: "#07080a"
  package: "#111316"
  cell: "#16191d"
  raised: "#1c2025"
  ink: "#ece9e2"
  ink-2: "#aeaca6"
  ink-3: "#8b8984"
  rule: "rgba(236,233,226,0.18)"
  rule-2: "rgba(236,233,226,0.09)"
  gold: "#d9a441"
  gold-ink: "#e7b95a"
  gold-press: "#f0c66b"
  on-gold: "#0b0a07"
  gold-wash: "rgba(217,164,65,0.08)"
  gold-rule: "rgba(217,164,65,0.32)"
  ok: "#5ad19a"
  warn: "#f2bd57"
  bad: "#ff7a72"
  day-ground: "#f2f0eb"
  day-package: "#ffffff"
  day-ink: "#121316"
  day-ink-2: "#4a4c52"
  day-ink-3: "#62646a"
  day-gold-ink: "#835a0c"
  day-ok: "#17784b"
  day-bad: "#b3261e"
typography:
  display:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontWeight: 700
    fontStretch: "125%"
    letterSpacing: "-0.035em"
  heading:
    fontFamily: "Archivo"
    fontSize: "15px"
    fontWeight: 650
    fontStretch: "112%"
  label:
    fontFamily: "Archivo"
    fontSize: "10.5px"
    fontWeight: 600
    fontStretch: "125%"
    letterSpacing: "0.16em"
    textTransform: uppercase
  body:
    fontFamily: "Archivo, -apple-system, Segoe UI, sans-serif"
    fontSize: "14px"
    lineHeight: 1.5
  mono:
    fontFamily: "Martian Mono, ui-monospace, Cascadia Mono, monospace"
    fontSize: "12px"
    letterSpacing: "-0.01em"
  figure:
    fontFamily: "Martian Mono"
    fontWeight: 500
    letterSpacing: "-0.05em"
    fontFeature: "tnum"
rounded:
  package: "12px"
  field: "9px"
  control: "999px"
spacing:
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "20px"
  2xl: "26px"
---

# Design System: Borrowed Compute

## Overview

**Creative North Star: "Silicon Die"**

Borrowed Compute borrows the chip in your Mac, so every surface is cut from one: a die-black ground, graphite packages that hold each part of the interface, and bond-wire gold for the one thing to press and for every number the Mac measured. The apps and the website share one token set. The website's hero is a live replica of the Windows app running a real job, so the product is the picture.

Dark is the default and the native light of the world; Daylight reprints the same parts on a pale wafer, with gold darkened to stay readable as text.

The system refuses the developer-tool defaults (glowing terminals, neon accents, gradient blobs, feature-card grids) and decoration that is not a part of the machine.

## Colors

- **Ground** is the window and page background. **Package** is every container; **Cell** fills a ruled part inside one; **Raised** is a switch track or a hover.
- **Ink** ramp: ink for text, ink-2 for secondary text, ink-3 for labels and placeholders. All pass 4.5:1 on package in both themes.
- **Gold** fills the one primary button, the switch when on, the progress of a run's phases and its marker, and the focus ring. **Gold ink** is gold as text: measured figures (time, CPU, cores, memory), codes to type or match, a selected Mac, the destination of a run. **Gold wash** and **gold rule** tint the result of a run, requests to approve and the selected Mac.
- **Ok, warn, bad** are only for state: badges, banners, the online dot.

**The Gold Rule.** Gold means "act here" or "this was measured". Nothing else is gold.

**The One Button Rule.** A package carries at most one filled gold button.

## Typography

Archivo is the only text face; its width axis carries hierarchy. Expanded 125% at 700 for display, the big device monograms (WPC, HMA) and run times' units; 112% for headings; 125% caps with wide tracking for small engraved labels. Body text is Archivo at normal width. Martian Mono is the machine's voice: commands, paths, output, ids, codes and measured figures, with tabular numerals.

## Layout and shape

Packages have 12px corners and a hairline (rule) border with a soft, long shadow; fields have 9px corners; buttons, chips, badges and segmented controls are fully round. Inside a package, cells are separated by rule-2 hairlines rather than nested cards. Spacing runs on 6, 8, 12, 16, 20 and 26px.

The PC app is a header bar, a rail of Macs and a deck: the run (route with phases, folder, command, Run) over the output, with the result docked to the output's right edge. The Mac app is one column on narrow windows and two at 900px and up: live work (requests, the latest run, servers, recent runs) on the left, setup (connect, paired devices, guests, settings) on the right.

## Components

- **Buttons:** transparent with a rule border; hover lifts to the cell fill. Primary is filled gold with dark text. Danger is bad-colored text and border. Links are underlined in gold.
- **Phases:** a hairline track per phase that fills gold when done and flows while active, with a gold marker riding the run's progress.
- **Badges:** Done, Exit N, Cancelled, Refused, Running: round, outlined in their state color with a dot.
- **Result:** the run's time large in gold mono, then a two-column ledger of measured figures.
- **Inputs:** cell fill, rule border, mono text; focus turns the border gold with a soft gold ring.
- **Switch:** raised track, gold when on.

## Motion

Phases fill and flow, the result slides in once when a run ends, requests rise in when they arrive. Everything respects reduced motion.
