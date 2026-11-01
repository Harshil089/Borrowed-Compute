---
name: Borrowed Compute
description: The established language of Stripe and native macOS apps, applied to both desktop apps.
colors:
  bg: "#f6f8fa"
  sidebar: "#eef1f5"
  surface: "#ffffff"
  subtle: "#f6f8fa"
  hover: "#edf0f4"
  border: "#e3e8ee"
  border-strong: "#c1c9d2"
  ink: "#1a1f36"
  ink-2: "#545969"
  ink-3: "#687385"
  accent: "#635bff"
  accent-hover: "#5851ea"
  accent-ink: "#4f46e5"
  accent-soft: "#eeedff"
  ok: "#0e6245"
  ok-soft: "#d7f7c2"
  warn: "#983705"
  warn-soft: "#fcedb9"
  bad: "#c0123c"
  bad-soft: "#ffe7f2"
  dark-bg: "#1b1b1f"
  dark-sidebar: "#232328"
  dark-surface: "#26262b"
  dark-border: "#36363d"
  dark-ink: "#f2f2f5"
  dark-ink-2: "#b4b4bd"
  dark-ink-3: "#9a9aa4"
  dark-accent: "#7a73ff"
typography:
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI Variable Text, Segoe UI, system-ui, sans-serif"
    fontSize: "13.5px"
    lineHeight: 1.5
  page-title:
    fontSize: "22px"
    fontWeight: 600
    letterSpacing: "-0.02em"
  heading:
    fontSize: "15px"
    fontWeight: 600
  label:
    fontSize: "12.5px"
    fontWeight: 500
  mono:
    fontFamily: "ui-monospace, SF Mono, Cascadia Mono, Menlo, Consolas, monospace"
    fontSize: "12.5px"
rounded:
  card: "10px"
  control: "6px"
  pill: "999px"
spacing:
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "18px"
  xl: "22px"
  2xl: "28px"
---

# Design System: Borrowed Compute

## Overview

The owner chose the established design language of best-in-class software over any themed world: Stripe's precision (ink, cool greys, one indigo accent, immaculate type and tables) and the structure of native macOS apps (a sidebar, system fonts, quiet motion). Light is the default; dark matches it role for role.

## Colors

Cool grey window, white cards, Stripe ink text in three steps. Indigo is the only accent: the primary button, the selected item, progress, links and focus. Green, amber and red appear only in status badges and banners, always as tinted pills.

## Typography

System fonts only: SF on the Mac, Segoe UI on Windows, so each app reads as native. Page titles 22px semibold, section headings 15px semibold, labels 12.5px medium, body 13.5px. Commands, paths, output, codes and measured figures use the system monospace.

## Layout

Both apps are a sidebar and a main area. The PC app's sidebar lists Macs (selected one raised on a white tile with its badges and actions) and running servers, with the account, appearance, Mac code, updates and sign out at the bottom. The main area is a page header (New run, PC to Mac, status badge), the run card (folder, command with suggestions, a four-step progress bar, options and the Run button in a grey footer) and the output card (log on the left, measured summary on the right).

The Mac app's sidebar switches pages like System Settings: Overview (requests to approve, the current run, running servers), Activity (recent runs table), Devices (the Mac's code, paired devices), Sharing (guests and invites) and Settings. A count badge on Overview shows pending requests.

## Components

- **Cards** have 10px corners and Stripe's layered shadow with no border.
- **Buttons** are white with a hairline shadow ring; primary is filled indigo with a soft indigo glow; danger is red text. Links are indigo text.
- **Fields** have a 1px ring that turns indigo with a soft focus halo.
- **Badges** are rounded pills: tinted green for done or online, red for failed or refused, indigo for running, grey otherwise.
- **Progress** is four bars that fill indigo, the active one animated.
- **Tables** have a grey header row and hairline row rules; rows tint on hover.
- **Switches and segmented controls** follow macOS.

## Motion

Short and functional: pages fade up 4px when switched, results slide in when a run ends, requests rise in when they arrive, the active progress bar flows. Reduced motion turns all of it off.
