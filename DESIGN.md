---
name: Borrowed Compute
description: A jet-age ticket wallet for runs that fly from your PC to your Mac and back.
colors:
  stock: "#eeefec"
  paper: "#ffffff"
  paper-2: "#f6f6f3"
  carrier-navy: "#0d1b3d"
  bar-ink: "#ffffff"
  bar-ink-2: "#b8c1d9"
  ink: "#0d1b3d"
  ink-2: "#44506d"
  ink-3: "#5f6983"
  rule: "#0d1b3d"
  rule-2: "#cfd3dd"
  jet-red: "#d7262d"
  red-ink: "#c0141c"
  red-press: "#b81d23"
  carbon: "#4b2e83"
  copy: "#ddd3f2"
  copy-rule: "#a48fd0"
  ok: "#1d7044"
  ok-tint: "#e3f1e8"
  warn: "#8a5a00"
  warn-tint: "#f8edd4"
  bad: "#c0141c"
  bad-tint: "#fbe4e4"
  night-stock: "#0b0b0c"
  night-paper: "#161618"
  night-paper-2: "#111113"
  night-bar: "#000000"
  night-bar-ink-2: "#a8a8ad"
  night-ink: "#f3f3f4"
  night-ink-2: "#b8b8bd"
  night-ink-3: "#929298"
  night-rule: "#4a4a50"
  night-rule-2: "#2a2a2e"
  night-red-ink: "#ff6b70"
  night-carbon: "#c4b1f5"
  night-copy: "#1e1a28"
  night-copy-rule: "#5a4f78"
  night-ok: "#5fd39a"
  night-ok-tint: "#0e2318"
  night-warn: "#f2bd57"
  night-warn-tint: "#2a2110"
  night-bad: "#ff6b70"
  night-bad-tint: "#2d1316"
typography:
  display:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "48px"
    fontWeight: 800
    lineHeight: 0.95
    letterSpacing: "0.01em"
  port-code:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "54px"
    fontWeight: 800
    lineHeight: 0.85
    letterSpacing: "0.02em"
  headline:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "26px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.02em"
  title:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "15px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.08em"
  label:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "0.12em"
  button:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "13px"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "0.1em"
  body:
    fontFamily: "Barlow, -apple-system, Segoe UI, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
  mono-output:
    fontFamily: "Martian Mono, ui-monospace, Cascadia Mono, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.65
    letterSpacing: "-0.01em"
  mono-field:
    fontFamily: "Martian Mono, ui-monospace, Cascadia Mono, monospace"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  figure-hero:
    fontFamily: "Martian Mono, ui-monospace, Cascadia Mono, monospace"
    fontSize: "38px"
    fontWeight: 500
    lineHeight: 0.9
    letterSpacing: "-0.05em"
    fontFeature: "tnum"
  code:
    fontFamily: "Martian Mono, ui-monospace, Cascadia Mono, monospace"
    fontSize: "34px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "0.14em"
    fontFeature: "tnum"
rounded:
  control: "3px"
  stamp: "4px"
  coupon: "5px"
  switch: "11px"
spacing:
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "14px"
  xl: "18px"
  2xl: "22px"
components:
  button:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "9px 14px 8px"
  button-hover:
    backgroundColor: "{colors.carrier-navy}"
    textColor: "{colors.bar-ink}"
  button-primary:
    backgroundColor: "{colors.jet-red}"
    textColor: "{colors.bar-ink}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "9px 14px 8px"
  button-primary-hover:
    backgroundColor: "{colors.red-press}"
    textColor: "{colors.bar-ink}"
  button-danger:
    textColor: "{colors.red-ink}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "9px 14px 8px"
  button-danger-hover:
    backgroundColor: "{colors.jet-red}"
    textColor: "{colors.bar-ink}"
  input:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.mono-field}"
    rounded: "{rounded.control}"
    padding: "9px 11px"
  coupon:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.coupon}"
  carbon-stub:
    backgroundColor: "{colors.copy}"
    textColor: "{colors.ink}"
    padding: "16px 20px 14px"
  tab:
    backgroundColor: "{colors.carrier-navy}"
    textColor: "{colors.bar-ink}"
    typography: "{typography.label}"
    padding: "7px 16px 6px"
  tab-red:
    backgroundColor: "{colors.jet-red}"
    textColor: "{colors.bar-ink}"
  pill:
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "4px 7px 3px"
  switch:
    backgroundColor: "{colors.rule-2}"
    rounded: "{rounded.switch}"
    width: "38px"
    height: "22px"
  switch-on:
    backgroundColor: "{colors.jet-red}"
---

# Design System: Borrowed Compute

## Overview

**Creative North Star: "The Jet-Age Ticket Wallet"**

A job is a flight from the PC to the Mac and back, so both windows are printed airline stationery rather than developer chrome. The client is the passenger's wallet: the run is a coupon you fill in, its phases are route legs that stamp themselves as they fly, and the result tears off as a carbon copy with the measured numbers on it. The agent is the station's operations board: boarding requests, departures, the manifest of who is cleared to fly here. The same world serves both because the two apps describe one trip from either end.

The material is paper on a wallet: coupon-white stock with printed carrier-navy hairlines, die-cut corners, chamfered tab heads and perforations where a stub tears off. Colour is spent like ink on a ticket, sparingly and with a job each: navy prints the structure, jet red marks the route and the one thing to press, carbon purple is the pressure-copy ink that every measured figure and every code you must match is struck in. Dark mode is the same stationery printed on black with white ink, not an inverted theme. Light is the default; the Appearance setting (Light, Dark, System) picks the theme, and System follows the operating system.

Density is operational: the client is used many times a day and must stay keyboard-first, the agent is glanced at and must read in one look. Big condensed codes (WPC, HMA, MAC) carry identity at a distance; everything else stays small, caps-set and ruled. The system refuses the default dev-tool look of graphite panels, neon accents and a generic sidebar over a console, and the default settings look of stacked grey cards with toggles.

**Key Characteristics:**
- Printed stationery: hairline navy rules, die-cut and chamfered corners, perforated stubs, double-ruled rubber stamps.
- Three inks with fixed jobs: carrier navy for structure, jet red for route and action, carbon purple for figures and codes.
- Condensed airline caps for every label, code and button; Barlow for sentences; Martian Mono for anything the machine produced or the user must type exactly.
- A night-flight dark mode that keeps every role and only reprints the stock.
- Motion that tells the trip: legs march, the jet rides the route, the stub tears off, a boarding pass arrives.

## Colors

Coupon paper and carrier navy carry nearly every surface; jet red and carbon purple are the two inks that mean something, and the status greens, ambers and reds are kept for state alone.

### Primary
- **Jet Red** (jet-red): the livery colour. Fills the one primary button on a coupon, the route line and its leg dots, the jet, the current run tab, the switch when on, and the rule under the navy bar. As a line it is also the focus ring, the caret and the underline of text links, so the user's place in the window is always drawn in the carrier's red.
- **Red Ink** (red-ink; night-red-ink in dark): the text weight of jet red, darkened for 4.5:1 on paper. Used for the triple-slash carrier mark in headings, the selected Mac's code, danger buttons, the VOID stamp, the live pill and stderr lines in the flight log.
- **Red Press** (red-press): the pressed and hovered fill of the primary button.

### Secondary
- **Carbon Purple** (carbon; night-carbon in dark): pressure-copy ink. Every measured figure (run time, CPU, memory, cores, connect, upload, first output), every code the user reads back or matches (sign-in code, pairing code, manifest device codes) and the Mac's address are set in it. It is also the border of a focused text field.
- **Carbon Copy** (copy; night-copy in dark): the lavender of the copy sheet. Fills the result stub, the sign-in and pairing stubs, the boarding pass's code stub, the address strip, the field focus halo and text selection.
- **Copy Rule** (copy-rule; night-copy-rule in dark): the dashed tear line and the ruled lines of the ledger printed on the copy.

### Neutral
- **Wallet Stock** (stock; night-stock): the window background behind the coupons.
- **Coupon Paper** (paper; night-paper): every coupon, field and default button.
- **Ruled Cell** (paper-2; night-paper-2): the run coupon's footer and the live row on the departures board.
- **Carrier Navy** (carrier-navy, rule; night-bar, night-rule): the top bar, the departures board head, tab heads and the 1.5px printed outline of coupons, fields and buttons. In light mode ink and rule are the same navy because they are the same printing plate.
- **Bar Ink** (bar-ink, bar-ink-2; night-bar-ink-2): white and pale blue text on the navy bar.
- **Ink ramp** (ink, ink-2, ink-3 and their night versions): primary text, secondary text and labels, and quiet placeholders. Ink-3 still clears 4.5:1 on paper in both modes.
- **Inner Rule** (rule-2; night-rule-2): cell dividers, row rules, the rail's edge and idle borders.

### Status
- **Ok, Warn, Bad** with their tints, in both modes: only for state. They colour pills, stamps and banners, never decoration. Bad and red-ink share a value in light mode, which is deliberate: a failed exit and a void are both struck in red.

### Named Rules
**The Three Inks Rule.** Navy prints structure, jet red draws the route and the action, carbon prints figures and codes. A new element takes one of these jobs or stays navy.

**The One Red Button Rule.** A coupon carries at most one filled jet-red button: Run on Mac, Sign in with GitHub, Save, Install and restart, Approve. Everything else is an outlined paper button or a text link. A boarding pass repeats Approve only because each request is its own coupon.

**The Carbon Ink Rule.** Carbon purple is for what was measured or must be matched exactly. If a value did not come off the Mac, the clock or a code, it is not set in carbon.

**The Night Flight Rule.** Dark mode reprints the same roles on black stock. Never invent a dark-only colour role; change the ink, keep the job.

## Typography

**Display Font:** Barlow Condensed (with Arial Narrow)
**Body Font:** Barlow (with -apple-system, Segoe UI)
**Label/Mono Font:** Martian Mono (with ui-monospace, Cascadia Mono)

**Character:** Barlow Condensed in caps is the airline's printed voice: codes, labels, tabs, headings and buttons, all tracked wide. Barlow is the same family set plainly for sentences, so the two never fight. Martian Mono is the machine's voice and appears only where the computer produced the text or the user must type it exactly. All three are self-hosted woff2 in each app's ui/fonts because the CSP allows styles only from the app itself.

### Hierarchy
- **Port Code** (800, 54px, 0.85): the three-letter airport codes on the route (WPC, HMA). The same face sets the station code (58px), the gate codes in the Mac rail (30px), the agent's arrivals route (40px), the stub route (46px) and the manifest codes (700, 24px, in carbon). Codes are derived from device names so they stay stable.
- **Display** (800, 48px, 0.95, caps, balanced): the check-in headline on the signed-out pass; 40px in the narrower agent.
- **Headline** (700, 26px, 1, caps): dialog titles.
- **Title** (700, 15px, 1, 0.08em, caps): section headings such as Your Macs, Flight log, Departures, Manifest, Route, Station. Each is prefixed in the same line by the carrier mark, a red triple slash.
- **Body** (400, 14px, 1.45): sentences, help text and Mac names; 12.5px to 13px for hints and secondary lines. The pass lede holds to 52ch.
- **Label** (600, 11px, 0.12em, caps): every printed field label, tab head, pill and ledger term.
- **Button** (600, 13px, 0.1em, caps): every button; 15px on the wide sign-in and the Run button.
- **Mono Field** (400, 13px, 1.3): text inputs, because they hold paths, commands and addresses.
- **Mono Output** (400, 12px, 1.65): the flight log, departures commands and timestamps, device ids.
- **Figure Hero** (500, 38px, 0.9, -0.05em, tabular): the run time on the carbon copy. Smaller figures use the same face at 11.5px to 12px, with units set small in condensed caps beside them.
- **Code** (500, 34px, 0.14em, tabular): sign-in and pairing codes; 46px in the pairing dialog, 28px on a boarding pass.

### Named Rules
**The Machine Voice Rule.** Martian Mono only for output, paths, commands, addresses, ids, codes and measured figures. Prose, names and labels never go mono, which is why the PC and Mac name fields reset to Barlow.

**The Printed Caps Rule.** Anything that would be printed on a ticket (labels, codes, tabs, statuses, buttons) is Barlow Condensed in caps with positive tracking. Sentences are never set in caps.

**The Small Unit Rule.** A figure's unit (ms, s, MB) is set small in condensed caps right after the carbon number, so the number reads first.

## Layout

The client is a fixed frame: a 54px navy bar across the top, a 252px rail of Macs on the left, and a deck that scrolls. The deck stacks the run coupon (fixed height) over the flight log, which takes the remaining height; the carbon-copy result docks as a 330px stub on the log's right edge. Below 1000px the stub drops beneath the console; below 880px the rail narrows to 212px, the account name leaves the bar, port codes shrink to 40px and the folder and command cells stack. The 760 by 520 minimum window still shows the whole run coupon.

The agent is a single narrow column under a sticky 52px bar: station strip, then boarding requests (only when pending), departures, manifest, route and station settings, with a quiet footer. When the Mac has no route for the PC yet, the route block moves above everything but the station strip, because it is the first thing the window needs.

Spacing runs on small, tight steps (6, 8, 12, 14, 18, 22px): 22 to 24px around the deck, 14 to 22px inside coupon cells, 8 to 12px between related controls. Rows on the boards are ruled, not spaced apart.

## Elevation & Depth

Depth is paper on a wallet: coupons sit one sheet above the stock with a soft, short, navy-tinted shadow and a printed navy outline. Nothing floats higher except dialogs, which sit on a navy dim. Die-cut coupons cannot cast their own shadow through the clip, so their container casts it with a drop-shadow filter instead.

### Shadow Vocabulary
- **Coupon** (`box-shadow: 0 1px 1px rgba(13,27,61,0.1), 0 6px 10px -8px rgba(13,27,61,0.35)`; night: `0 1px 1px rgba(0,0,0,0.35), 0 6px 10px -8px rgba(0,0,0,0.8)`): every coupon, the selected Mac and dialogs.
- **Die-cut container** (`filter: drop-shadow(0 1px 1px rgba(13,27,61,0.1)) drop-shadow(0 4px 5px rgba(13,27,61,0.14))`): the parent of any die-cut coupon.
- **Primary button** (`box-shadow: 0 1px 0 rgba(255,255,255,0.22) inset, 0 2px 5px -2px rgba(4,10,30,0.45)`): the slight emboss of the one red button.
- **Overlay dim** (`rgba(4,10,30,0.55)`): behind dialogs and ask boxes.

### Named Rules
**The One Sheet Rule.** Coupons are one sheet thick. There is no stacked elevation scale; a selected item rises only to coupon level.

## Shapes

Coupons keep small printed rounded corners (5px), because clipping them would cut away the navy border and shadow that define the sheet. Controls are tighter (3px), stamps 4px, and only the switch is fully round. Ticket geometry comes from cuts instead of radius: the run coupon and the departures board have their top-right corner die-cut at 16px with a navy edge printed along the cut; tab heads have both top corners chamfered at 7px; perforations are half-circle notches punched into both edges of a dashed tear line. Every outline is 1.5px, and stamps use a 2.5px double rule.

## Components

### Buttons
Printed and firm, set in condensed caps.
- **Shape:** tight printed corners (3px) with a 1.5px navy outline.
- **Default:** coupon paper with navy text; hover fills carrier navy with white text, so every hover is the same inversion. Pressing nudges it down 1px.
- **Primary:** filled jet red with white text and a faint emboss; hover and press go to red press. One per coupon.
- **Danger:** red-ink outline and text on transparent; hover fills jet red. Used for Cancel run and Remove.
- **Link:** no box, condensed caps with a 1.5px jet-red underline; hover turns the text red. Used for Clear, Open log, Logs, Stop sharing and Sign out.
- **Bar link:** white caps on the navy bar, a red underline appears on hover.
- **Disabled:** 45% opacity.

### Inputs / Fields
- **Style:** coupon paper, 1.5px navy outline, 3px corners, Martian Mono 13px, ink-3 placeholder. A printed caps label sits above each field.
- **Focus:** the border turns carbon and a 3px carbon-copy halo appears, as if pressed through to the copy.
- **Name fields** reset to Barlow 14px because a name is not machine text.

### Switch
A real switch for settings that take effect at once (Run in background): 38 by 22px, inner-rule track with a navy outline; on, the track fills jet red and the knob slides 16px. Focus draws the red ring on the track.

### Pills
Status printed in a box, like ON TIME on a departures sheet: condensed caps 11px in a 1.5px outline of the text's own colour, 3px corners. Ok, warn and bad take their status colours; live is red ink with a pulsing beacon dot. On the navy bar the pill uses brighter status colours to hold contrast.

### Stamps
Rubber stamps struck onto the carbon copy: condensed caps 15px, 2.5px double rule, 4px corners, rotated -6 degrees. Ok for Landed, bad for Exit N, Timed out and Refused, red ink for Void, quiet for everything else.

### Coupon
The one container. Paper, 1.5px navy outline, 5px corners, coupon shadow. Interiors are divided by inner rules into cells rather than nested cards.

### Die-cut corner
The top-right corner cut off at 16px with a 2px navy line printed along the cut. Used on the run coupon, the departures board and boarding passes, where the sheet should read as a ticket rather than a panel.

### Chamfered tab heads
Solid navy tabs with both top corners chamfered at 7px, sitting on the coupon's top edge like FLIGHT and DATE heads on a ticket. The run coupon carries two: the run state (jet red) and the date (navy).

### Perforation
A dashed inner-rule tear line with half-circle notches punched through both edges in the colour behind the coupon. Separates a dialog's body from its carbon stub (the pairing code).

### Carbon-copy stub
The lavender copy sheet, separated from its coupon by a dashed copy-rule edge. On the client it is the result: run time huge in carbon, the stamp, the message, door-to-door time and route (direct or relay), and a two-pairs-per-row ledger of CPU, cores, memory, upload, connect and first output. It is also the sign-in stub, the pairing stub and the code half of a boarding pass.

### Route legs
The client's route header: FROM and TO port codes joined by a straight jet-red flight path in four equal legs (Pack, Upload, Run, Return). Straight rather than arced so the four labels align and progress reads at a glance. Unflown legs are dashed at half strength; the active leg's dashes march; flown legs go solid with filled dots. A jet rides the path at the run's progress. Idle, the jet is ink-3; on failure the active leg and jet turn bad.

### Arrivals
The agent's version of the route, shared with the client through base.css: FROM the sending PC's code TO this Mac's code, in three legs (Upload, Run, Return). The agent hears of a run once its upload is in, so a live run shows Upload flown and Run marching, with a ticking clock in carbon; a landed run flies every leg and tears a carbon-copy stub down beneath it (run time, CPU, cores, memory, and the stamp). A failed run stops on Run in bad. With no runs yet the route waits, dashed and idle. It sits below any boarding request, because an approval outranks a report.

### Boarding pass
A pending pairing request on the agent: the requesting device's name in 26px condensed caps on the left, its code in carbon on the copy stub at right with the red Approve beneath. It arrives by rising 8px and fading in once.

### Departures board and manifest
Ruled rows under a navy head (Time, Run, Status): time in mono, the command in mono, a sub-line of condensed caps terms with carbon figures, and a status pill. The live row sits on ruled-cell fill. The manifest lists cleared devices with their carbon code, name, a mono id and a danger Remove.

### Navigation
The navy bar is the only navigation: wordmark with the dashed-trail jet mark on the left, account and bar links on the right, and a 3px jet-red rule along its bottom edge. The agent's bar carries the station status pill (Open for runs, Signed out, Closed) instead of links.

### State vocabulary
States are named as a flight would be. The run tab reads **Next run** when idle, **In flight** while running, and takes the stamp's word when the run ends. Outcomes are **Landed** (exit 0), **Exit N** (non-zero, with the code), **Timed out**, **Void** (cancelled), **Refused** (an unpaired device), **Failed**, and **Lost contact** when the agent never heard the end. In flight is the live pill; Landed is ok; Exit, Timed out, Refused and Failed are bad; Void is red ink on the client's stamp and quiet on the agent's board.

### Motion
One easing (cubic-bezier(0.16, 1, 0.3, 1)), short durations, and only motion that tells the trip. Buttons and fields change in 0.16s. The jet glides between legs in 0.9s and the active leg's dashes march every 0.6s. The result stub tears in from 22px to the right over 0.55s. A boarding pass rises in over 0.5s once; it is marked seen so later polls never replay it. The live beacon pulses every 1.4s. Under reduced motion every animation and transition collapses to an instant.

### Named Rules
**The Render-on-Change Rule.** The agent polls every five seconds, so it rebuilds a region only when its rendered markup has actually changed. A poll that finds nothing new must not touch the DOM, which is what keeps focus, hover and a half-pressed Approve stable.

**The Story Motion Rule.** Animate only what moves in the trip: the jet, the legs, the stub, the arriving pass, the live beacon. Nothing else enters, bounces or parallaxes.

## Do's and Don'ts

### Do:
- **Do** build every new surface out of coupons with 1.5px navy outlines and 5px corners, divided into ruled cells.
- **Do** keep one filled jet-red button per coupon and make every other action a paper button or a red-underlined link.
- **Do** set measured figures and codes to be matched in carbon Martian Mono with tabular numerals, and the unit small in condensed caps.
- **Do** label fields, tabs and statuses in Barlow Condensed caps at 11px with 0.12em tracking.
- **Do** prefix section headings with the red triple-slash carrier mark in the same line.
- **Do** name states in the flight vocabulary: Next run, In flight, Landed, Exit N, Void, Refused, Lost contact.
- **Do** keep base.css identical in both apps and check every text colour at 4.5:1 in light and dark.
- **Do** re-render polled regions only when their markup changes, and respect reduced motion.

### Don't:
- **Don't** use jet red as decoration or as a second filled button on the same coupon.
- **Don't** set prose, names or labels in Martian Mono, or set a non-measured value in carbon.
- **Don't** put a small caps label line above a section heading; the heading and its carrier mark are the whole head.
- **Don't** nest coupons inside coupons or add a second elevation level; divide with inner rules instead.
- **Don't** clip a coupon's rounded corners away except with the documented die-cut, which brings its own printed edge and container shadow.
- **Don't** reach for the dev-tool defaults this world replaced: graphite panels, neon accents, stacked grey settings cards.
- **Don't** load fonts or styles from anywhere but the app's own ui folder; the CSP allows styles only from the app.
