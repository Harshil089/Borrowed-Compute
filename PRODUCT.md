# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

Desktop apps built with Tauri 2: each app's UI is plain HTML, CSS and JS in `ui/`, rendered by the system web view (WKWebView on macOS, WebView2 on Windows). No bundler, no framework.

## Users

A solo developer who works on a Windows PC and also owns a Mac with Apple Silicon (often a MacBook at home or work). They want their builds, test suites, scripts and small ML training runs to execute on the Mac's chip instead of the PC, without remote desktop, SSH or server setup. Teams sharing Macs are a later audience; nothing should block them, but nothing is designed for them yet.

## Product Purpose

Borrowed Compute runs code jobs from a Windows PC on the user's own paired Mac. The PC sends the project source plus a job spec; the Mac runs its native arm64 toolchain in a throwaway workspace and streams output back live. Success: the developer forgets the job ran on another machine, except that it was faster.

## Positioning

Not pixels (remote desktop, cloud streaming) and not infrastructure (SSH build farms, Docker, Kubernetes). Account-based, zero-setup borrowing of your own Mac's silicon: install on both, sign in with the same GitHub account, pair once, run. The first-minute promise is **speed: your Mac is right here**. Jobs start fast, output streams live, and the apps show real numbers for time and compute used.

## Operating Context

- Two apps:
  - **Borrowed Compute** (Windows client; also runs on macOS): pick a paired Mac, pick a project folder, type a command or use the folder's `bc.toml`, watch live output, see timings (connect, upload, first output, total, direct vs relay) and what the Mac used (CPU time, cores, peak memory, run time). Each run's log is saved locally.
  - **Borrowed Compute Agent** (Mac): sign in, share the built-in cloud through Tailscale so the PC can reach it, approve pairing requests by matching a 6-digit code, see paired devices, recent jobs with compute usage, logs, Run in background, and updates.
- A `bc` command-line tool does the same as the client app; both share one device identity.
- Sign-in is GitHub device flow: the app shows a code, the user enters it on github.com.
- Jobs are batch only: no remote shells, debuggers or language servers.
- Typical jobs: `cargo test`, `npm run build`, `python train.py`, benchmarks lasting seconds to an hour.

## Capabilities and Constraints

- Jobs are `argv` arrays, never shell strings.
- Nothing from a job stays on the Mac: the workspace is deleted after every job, and job output is never written on the Mac.
- Only devices on the same account can pair, and the Mac owner approves each one.
- Connection is direct when possible, otherwise through an encrypted relay; the UI reports which.
- Updates: both apps check a public releases feed and install signed updates in place.
- The Mac app is a normal Dock app; with Run in background on, quitting only hides it and a menu-bar icon remains.
- Phase two (not built): sandbox user plus Seatbelt profile, toolchain provisioning, chunked sync, declared outputs coming back.
- Undecided: pricing, team/admin features, start at login, Developer ID signing.

## Brand Commitments

None fixed. The product name "Borrowed Compute", the app names, and the current icons (`design/icon-client.svg`, `design/icon-agent.svg`, `design/tray.svg`) may all be revisited.

## Evidence on Hand

- Real measured run: a CPU-heavy Python benchmark used 324.5 s of CPU across 5.3 cores, peak memory 618 MB, in 61.0 s on the Mac.
- No testimonials, customers, benchmarks against competitors, or press exist. Do not fabricate any.

## Product Principles

1. **Speed is the product.** Every screen should make the Mac feel close: show progress as it happens and real numbers when it ends.
2. **Truthful numbers.** Times, usage and connection type come from measurement; never decorate or estimate them.
3. **One obvious next step.** Sign in, pair, run: each state of each app has a single clear action.
4. **The Mac stays the owner's.** Approvals, unpairing and what ran are always visible and reversible on the Mac.
5. **Developer-native.** Commands, output and paths are first-class content, set the way developers read them.

## Accessibility & Inclusion

Text colors pass WCAG AA (4.5:1) in light and dark mode. Both apps follow the system light/dark setting. Every action is reachable by keyboard.
