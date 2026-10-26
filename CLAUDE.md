# Borrowed Compute

Run code jobs (builds, tests, scripts, small ML training) on a paired Mac's Apple Silicon from a Windows PC. The PC sends **source plus a job spec**; the Mac runs its native arm64 toolchain and streams output back. No Windows binaries ever run on the Mac.

Product and design docs live outside this repo, in the parent folder `../`: `Borrowed Compute Foundation.md` (locked v1 decisions, scope, milestone) and `Q1`–`Q6` (toolchains, sync, latency, sandbox, ML stack, legal). Read the Foundation doc before changing scope or architecture.

## Layout

| Path | What |
|---|---|
| `crates/bc-proto` | `bc.toml` spec (`spec.rs`), device-to-device CBOR frames (`wire.rs`), cloud JSON bodies (`cloud.rs`). No networking. |
| `crates/bc-net` | iroh endpoint setup, device key files, `send_msg`/`recv_msg` over QUIC streams |
| `crates/bc-secrets` | Device key and session token storage: OS keychain (macOS Keychain, Windows Credential Manager) when a program calls `bc_net::use_keychain()` at startup, migrating old files in; owner-only files otherwise, which is what libraries and tests use |
| `crates/bc-auth` | HTTP client for `bc-cloud`, session file storage |
| `apps/bc-agent` | Mac agent: `runner.rs` (workspace, process, output), `server.rs` (per-connection protocol), `paired.rs` (allowlist), `cloud_sync.rs` (heartbeat + paired list), `service.rs` (the running agent, shared by `bc-agent serve` and the menu-bar app), `events.rs` (job and sync events for a UI) |
| `apps/bc-cli` | `bc` binary: `pack.rs` (tar of project), `run.rs` (client side of a job), `detect.rs` (project type and suggested commands), `flows.rs` (steps shared with the desktop app: resolve Mac, load spec, split command, wait for pairing), `main.rs` (commands) |
| `apps/bc-desktop` | Tauri 2 client app ("Borrowed Compute"): `src/main.rs` commands, `ui/` plain HTML/CSS/JS. Shares the CLI's home dir, so it is the same device as `bc`. |
| `apps/bc-desktop-agent` | Tauri 2 app on the Mac ("Borrowed Compute Agent"): a normal Dock app; with Run in background on, quitting only hides it (`src/quit_guard.rs` adds `applicationShouldTerminate:`, which Tauri lacks) and a menu-bar icon reopens it. Runs `bc_agent::service::Agent`, hosts `bc-cloud` on SQLite when the Mac's cloud address is the default, shares it with `tailscale serve` (`src/tailscale.rs`), pairing approval, recent jobs |
| `design/` | Icon sources (SVG). Regenerate with `npx @tauri-apps/cli@2.12.4 icon design/icon-client.svg -o apps/bc-desktop/icons` |
| `apps/bc-cloud` | axum + sqlx `Any`: GitHub device-flow sign-in, devices, pairings, presence. Runs on SQLite (inside the Mac app) or Postgres. Migrations in `migrations/sqlite` and `migrations/postgres` |

## Commands

```bash
cargo test --workspace
cargo clippy --workspace --all-targets
```

Cloud API tests always run on a fresh SQLite file; with `BC_TEST_DATABASE_URL` set they also run on Postgres (each test creates its own database):

```bash
BC_TEST_DATABASE_URL=postgres://bc@127.0.0.1:54329/postgres cargo test --workspace
```

Throwaway local Postgres (Homebrew): `initdb -D <dir> -U bc --auth=trust`, then start with `LC_ALL=en_US.UTF-8 pg_ctl -D <dir> -o "-p 54329 -k /tmp/bcpg -h 127.0.0.1" start`. Postgres refuses to start without `LC_ALL`, and Unix socket paths over 103 bytes fail, hence `-k /tmp/bcpg`.

Desktop apps: `cargo run -p bc-desktop` or `cargo run -p bc-desktop-agent` for development (the UI is static files in `ui/`, no bundler). Bundles: `npx @tauri-apps/cli@2.12.4 build` inside the app's folder (`--debug --bundles app` for a quick macOS `.app`). The UI is embedded at compile time, so rebuild after editing `ui/`. For a `.dmg`, run with `CI=true` so the bundler skips its Finder AppleScript styling, which times out without automation permission. Release builds rely on `[profile.release.build-override] strip = false` in `Cargo.toml`; without it, macOS 27 fails every release build with "mis-aligned LINKEDIT string pool".

Run the cloud on its own: `BC_DATABASE_URL=postgres://… or sqlite://<file>?mode=rwc BC_GITHUB_CLIENT_ID=... cargo run -p bc-cloud` (listens on `127.0.0.1:8080`). Normally the Mac app hosts it. A device's cloud address is `BC_CLOUD_URL`, else the `cloud-url` file in its home folder (set by the apps or `bc cloud <address>`), else `http://127.0.0.1:8080`. `BC_HOME` relocates keys, sessions, the cloud address and the Mac app's `cloud/cloud.db`.

## Releases and updates

Every merge to `main` runs `.github/workflows/release.yml`. The Mac app (agent) and the Windows app (client) are **versioned separately**: `.github/release-plan.sh` compares the push with each app's last tag in this repo (`agent-v0.2.3`, `client-v0.2.7`; both start at 0.2.0) and releases only the apps whose files changed (shared crates and toolchain files release both). Each built app is signed and published to the public repo `Harshil089/Borrowed-Compute-releases` as its own release (`agent-v…` with the dmg, `client-v…` with the installer), then this repo is tagged. Every release carries both `agent.json` and `client.json` (the other app's copied from the previous latest release), because each app's Check for updates reads its manifest from the releases repo's latest release. Run the workflow by hand with `apps: both|agent|client` to force a release. Each app installs only packages signed by the key whose public half is `plugins.updater.pubkey` in its `tauri.conf.json`. The private key is the `TAURI_SIGNING_PRIVATE_KEY` secret (local copy `~/.tauri/borrowed-compute-updater.key`); publishing uses the `RELEASES_TOKEN` secret. Losing the key means installed apps can never update again; reinstall by hand with a new key. Local builds stay `0.1.0`, so any release counts as newer.

## Rules

- **Rust 1.91 is pinned** in `rust-toolchain.toml` because iroh 1.x requires it. Do not lower it.
- **Insecure relay TLS is test-only.** `NetConfig::insecure_relay_tls` exists only with `bc-net`'s `test-utils` feature. Never enable that feature in a non-dev dependency.
- **Jobs are `argv` arrays, never shell strings.** Keep it that way in the spec, wire format and runner.
- **Unknown `bc.toml` keys are errors** (`deny_unknown_fields`). Add new keys explicitly and validate them in `JobSpec::validate`.
- **Bump `PROTOCOL_VERSION` and `JOB_ALPN`** together on any incompatible change to `wire.rs`.
- **Never store provider tokens** (GitHub or Google). The cloud uses one once to read the user and their verified emails, then drops it. Session tokens are stored only as SHA-256.
- **Accounts are reached through identities** (`identities` table: provider + provider user id). A new identity joins the account of an existing identity with the same **verified** email, else creates an account (`account_for` in `auth.rs`). Never link on an unverified address. Google sign-in is offered only when a client id and secret are configured (`BC_GOOGLE_CLIENT_ID`/`BC_GOOGLE_CLIENT_SECRET`, or the Mac app's built-in id plus the `GOOGLE_CLIENT_SECRET` CI secret).
- **SQLite migrations run with foreign keys off** (`migrate_sqlite` in `bc-cloud`), then `pragma foreign_key_check`, because table rebuilds would otherwise cascade-delete sessions and devices. sqlx always wraps a SQLite migration in a transaction, so pragmas inside a migration file do nothing.
- **Device registration requires a signature** over `register_message` (bound to the session hash). Do not add a registration path without it.
- **Every cloud query is scoped by `account_id`.** v1 pairs only devices on the same account (Foundation decision; Q6).
- **Pairing codes are a secret between the requesting device and the person at the Mac.** Only `create` returns the code (to the requester); every list returns `code: None`, so the Mac owner must type it. Codes are 8 characters from an unambiguous alphabet, compared case- and dash-insensitively; three wrong codes delete the request (`pairings.attempts`).
- **The agent checks the paired list on every connection.** It must never run a job from an unpaired device.
- **Packing fixes Windows files** (`apps/bc-cli/src/pack.rs`): executable bit from file system, git index (`100755`) or shebang for untracked files; CRLF to LF only in scripts. Do not normalize other text files.
- **Tests that need `sh` or the Mac agent are `cfg(unix)`**, so `cargo test --workspace` still runs on Windows CI.
- **Logs.** The Mac app writes `<home>/logs/agent.log` (its own stderr, unless run from a terminal) and one file per job in `<home>/logs/jobs/` (command, result, usage; newest 500 kept). The recent-jobs list is reloaded at startup from `<home>/agent/job-history.jsonl`. It never writes job output on the Mac. The client app writes each run's output and result to `<client home>/logs/job-<time>.log`, keeping the newest 200.
- **Job usage comes from `wait4`** in `runner.rs`: CPU time and peak memory of the job's process plus every process it waited for. `AgentMsg::Exited.usage` is `#[serde(default)]`, so old and new clients and agents still talk without a protocol bump.
- **Jobs start from empty dependency caches** (`isolated_env` in `runner.rs`): `CARGO_HOME`, npm/pnpm/yarn/bun, pip/uv/poetry, Go, Gradle, Maven, NuGet, `XDG_CACHE_HOME` and `TMPDIR` all point into `<workspace>/cache`, set after the job's own env so a job cannot redirect them. Every build re-downloads its dependencies; that is the owner's zero-data choice, not a bug to optimize away. Installed toolchains stay where they are (`RUSTUP_HOME` is set explicitly so rustup still finds them). The uploaded project lives in `<workspace>/project`.
- **Jobs get the user's login-shell PATH** (`job_path` in `runner.rs`), asked once with `$SHELL -l -i`, because an app started from Finder only has `/usr/bin:/bin:/usr/sbin:/sbin` and would not find cargo, npm (nvm), Homebrew tools and the like.
- **Jobs run in a Seatbelt sandbox on macOS** (`apps/bc-agent/src/sandbox.rs`, through `/usr/bin/sandbox-exec`): read and write only the job's workspace; read system folders, Homebrew, Xcode and the user's toolchains (`.rustup`, nvm, pyenv…, plus PATH folders inside home); never the rest of home (`~/.cargo` beyond `bin` holds credentials). Outbound network allowed; inbound only on localhost and on Unix sockets inside the workspace. When a tool breaks inside the sandbox, run `cargo run -p bc-agent --example sandbox_check` and widen the profile narrowly; `BC_NO_SANDBOX=1` on the agent turns it off for debugging.
- **Workspaces are deleted after every job**, and the job's process group is killed on exit, timeout and cancel (`GroupGuard` in `runner.rs`). Keep both guarantees.
- **Desktop apps call the libraries, never the CLI binaries.** New client or agent behavior goes in `bc_cli::flows` or `bc_agent::service` first, then both the CLI and the app use it.
- **Never use `confirm()`, `alert()` or `prompt()` in the desktop UIs.** The web view has no native handler for them on macOS (the Mac app's `confirm()` returned "no" without showing anything). Use `ask()` in each `ui/app.js`.
- **Keep `ui/base.css` and `ui/fonts/` identical** in both desktop apps. Text colors must pass WCAG AA (4.5:1) in light and dark mode.
- **Appearance is the app's choice, not only the system's.** `ui/theme.js` (identical in both apps, loaded in `<head>` before CSS so there is no flash) sets `data-theme` to `light` (default), `dark` or `system`; base.css keys dark tokens on `[data-theme="dark"]` and on `[data-theme="system"]` inside `prefers-color-scheme: dark`. Dark is black, not navy.
- **The visual system is in `DESIGN.md`** (product context in `PRODUCT.md`): a jet-age ticket wallet. Fonts are self-hosted in `ui/fonts` because the CSP allows no external styles. Carbon purple is only for measured figures and codes to match; at most one filled red button per coupon; no label lines above headings.
- **The agent window re-renders only what changed** (`setHtml` in its `app.js`), so the 5-second poll never resets focus or replays animations. Keep new lists on that path.
- Cloud SQL uses runtime `sqlx::query`/`query_as`, not compile-time macros, so builds need no database.
- **Cloud SQL must run on SQLite and Postgres.** No database-side time (`now()`, intervals): pass Unix seconds from `bc_cloud::now()`. Number `$N` parameters in order of first use. Keep both migration folders describing the same schema. Tests run every API test on SQLite.
- **The apps read the session from disk on every call** (`AppState::cloud()`), so the CLI and the app stay one device. A 401 means the session belongs to another cloud: sign out locally and show sign-in.
- Write code comments and docs in plain prose: why, not what.

## Known gaps (Phase one)

- `bc` cannot cross-compile for Windows from macOS (`ring` needs Windows SDK headers). `.github/workflows/windows.yml` builds `bc.exe` and runs the Windows-capable tests on `windows-latest`. A real job from a Windows PC to a Mac is still untested.
- Jobs run inside a Seatbelt sandbox but still as the agent's user; the separate `_bcjob` user needs a root helper and Developer ID signing (Phase two, Q4).
- The whole project is uploaded per run (up to 2 GiB), by design: the owner decided the Mac stores zero project data, so there is no Mac-side cache or chunked sync (Foundation doc, 2026-10-25). Compressed uploads (zstd) are the planned speed-up.
- **Route memory** (`apps/bc-cli/src/route.rs`): when a direct path to a Mac fails, the client records it in `route-memory.json` and sends that Mac's runs through the relay first for an hour, or until the Mac's IP addresses change, or until a direct run succeeds.
- Discovery and relays use iroh's public relays (dev only). Our own `iroh-relay` deployment is not written.
- The Mac app's built-in cloud is reachable from other devices only through Tailscale. Tested on one Mac (share, HTTPS address, sign-in and a job through it); not yet from a second device.
- The desktop apps are verified on macOS only. CI builds the Windows installers, but nobody has run them on a Windows PC yet.
- The Mac agent app is ad-hoc signed (`bundle.macOS.signingIdentity: "-"`), not Developer ID signed or notarized, so a downloaded copy needs System Settings › Privacy & Security › Open Anyway once. Without the bundle signature macOS calls the download "damaged". It does not start at login yet. Updates installed by the app are not quarantined, so they open directly.
