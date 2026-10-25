# Borrowed Compute

Run code jobs (builds, tests, scripts, small ML training) on a paired Mac's Apple Silicon from a Windows PC. The PC sends **source plus a job spec**; the Mac runs its native arm64 toolchain and streams output back. No Windows binaries ever run on the Mac.

Product and design docs live outside this repo, in the parent folder `../`: `Borrowed Compute Foundation.md` (locked v1 decisions, scope, milestone) and `Q1`–`Q6` (toolchains, sync, latency, sandbox, ML stack, legal). Read the Foundation doc before changing scope or architecture.

## Layout

| Path | What |
|---|---|
| `crates/bc-proto` | `bc.toml` spec (`spec.rs`), device-to-device CBOR frames (`wire.rs`), cloud JSON bodies (`cloud.rs`). No networking. |
| `crates/bc-net` | iroh endpoint setup, device key files, `send_msg`/`recv_msg` over QUIC streams |
| `crates/bc-auth` | HTTP client for `bc-cloud`, session file storage |
| `apps/bc-agent` | Mac agent: `runner.rs` (workspace, process, output), `server.rs` (per-connection protocol), `paired.rs` (allowlist), `cloud_sync.rs` (heartbeat + paired list), `service.rs` (the running agent, shared by `bc-agent serve` and the menu-bar app), `events.rs` (job and sync events for a UI) |
| `apps/bc-cli` | `bc` binary: `pack.rs` (tar of project), `run.rs` (client side of a job), `flows.rs` (steps shared with the desktop app: resolve Mac, load spec, split command, wait for pairing), `main.rs` (commands) |
| `apps/bc-desktop` | Tauri 2 client app ("Borrowed Compute"): `src/main.rs` commands, `ui/` plain HTML/CSS/JS. Shares the CLI's home dir, so it is the same device as `bc`. |
| `apps/bc-desktop-agent` | Tauri 2 menu-bar app on the Mac ("Borrowed Compute Agent"): runs `bc_agent::service::Agent`, pairing approval, recent jobs |
| `design/` | Icon sources (SVG). Regenerate with `npx @tauri-apps/cli@2.12.4 icon design/icon-client.svg -o apps/bc-desktop/icons` |
| `apps/bc-cloud` | axum + Postgres: GitHub device-flow sign-in, devices, pairings, presence. Migrations in `migrations/` |

## Commands

```bash
cargo test --workspace
cargo clippy --workspace --all-targets
```

Cloud API tests need Postgres; without `BC_TEST_DATABASE_URL` they are skipped (and pass silently). Each test creates its own database:

```bash
BC_TEST_DATABASE_URL=postgres://bc@127.0.0.1:54329/postgres cargo test --workspace
```

Throwaway local Postgres (Homebrew): `initdb -D <dir> -U bc --auth=trust`, then start with `LC_ALL=en_US.UTF-8 pg_ctl -D <dir> -o "-p 54329 -k /tmp/bcpg -h 127.0.0.1" start`. Postgres refuses to start without `LC_ALL`, and Unix socket paths over 103 bytes fail, hence `-k /tmp/bcpg`.

Desktop apps: `cargo run -p bc-desktop` or `cargo run -p bc-desktop-agent` for development (the UI is static files in `ui/`, no bundler). Bundles: `npx @tauri-apps/cli@2.12.4 build` inside the app's folder (`--debug --bundles app` for a quick macOS `.app`). The UI is embedded at compile time, so rebuild after editing `ui/`.

Run the cloud: `BC_DATABASE_URL=... BC_GITHUB_CLIENT_ID=... cargo run -p bc-cloud` (listens on `127.0.0.1:8080`). Devices use `BC_CLOUD_URL`; `BC_HOME` relocates keys, sessions and the timing log.

## Rules

- **Rust 1.91 is pinned** in `rust-toolchain.toml` because iroh 1.x requires it. Do not lower it.
- **Insecure relay TLS is test-only.** `NetConfig::insecure_relay_tls` exists only with `bc-net`'s `test-utils` feature. Never enable that feature in a non-dev dependency.
- **Jobs are `argv` arrays, never shell strings.** Keep it that way in the spec, wire format and runner.
- **Unknown `bc.toml` keys are errors** (`deny_unknown_fields`). Add new keys explicitly and validate them in `JobSpec::validate`.
- **Bump `PROTOCOL_VERSION` and `JOB_ALPN`** together on any incompatible change to `wire.rs`.
- **Never store GitHub tokens.** The cloud uses one once to read the user, then drops it. Session tokens are stored only as SHA-256.
- **Device registration requires a signature** over `register_message` (bound to the session hash). Do not add a registration path without it.
- **Every cloud query is scoped by `account_id`.** v1 pairs only devices on the same account (Foundation decision; Q6).
- **The agent checks the paired list on every connection.** It must never run a job from an unpaired device.
- **Packing fixes Windows files** (`apps/bc-cli/src/pack.rs`): executable bit from file system, git index (`100755`) or shebang for untracked files; CRLF to LF only in scripts. Do not normalize other text files.
- **Tests that need `sh` or the Mac agent are `cfg(unix)`**, so `cargo test --workspace` still runs on Windows CI.
- **Workspaces are deleted after every job**, and the job's process group is killed on exit, timeout and cancel (`GroupGuard` in `runner.rs`). Keep both guarantees.
- **Desktop apps call the libraries, never the CLI binaries.** New client or agent behavior goes in `bc_cli::flows` or `bc_agent::service` first, then both the CLI and the app use it.
- **Keep `ui/base.css` identical** in both desktop apps. Text colors must pass WCAG AA (4.5:1) in light and dark mode.
- Cloud SQL uses runtime `sqlx::query`/`query_as`, not compile-time macros, so builds need no database.
- Write code comments and docs in plain prose: why, not what.

## Known gaps (Phase one)

- `bc` cannot cross-compile for Windows from macOS (`ring` needs Windows SDK headers). `.github/workflows/windows.yml` builds `bc.exe` and runs the Windows-capable tests on `windows-latest`. A real job from a Windows PC to a Mac is still untested.
- No sandbox yet: jobs run as the agent's user. The `_bcjob` user plus a Seatbelt profile is Phase two (Q4).
- The whole project is uploaded per run (up to 2 GiB). Content-addressed chunked sync is Phase two (Q2).
- Keys and session tokens are owner-only files, not yet in the OS keychain.
- Without sign-in, discovery uses iroh's public relays (dev only). Our own `iroh-relay` deployment is not written.
- The desktop apps are verified on macOS only. CI builds the Windows installers, but nobody has run them on a Windows PC yet.
- The Mac agent app is not signed or notarized, and does not start at login yet.
