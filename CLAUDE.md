# Borrowed Compute

Run code jobs (builds, tests, scripts, small ML training) on a paired Mac's Apple Silicon from a Windows PC. The PC sends **source plus a job spec**; the Mac runs its native arm64 toolchain and streams output back. No Windows binaries ever run on the Mac.

Product and design docs live outside this repo, in the parent folder `../`: `Borrowed Compute Foundation.md` (locked v1 decisions, scope, milestone) and `Q1`–`Q6` (toolchains, sync, latency, sandbox, ML stack, legal). Read the Foundation doc before changing scope or architecture.

## Layout

| Path | What |
|---|---|
| `crates/bc-proto` | `bc.toml` spec (`spec.rs`), device-to-device CBOR frames (`wire.rs`), cloud JSON bodies (`cloud.rs`). No networking. |
| `crates/bc-net` | iroh endpoint setup, device key files, `send_msg`/`recv_msg` over QUIC streams |
| `crates/bc-auth` | HTTP client for `bc-cloud`, session file storage |
| `apps/bc-agent` | Mac agent: `runner.rs` (workspace, process, output), `server.rs` (per-connection protocol), `paired.rs` (allowlist), `cloud_sync.rs` (heartbeat + paired list) |
| `apps/bc-cli` | `bc` binary: `pack.rs` (tar of project), `run.rs` (client side of a job), `main.rs` (commands) |
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
- **Workspaces are deleted after every job**, and the job's process group is killed on exit and timeout. Keep both guarantees.
- Cloud SQL uses runtime `sqlx::query`/`query_as`, not compile-time macros, so builds need no database.
- Write code comments and docs in plain prose: why, not what.

## Known gaps (Phase one)

- `bc` cannot cross-compile for Windows from macOS (`ring` needs Windows SDK headers). Build and test on Windows or a Windows CI runner. A real Windows run is still untested.
- No sandbox yet: jobs run as the agent's user. The `_bcjob` user plus a Seatbelt profile is Phase two (Q4).
- The whole project is uploaded per run (up to 2 GiB). Content-addressed chunked sync is Phase two (Q2).
- Keys and session tokens are owner-only files, not yet in the OS keychain.
- Without sign-in, discovery uses iroh's public relays (dev only). Our own `iroh-relay` deployment is not written.
- On cancel, background processes started by a job may survive.
