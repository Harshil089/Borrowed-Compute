# Borrowed Compute

Run code jobs on a paired Mac's Apple Silicon from a Windows PC.

See `../Borrowed Compute Foundation.md` for decisions, scope and the first milestone.

## Layout

| Path | Purpose |
|---|---|
| `crates/bc-proto` | Job spec (`bc.toml`), device-to-device wire messages, cloud API bodies |
| `crates/bc-net` | iroh endpoint setup, device keys, framed messages over QUIC |
| `crates/bc-auth` | Client for the cloud API, used by `bc` and `bc-agent` |
| `apps/bc-agent` | Mac agent: job runner, pairing checks, cloud presence sync |
| `apps/bc-cli` | `bc` command: sign-in, pairing, packing the project, running jobs |
| `apps/bc-cloud` | Cloud API: GitHub sign-in, device registry, pairing, presence (axum + Postgres) |

Not built yet: `bc-sync` (chunked sync), the `_bcjob` user and Seatbelt sandbox, our own relay deployment.

## Build and test

Rust 1.91 is pinned in `rust-toolchain.toml` (iroh 1.x needs it).

```bash
cargo test --workspace
```

The cloud API tests need Postgres. Point `BC_TEST_DATABASE_URL` at an admin connection; each test creates its own database:

```bash
BC_TEST_DATABASE_URL=postgres://bc@127.0.0.1:54329/postgres cargo test --workspace
```

Without it, those tests are skipped. They use a fake GitHub, so they need no network.

The Windows build of `bc` must run on Windows (or a Windows CI runner). Cross-compiling from macOS fails because `ring` needs the Windows SDK C headers.

## Run the cloud locally

Needs a GitHub OAuth app with **Device Flow enabled**. Only the client id is used; no client secret.

```bash
BC_DATABASE_URL=postgres://bc@127.0.0.1:54329/bc_cloud BC_GITHUB_CLIENT_ID=<client id> cargo run -p bc-cloud
```

It listens on `127.0.0.1:8080` (`BC_LISTEN` to change). Devices find it through `BC_CLOUD_URL`, which defaults to `http://127.0.0.1:8080`.

## Use it

On the Mac:

```bash
bc-agent login --name studio
bc-agent serve
```

On the Windows PC:

```bash
bc login --name laptop
bc pair studio            # shows a 6-digit code and waits
```

Back on the Mac, approve with the same code (or list requests with `bc-agent pairings`):

```bash
bc-agent approve 123456
```

Then, in the project folder on Windows:

```bash
bc use studio
bc run --timings -- cargo test
```

Other commands: `bc devices`, `bc logout`, `bc-agent unpair laptop`. `bc run --relay-only` forces the relay path. Every run appends a line to `timings.jsonl` in the `cli` folder under the config directory (`BC_HOME` overrides it).

Without signing in, the agent's local list still works for development: `bc-agent allow <device id>` on the Mac, and `bc run --to <mac device id>` on the PC.

## How sign-in and pairing work

- The cloud runs the GitHub device flow. It uses the GitHub token once to read the user's id and login, then drops it.
- The cloud issues its own session token. It stores only the token's SHA-256.
- A device registers its iroh public key by signing a message bound to the current session, so nobody can register a key they do not hold, and a signature cannot be reused by another account.
- Pairing is limited to devices on the same account (Foundation decision). The PC asks, the Mac owner approves with the code shown on both screens. Codes expire after 10 minutes.
- The agent sends a heartbeat every 15 seconds with its current address, and copies the approved list into its local paired file. A revoked pairing stops new jobs within 15 seconds.
- The cloud never sees job contents. Jobs go device to device, end-to-end encrypted.

## Phase one limits

- The whole project is uploaded on every run (up to 2 GiB). Chunked sync comes in Phase two.
- Jobs run as the agent's own user, with no sandbox. Use only with your own devices.
- Session tokens and device keys are files readable only by their owner, not yet in the OS keychain.
- If a job is cancelled, its own process is killed, but background processes it started may survive. On normal exit and on timeout the whole process group is killed.
- Without signing in, devices are found through iroh's public relays and address lookup, which is for development only.
