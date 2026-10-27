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
| `apps/bc-cloud` | Cloud API: GitHub sign-in, device registry, pairing, presence (axum; SQLite inside the Mac app, or Postgres on a server) |

| `apps/bc-desktop` | Desktop app "Borrowed Compute" for the PC (Tauri 2) |
| `apps/bc-desktop-agent` | Menu-bar app "Borrowed Compute Agent" for the Mac (Tauri 2) |

Not built yet: `bc-sync` (chunked sync), the `_bcjob` user and Seatbelt sandbox, our own relay deployment.

## Build and test

Rust 1.91 is pinned in `rust-toolchain.toml` (iroh 1.x needs it).

```bash
cargo test --workspace
```

The cloud API tests always run against a fresh SQLite file. To run them against Postgres as well, point `BC_TEST_DATABASE_URL` at an admin connection; each test creates its own database:

```bash
BC_TEST_DATABASE_URL=postgres://bc@127.0.0.1:54329/postgres cargo test --workspace
```

They use a fake GitHub, so they need no network.

The Windows build of `bc` must run on Windows. Cross-compiling from macOS fails because `ring` needs the Windows SDK C headers. The `windows` GitHub Actions workflow builds `bc.exe` on `windows-latest` for every push to `main` and every pull request, runs the tests that work there, and uploads `bc.exe` as the `bc-windows-x86_64` artifact. The agent runner tests and the end-to-end test are Unix only (they use `sh` and Unix process groups).

## Where the cloud runs

The cloud needs no server and no VPN: **Borrowed Compute Agent hosts it on the Mac**, in a SQLite file under the app's data folder, listening on `127.0.0.1:8080`. The PC reaches it over iroh, the same encrypted device-to-device connection jobs use:

1. In Borrowed Compute Agent, under **Route**, copy this Mac's code.
2. In Borrowed Compute on the PC, paste it under **Your Mac's code**, click **Test**, then sign in.

The PC finds the Mac by its code through iroh's public discovery, connects directly when it can and through iroh's public relays when it cannot. Relays forward encrypted traffic they cannot read. Anyone with the code can reach the sign-in page, but a device can send jobs only after you approve its pairing on the Mac.

The `bc` command uses the same saved address (`bc cloud <code or address>` sets it). `BC_CLOUD_URL` overrides it for one run.

To run the cloud on a server instead (needs a GitHub OAuth app with **Device Flow enabled**; only the client id is used, no client secret):

```bash
BC_DATABASE_URL=postgres://user@host/db BC_GITHUB_CLIENT_ID=<client id> cargo run -p bc-cloud
```

`BC_DATABASE_URL` may also be `sqlite://<file>?mode=rwc`. It listens on `127.0.0.1:8080` (`BC_LISTEN` to change). When a Mac's saved cloud address points at such a server, Borrowed Compute Agent uses it and does not host its own.

## Desktop apps

- **Borrowed Compute** (`apps/bc-desktop`, for the PC): sign in with GitHub, see your Macs and whether they are online, pair with a code, choose a project folder, type a command and run it with live output, cancel and timings. If the folder has a `bc.toml`, its command is used unless you type another.
- **Borrowed Compute Agent** (`apps/bc-desktop-agent`, for the Mac): lives in the menu bar, runs the agent, signs in, shows pairing requests with the code to compare and an Approve button, paired devices with Remove, and recent jobs. Closing the window keeps it running; Quit in the menu-bar menu stops it.

The apps share keys and sessions with `bc` and `bc-agent`, so signing in once covers both.

Build installers with the Tauri CLI (needs Node):

```bash
cd apps/bc-desktop && npx @tauri-apps/cli@2.12.4 build
```

```bash
cd apps/bc-desktop-agent && npx @tauri-apps/cli@2.12.4 build
```

The Windows installers (`.exe` and `.msi`) come from the `windows` workflow as the `borrowed-compute-windows-installers` artifact.

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

## Files from Windows

Windows does not record Unix permissions and often checks files out with CRLF line endings. Both break scripts on the Mac, so `bc` fixes them while packing:

- **Executable bit:** a file is executable if the file system says so, or git records it as executable (`100755`, as with `git update-index --chmod=+x`). A file git does not track is executable if it starts with `#!`.
- **Line endings:** CRLF becomes LF in scripts only: executable files, files starting with `#!`, and `.sh`, `.bash` and `.zsh` files. Binary files and other text files (`.bat`, data, fixtures) are sent unchanged.

## How sign-in and pairing work

- The cloud runs the GitHub device flow. It uses the GitHub token once to read the user's id and login, then drops it.
- The cloud issues its own session token. It stores only the token's SHA-256.
- A device registers its iroh public key by signing a message bound to the current session, so nobody can register a key they do not hold, and a signature cannot be reused by another account.
- Pairing is limited to devices on the same account (Foundation decision). The PC asks, the Mac owner approves with the code shown on both screens. Codes expire after 10 minutes.
- The agent sends a heartbeat every 15 seconds with its current address, and copies the approved list into its local paired file. A revoked pairing stops new jobs within 15 seconds.
- The cloud never sees job contents. Jobs go device to device, end-to-end encrypted.
- Changing a device's cloud address signs it out, because each cloud has its own accounts and devices.

## Phase one limits

- The whole project is uploaded on every run (up to 2 GiB). Chunked sync comes in Phase two.
- Jobs run as the agent's own user, with no sandbox. Use only with your own devices.
- Session tokens and device keys are files readable only by their owner, not yet in the OS keychain.
- The desktop apps are tested on macOS only. The Windows installers are built in CI but not yet run on a Windows PC. The Mac app is not signed or notarized.
- Without signing in, devices are found through iroh's public relays and address lookup, which is for development only.
