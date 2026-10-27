# Borrowed Compute

**Run your code on your Mac's Apple Silicon from your Windows PC.**

Pick a project folder on the PC, type a command, and it runs on your Mac: builds, tests,
scripts, dev servers. Output streams back live, and dev servers open in the PC's browser at
`localhost`. Nothing to install on the PC beyond the app, no VPN, no server to rent, and
nothing from your project stays on the Mac once the job ends.

```
  Windows PC                                       Mac
  ┌────────────────────┐   encrypted, direct    ┌─────────────────────────┐
  │ Borrowed Compute   │ ─────────────────────▶ │ Borrowed Compute Agent  │
  │  project folder    │   (relay as fallback)  │  sandboxed job          │
  │  live output       │ ◀───────────────────── │  your toolchains        │
  │  localhost:5173    │                        │  built-in cloud         │
  └────────────────────┘                        └─────────────────────────┘
```

## What it does

- **Runs any command on the Mac.** `cargo test`, `npm run build`, `python train.py`, a
  shell script. The project is packed, sent, run in a fresh workspace, and deleted after.
- **Streams output live**, with timings (connect, upload, first output, total, direct or
  relay) and what the Mac used (CPU time, cores, peak memory).
- **Knows your project.** It recognizes Node, Python, Rust, Go, plain HTML and more, and
  suggests the right commands. A `bc.toml` in the folder always wins.
- **Fixes the usual first-run failures.** `node_modules`, `.venv` and build output are not
  uploaded, so it installs dependencies and builds first when a command needs them, and
  after a failure (a missing module, command or build file) it fixes the cause and runs
  once more.
- **Keeps servers running.** Tick *Keep running* (it ticks itself for `dev`, `start` and
  `serve` scripts) and the server stays up on the Mac, which stays awake, even after the PC
  disconnects, until you press Stop.
- **Forwards their ports.** A server on the Mac's port 5173 opens at
  `http://localhost:5173` on the PC. Ports reconnect quietly over flaky networks.
- **Fixes files from Windows.** Scripts get their executable bit and LF line endings back
  while packing; binaries and other text files are sent unchanged.
- **Keeps a log on both sides.** Each run's output is saved on the PC, and the Mac keeps
  the command, result, usage and the last 1 MB of output per job.
- **Updates itself.** Check for updates downloads, verifies and installs signed releases,
  retrying dropped downloads.

## Get started

1. **On the Mac**, download `BorrowedComputeAgent.dmg` from the
   [latest release](https://github.com/Harshil089/Borrowed-Compute-releases/releases),
   drag the app to Applications and open it. It is not notarized yet, so the first time
   macOS blocks it: open System Settings › Privacy & Security and click **Open Anyway**.
2. **Sign in** on the Mac with GitHub (or Google). The Mac now hosts your account.
3. Under **Route**, copy this Mac's **code**.
4. **On the PC**, install `BorrowedCompute-setup.exe` from the same release page. Windows
   SmartScreen may warn; choose More info › Run anyway.
5. Paste the code under **Your Mac's code**, click **Test**, then sign in with the **same
   account** as on the Mac.
6. Click **Pair** next to your Mac. The PC shows a code; approve the request on the Mac
   when the codes match.
7. Choose a project folder, pick or type a command, and press **Run**.

The PC finds the Mac by its code alone. It connects directly when the network allows and
through a relay when it does not (a VPN, a strict Wi-Fi). Either way the connection is
end-to-end encrypted.

## Security

Jobs are code from your PC plus every package they install, so the Mac treats them as
untrusted:

- **Sandboxed jobs.** Each job runs in a macOS Seatbelt sandbox. It can read and write its
  own workspace and read system folders and your toolchains (Homebrew, rustup, nvm, pyenv,
  Xcode). It cannot read the rest of your home folder (`~/Documents`, `~/.ssh`, `~/.aws`),
  the clipboard, the keychain or other apps' settings, cannot open apps, and cannot reach
  your ssh agent or well-known local services such as PostgreSQL, Redis, Docker or Ollama.
  It can download dependencies and run its own servers on localhost.
- **Nothing stays.** The workspace, including every downloaded dependency, is deleted when
  the job ends. Each run starts from empty caches.
- **Pairing.** Only devices signed in to the Mac's own account can ask to pair, and only
  after you approve a code shown on both screens. Codes expire after 10 minutes and three
  wrong tries delete the request. Removing a device stops its new jobs within 15 seconds.
- **One owner.** The Mac's built-in cloud accepts only its owner's account. Anyone else who
  learns the Mac's code is refused at sign-in. Sign-ins in progress and cloud connections
  are capped, refused devices are logged at most every 10 minutes, and one paired device
  can run at most 4 jobs and 8 servers at once.
- **Encrypted transport.** Devices talk over iroh QUIC with their own keys. Relays forward
  traffic they cannot read. The cloud never sees job contents.
- **Secrets.** Sign-in sessions and device keys live in the system keychain. GitHub and
  Google tokens are used once at sign-in and dropped.
- **Signed updates.** Each app installs only packages signed with the project's update key.

Known limits: jobs still run as your own macOS user (a separate job user is planned), the
Mac app is ad-hoc signed and not notarized, and the Windows installer is not code-signed.

## Command line

The `bc` command does what the PC app does, and shares its device identity:

```bash
bc cloud <mac code>        # where your Mac is
bc login                   # --google for Google
bc pair studio             # shows a code; approve it on the Mac
bc use studio              # default Mac for bc run

bc run -- cargo test       # run a command; --timings, --relay-only
bc run --service -- npm run dev   # keep a server running and forward its ports
bc services                # servers this device started
bc attach <id>             # watch one; Ctrl+C stops watching only
bc forward                 # forward their ports to localhost
bc stop <id>
```

On the Mac, `bc-agent` runs the agent without the app: `bc-agent login`, `bc-agent serve`,
`bc-agent pairings`, `bc-agent approve <code>`, `bc-agent unpair <device>`.

## Develop

| Path | What it is |
|---|---|
| `crates/bc-proto` | Job spec (`bc.toml`), device-to-device wire messages, cloud API bodies |
| `crates/bc-net` | iroh endpoints, device keys, framed messages, the cloud over iroh |
| `crates/bc-auth` | Cloud API client and session storage |
| `crates/bc-secrets` | Keychain storage for keys and sessions |
| `apps/bc-agent` | Mac agent: job runner, sandbox, services, port forwarding, cloud sync |
| `apps/bc-cli` | `bc`: packing, running, project detection, auto-fix, forwarding |
| `apps/bc-cloud` | Cloud API: sign-in, devices, pairing, presence (axum; SQLite or Postgres) |
| `apps/bc-desktop` | PC app "Borrowed Compute" (Tauri 2, plain HTML/CSS/JS UI) |
| `apps/bc-desktop-agent` | Mac app "Borrowed Compute Agent" (Tauri 2), hosts the cloud |

Rust 1.91 is pinned in `rust-toolchain.toml`.

```bash
cargo test --workspace
```

- The cloud tests use a fake GitHub and Google and run against SQLite. Set
  `BC_TEST_DATABASE_URL=postgres://…/postgres` to run them against Postgres too.
- `cargo run -p bc-agent --example sandbox_check` runs real Rust, Python, Node, Go and
  Homebrew projects inside the job sandbox (needs network).
- `cargo test -p bc-net --test cloud -- --ignored` finds a Mac by its code through public
  discovery (needs network).
- The Windows build must run on Windows (`ring` needs the Windows SDK). The `windows`
  workflow builds and tests it on every pull request.

Build the apps with the Tauri CLI:

```bash
cd apps/bc-desktop && npx @tauri-apps/cli@2.12.4 build
```

```bash
cd apps/bc-desktop-agent && npx @tauri-apps/cli@2.12.4 build
```

**Releases.** Every merge to `main` runs the `release` workflow, which releases only the
apps whose files changed. The Mac app (`agent-v…`) and the PC app (`client-v…`) have their
own versions and are published, signed, to
[Borrowed-Compute-releases](https://github.com/Harshil089/Borrowed-Compute-releases).

**A cloud on a server** instead of on the Mac (needs a GitHub OAuth app with Device Flow
enabled; only the client id is used):

```bash
BC_DATABASE_URL=postgres://user@host/db BC_GITHUB_CLIENT_ID=<client id> cargo run -p bc-cloud
```

Devices then use its web address instead of a Mac code. `BC_SINGLE_OWNER=1` locks it to
one account, and `BC_RELAY_URL` advertises a relay of your own.

`CLAUDE.md` holds the design rules and decisions behind all of this.
