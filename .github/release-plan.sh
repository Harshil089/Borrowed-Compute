#!/bin/sh
# Decides which apps to release on this push and their next versions, as KEY=VALUE lines
# for $GITHUB_OUTPUT. The Mac build (agent) and the Windows build (client) of the one app
# are versioned separately: each has its own tags (agent-v0.2.3, client-v0.2.7) and is
# released when something it is built from changed since its last tag.
#
# usage: release-plan.sh [auto|both|agent|client]
set -eu
force=${1:-auto}

last_tag() {
  git tag --list "$1-v*" --sort=-v:refname | head -n 1
}

# The next patch version after the app's last tag. Apps start at 0.2.0, above the
# 0.1.x versions they shared before, so installed apps see it as newer.
next_version() {
  last=$(last_tag "$1")
  if [ -z "$last" ]; then
    echo "0.2.0"
    return
  fi
  v=${last#"$1-v"}
  # The merged app starts at 0.3.0 on both.
  case "$v" in
    0.2.*) echo "0.3.0" ;;
    *) echo "${v%.*}.$(( ${v##*.} + 1 ))" ;;
  esac
}

# True if any of the paths changed since the app's last tag (or it was never released).
changed() {
  app=$1
  shift
  last=$(last_tag "$app")
  [ -z "$last" ] && return 0
  [ -n "$(git diff --name-only "$last" HEAD -- "$@")" ]
}

# Shared by both apps: a change here releases both.
SHARED="crates Cargo.toml Cargo.lock rust-toolchain.toml .github/workflows/release.yml .github/set-version.mjs .github/update-manifest.mjs .github/release-plan.sh"

agent=false
client=false
case "$force" in
  both) agent=true; client=true ;;
  agent) agent=true ;;
  client) client=true ;;
  *)
    # shellcheck disable=SC2086
    if changed agent apps $SHARED; then agent=true; fi
    # shellcheck disable=SC2086
    if changed client apps $SHARED; then client=true; fi
    ;;
esac

echo "agent=$agent"
echo "agent_version=$(next_version agent)"
echo "client=$client"
echo "client_version=$(next_version client)"
