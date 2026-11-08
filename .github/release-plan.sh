#!/bin/sh
# Decides which apps to release on this push and their next versions, as KEY=VALUE lines
# for $GITHUB_OUTPUT. The Mac build (agent) and the Windows build (client) of the one app
# are versioned separately: each has its own tags (agent-v0.2.3, client-v0.2.7) and is
# released when something it is built from changed since its last tag.
#
# usage: release-plan.sh [auto|both|agent|client]
set -eu
force=${1:-auto}

# versionsort.suffix sorts 0.3.29-1 before 0.3.29, as the updater (semver) does.
last_tag() {
  git -c versionsort.suffix=- tag --list "$1-v*" --sort=-v:refname | head -n 1
}

# A push straight to main (no "(#N)" from a merged pull request) is a hotfix: it gets a
# fourth number, 0.3.28.1, shown by the app and release titles. Semver has no fourth
# number, so it is written 0.3.29-1: above 0.3.28 and below the next release, 0.3.29.
hotfix=false
if ! git log -1 --format=%s | grep -Eq '\(#[0-9]+\)$'; then hotfix=true; fi

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
    0.2.*) echo "0.3.0"; return ;;
  esac
  base=${v%%-*}
  case "$v" in
    *-*) n=${v#*-}; next=$base ;;
    *) n=0; next="${base%.*}.$(( ${base##*.} + 1 ))" ;;
  esac
  if [ "$hotfix" = true ]; then echo "$next-$(( n + 1 ))"; else echo "$next"; fi
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
