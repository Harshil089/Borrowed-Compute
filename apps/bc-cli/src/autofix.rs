//! Fixes the usual reasons a project does not run on a fresh Mac workspace: dependency
//! folders (`node_modules`, `.venv`) and build output (`dist/`) are never uploaded, so a
//! plain `node server.js` or `python app.py` finds nothing installed.
//!
//! [`preflight`] adds the missing install or build before a run. [`after_failure`] reads
//! a failed run's output, recognizes a missing module, command or build file, and returns
//! a corrected command to run once more. Both return a [`Fix`] that says what it does,
//! for the person watching. Everything installs inside the job's workspace, so nothing
//! stays on the device. Fixes are written for the device that runs the job ([`Os`]).

use std::path::Path;

use crate::detect::{node_package_manager, python_setup, Os};

#[derive(Debug, Clone, PartialEq)]
pub struct Fix {
    /// One sentence for the flight log, such as "Installing dependencies first".
    pub reason: String,
    pub argv: Vec<String>,
}

/// Commands that run Node code or Node tooling.
const NODE_TOOLS: &[&str] = &[
    "node", "npm", "npx", "pnpm", "yarn", "bun", "tsx", "ts-node", "tsc", "vite", "next", "nuxt", "astro",
    "webpack", "rollup", "esbuild", "jest", "vitest", "mocha", "eslint", "prettier", "nodemon",
];

/// Commands that run Python code or Python tooling.
const PYTHON_TOOLS: &[&str] = &[
    "python", "python3", "pytest", "flask", "uvicorn", "gunicorn", "django-admin", "streamlit", "jupyter",
];

/// Import names that differ from the package that provides them.
const PYTHON_PACKAGES: &[(&str, &str)] = &[
    ("cv2", "opencv-python"), ("PIL", "pillow"), ("sklearn", "scikit-learn"), ("yaml", "pyyaml"),
    ("bs4", "beautifulsoup4"), ("dotenv", "python-dotenv"), ("dateutil", "python-dateutil"),
    ("jwt", "pyjwt"), ("magic", "python-magic"), ("Crypto", "pycryptodome"), ("serial", "pyserial"),
];

/// Build output folders a start command may need.
const BUILD_DIRS: &[&str] = &["dist", "build", "out", ".next"];

/// Before a run: the install (and build) the command needs but the upload left out.
pub fn preflight(project: &Path, argv: &[String], os: Os) -> Option<Fix> {
    let program = program_name(argv)?;
    if already_installs(argv) {
        return None;
    }
    if NODE_TOOLS.contains(&program) {
        let (pm, install) = node_package_manager(project)?;
        if project.join("node_modules").is_dir() {
            return None;
        }
        let mut steps = vec![install.to_string()];
        let mut reason = "Installing dependencies first (node_modules is not uploaded)".to_string();
        if let Some(build) = missing_build(project, argv, pm) {
            steps.push(build);
            reason = "Installing dependencies and building first (node_modules and build output are not uploaded)".into();
        }
        return Some(chain(reason, &steps, argv, os));
    }
    if PYTHON_TOOLS.contains(&program) {
        let setup = python_setup(project, os)?;
        if project.join(".venv").is_dir() {
            return None;
        }
        return Some(Fix {
            reason: "Installing Python dependencies into the workspace first".into(),
            argv: os.shell(&format!("{setup} && {}", in_venv(argv, os))),
        });
    }
    None
}

/// After a failed run: a corrected command, when the output shows what was missing.
pub fn after_failure(project: &Path, argv: &[String], output: &str, os: Os) -> Option<Fix> {
    // A TypeScript file run with plain node.
    if output.contains("ERR_UNKNOWN_FILE_EXTENSION") || output.contains("Unknown file extension \".ts\"") {
        if let Some(script) = argv.iter().find(|a| a.ends_with(".ts") || a.ends_with(".mts")) {
            let mut fixed = vec!["npx".to_string(), "--yes".into(), "tsx".into(), script.clone()];
            fixed.extend(argv.iter().skip_while(|a| *a != script).skip(1).cloned());
            return Some(Fix { reason: "Running the TypeScript file with tsx".into(), argv: fixed });
        }
    }

    // Python: a package that would not build, most often for a Python newer than its wheels.
    if let Some(fix) = wheel_failure(argv, output, os) {
        return Some(fix);
    }

    // Python: a module that is not installed.
    if let Some(module) = between(output, "No module named '", "'") {
        let top = module.split('.').next().unwrap_or(module);
        let package = PYTHON_PACKAGES.iter().find(|(m, _)| *m == top).map(|(_, p)| *p).unwrap_or(top);
        let bin = os.venv_bin();
        // Subprojects' requirements (gateway/requirements.txt) often hold what is missing.
        let script = Os::shell_script(argv).unwrap_or_default();
        let subs: String = sub_requirements(project).iter().filter(|r| !script.contains(r.as_str())).map(|r| format!("-r {r} ")).collect();
        let install = format!("{bin}pip install -q {subs}{package}");
        let reason = format!("Installing the missing Python package {package} into the workspace");
        // A command that already sets up the virtualenv keeps its Python: install just before the run.
        if script.contains("-m venv .venv") {
            let mut steps: Vec<&str> = script.split(" && ").collect();
            steps.insert(steps.len() - 1, &install);
            return Some(Fix { reason, argv: os.shell(&steps.join(" && ")) });
        }
        let setup = match python_setup(project, os) {
            Some(setup) => format!("{setup} && {install}"),
            None => format!("{} -m venv .venv && {install}", os.python()),
        };
        return Some(Fix { reason, argv: os.shell(&format!("{setup} && {}", in_venv(argv, os))) });
    }

    let (pm, install) = node_package_manager(project)?;
    // Build output that is not there: build, then run again.
    let missing_build_file = BUILD_DIRS.iter().any(|dir| {
        output.lines().any(|l| {
            (l.contains("Cannot find module") || l.contains("ENOENT"))
                && (l.contains(&format!("/{dir}/")) || l.contains(&format!("\\{dir}\\")))
        })
    });
    if missing_build_file {
        if let Some(build) = build_script(project, pm) {
            let steps = if project.join("node_modules").is_dir() || already_installs(argv) {
                vec![build]
            } else {
                vec![install.to_string(), build]
            };
            return Some(chain("Building first; the build output was missing".into(), &steps, argv, os));
        }
    }
    // A package or tool that is not installed.
    let missing_package = output.contains("Cannot find module '")
        || output.contains("Cannot find package '")
        || output.contains("ERR_MODULE_NOT_FOUND")
        || output.contains("command not found")
        || output.contains("is not recognized as an internal or external command");
    if missing_package && !already_installs(argv) {
        return Some(chain("Installing dependencies; a package was missing".into(), &[install.to_string()], argv, os));
    }
    None
}

/// `requirements.txt` files one or two folders down (`gateway/`, `services/x/`), as
/// relative paths with `/`, which pip reads on every system. Paths that would need
/// quoting are left out.
fn sub_requirements(project: &Path) -> Vec<String> {
    let skip = |n: &str| n.starts_with('.') || ["venv", "node_modules", "__pycache__"].contains(&n);
    let dirs = |dir: &Path| -> Vec<(String, std::path::PathBuf)> {
        let mut found: Vec<_> = std::fs::read_dir(dir).into_iter().flatten().filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
            .filter(|(n, _)| !skip(n))
            .collect();
        found.sort();
        found
    };
    let mut out = Vec::new();
    for (name, path) in dirs(project) {
        let mut rels = vec![(name.clone(), path.clone())];
        rels.extend(dirs(&path).into_iter().map(|(n, p)| (format!("{name}/{n}"), p)));
        for (rel, p) in rels {
            if p.join("requirements.txt").is_file() && rel.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-/".contains(&b)) {
                out.push(format!("{rel}/requirements.txt"));
            }
        }
    }
    out
}

/// The oldest Python a retry falls back to.
const OLDEST_PYTHON: u32 = 9;
/// Printed by a retry that found no older Python, followed by the package that failed.
const NO_OLDER_PYTHON: &str = "bc: no older Python on this device for ";
/// The requirements a retry installs without the package that failed to build.
const RELAXED: &str = ".bc-requirements.txt";

/// A package failed to build its wheel. First retry the virtualenv on each older Python
/// the device has, newest first, below the one that failed; when none is left, install
/// that package's newest release instead of the pinned one, which may have a wheel.
// shortcut: Unix shells only, add the same steps with `py -3.N` once Windows needs them.
fn wheel_failure(argv: &[String], output: &str, os: Os) -> Option<Fix> {
    if os != Os::Unix {
        return None;
    }
    let script = Os::shell_script(argv)?;
    let mut steps: Vec<String> = script.split(" && ").map(str::to_string).collect();
    let venv = steps.iter().position(|s| s.ends_with("-m venv .venv"))?;
    // Package names go into the shell command, so only plain ones are used.
    let plain = |p: &&str| !p.is_empty() && p.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b));

    if let Some(package) = between(output, "Failed building wheel for ", "\n").map(str::trim).filter(plain) {
        let minor = python_minor(output).filter(|m| *m > OLDEST_PYTHON)?;
        let older: Vec<String> = (OLDEST_PYTHON..minor).rev().map(|v| v.to_string()).collect();
        steps[venv] = format!(
            // No " && " inside: the script is split on it.
            "PY=$(for v in {}; do if command -v python3.$v; then break; fi; done); [ -n \"$PY\" ] || {{ echo \"{NO_OLDER_PYTHON}{package}\" >&2; exit 1; }}; \"$PY\" -m venv .venv",
            older.join(" ")
        );
        return Some(Fix {
            reason: format!("{package} would not build for Python 3.{minor}; trying an older Python"),
            argv: os.shell(&steps.join(" && ")),
        });
    }

    let package = between(output, NO_OLDER_PYTHON, "\n").map(str::trim).filter(plain)?;
    if script.contains(RELAXED) {
        return None;
    }
    let pip = steps.iter().position(|s| s.contains("pip install") && s.contains(" -r "))?;
    let file = between(&steps[pip], " -r ", " ").or_else(|| steps[pip].split(" -r ").nth(1))?.to_string();
    let bin = os.venv_bin();
    steps[venv] = format!("{} -m venv .venv", os.python());
    steps[pip] = format!("grep -iv '^{package}' {file} > {RELAXED}; {}", steps[pip].replacen(&format!("-r {file}"), &format!("-r {RELAXED}"), 1));
    steps.insert(pip + 1, format!("{bin}pip install -q {package}"));
    Some(Fix {
        reason: format!("No older Python to try; installing the newest {package} instead of the pinned one"),
        argv: os.shell(&steps.join(" && ")),
    })
}

/// The Python minor version a failed build ran on: 15 for `cpython-315` or `python3.15`.
fn python_minor(output: &str) -> Option<u32> {
    let digits = |s: &str| s.bytes().take_while(u8::is_ascii_digit).map(char::from).collect::<String>().parse().ok();
    output.find("cpython-3").and_then(|i| digits(&output[i + 9..]))
        .or_else(|| output.find("python3.").and_then(|i| digits(&output[i + 8..])))
}

/// The build step a start command needs when its build output is missing: the command
/// names a build folder, or the package's start script does.
fn missing_build(project: &Path, argv: &[String], pm: &str) -> Option<String> {
    let start_script = scripts(project)
        .and_then(|s| s.get("start").and_then(|v| v.as_str()).map(str::to_string))
        .unwrap_or_default();
    let runs_start = argv.iter().any(|a| a == "start");
    let needs = BUILD_DIRS.iter().any(|dir| {
        let named = argv.iter().any(|a| a.starts_with(&format!("{dir}/")) || a.contains(&format!("/{dir}/")));
        let in_start = runs_start && start_script.contains(&format!("{dir}/"));
        (named || in_start) && !project.join(dir).exists()
    });
    needs.then(|| build_script(project, pm)).flatten()
}

fn build_script(project: &Path, pm: &str) -> Option<String> {
    scripts(project)?.contains_key("build").then(|| format!("{pm} run build"))
}

fn scripts(project: &Path) -> Option<serde_json::Map<String, serde_json::Value>> {
    let package: serde_json::Value = serde_json::from_slice(&std::fs::read(project.join("package.json")).ok()?).ok()?;
    package.get("scripts")?.as_object().cloned()
}

/// The program a command runs, looking through `sh -c "..."` or `cmd /c "..."`.
fn program_name(argv: &[String]) -> Option<&str> {
    if let Some(script) = Os::shell_script(argv) {
        return script.split_whitespace().next();
    }
    let first = argv.first()?;
    Some(first.rsplit(['/', '\\']).next().unwrap_or(first))
}

/// True when the command already installs dependencies itself.
fn already_installs(argv: &[String]) -> bool {
    let line = argv.join(" ");
    [" install", " ci", "pip install", " i "].iter().any(|w| line.contains(w)) || line.ends_with(" i")
}

/// Runs `steps` and then the original command, as one shell line.
fn chain(reason: String, steps: &[String], argv: &[String], os: Os) -> Fix {
    Fix { reason, argv: os.shell(&format!("{} && {}", steps.join(" && "), command_line(argv, os))) }
}

/// The original command as a shell line, unwrapping an existing shell command.
fn command_line(argv: &[String], os: Os) -> String {
    match Os::shell_script(argv) {
        Some(script) => script.to_string(),
        None => argv.iter().map(|a| quote(a, os)).collect::<Vec<_>>().join(" "),
    }
}

/// The command with Python and its tools taken from the workspace virtualenv.
fn in_venv(argv: &[String], os: Os) -> String {
    let line = command_line(argv, os);
    let mut words = line.splitn(2, ' ');
    let first = words.next().unwrap_or_default();
    let rest = words.next().map(|r| format!(" {r}")).unwrap_or_default();
    let program = first.rsplit(['/', '\\']).next().unwrap_or(first);
    let bin = os.venv_bin();
    match program {
        "python" | "python3" => format!("{bin}python{rest}"),
        p if PYTHON_TOOLS.contains(&p) => format!("{bin}{p}{rest}"),
        _ => line,
    }
}

fn quote(arg: &str, os: Os) -> String {
    if !arg.is_empty() && arg.chars().all(|c| c.is_ascii_alphanumeric() || "-_./=:@+,%".contains(c)) {
        arg.to_string()
    } else if os == Os::Windows {
        // cmd.exe has no single quotes; the program's own C runtime reads \" as a quote.
        format!("\"{}\"", arg.replace('"', "\\\""))
    } else {
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let from = text.find(start)? + start.len();
    let len = text[from..].find(end)?;
    Some(&text[from..from + len])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            let path = dir.path().join(name);
            if name.ends_with('/') {
                std::fs::create_dir_all(path).unwrap();
            } else {
                std::fs::write(path, body).unwrap();
            }
        }
        dir
    }

    fn argv(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    fn line(fix: &Fix) -> &str {
        &fix.argv[2]
    }

    #[test]
    fn fixes_fit_a_windows_pc() {
        let p = project(&[("package.json", "{}"), ("package-lock.json", "")]);
        let fix = preflight(p.path(), &argv(&["node", "-e", "console.log('hi')"]), Os::Windows).unwrap();
        assert_eq!(fix.argv, argv(&["cmd", "/d", "/c", "npm ci && node -e \"console.log('hi')\""]));
        let p = project(&[("requirements.txt", "flask"), ("app.py", "")]);
        let fix = preflight(p.path(), &argv(&["python", "app.py"]), Os::Windows).unwrap();
        assert_eq!(fix.argv[3], "python -m venv .venv && .venv\\Scripts\\pip install -q -r requirements.txt && .venv\\Scripts\\python app.py");
    }

    #[test]
    fn installs_node_modules_before_node_commands() {
        let p = project(&[("package.json", "{}"), ("package-lock.json", "")]);
        let fix = preflight(p.path(), &argv(&["node", "server.js"]), Os::Unix).unwrap();
        assert_eq!(line(&fix), "npm ci && node server.js");

        let uploaded = project(&[("package.json", "{}"), ("node_modules/", "")]);
        assert_eq!(preflight(uploaded.path(), &argv(&["node", "server.js"]), Os::Unix), None);
        assert_eq!(preflight(p.path(), &argv(&["npm", "install"]), Os::Unix), None, "already installs");
        assert_eq!(preflight(p.path(), &argv(&["cargo", "build"]), Os::Unix), None, "not a Node command");
    }

    #[test]
    fn builds_before_a_start_that_needs_build_output() {
        let pkg = r#"{"scripts":{"build":"tsc","start":"node dist/index.js"}}"#;
        let p = project(&[("package.json", pkg), ("pnpm-lock.yaml", "")]);
        let fix = preflight(p.path(), &argv(&["pnpm", "start"]), Os::Unix).unwrap();
        assert_eq!(line(&fix), "pnpm install --frozen-lockfile && pnpm run build && pnpm start");
        let fix = preflight(p.path(), &argv(&["node", "dist/index.js"]), Os::Unix).unwrap();
        assert!(line(&fix).contains("pnpm run build && node dist/index.js"), "{}", line(&fix));
    }

    #[test]
    fn python_runs_from_a_workspace_venv() {
        let p = project(&[("requirements.txt", "flask"), ("app.py", "")]);
        let fix = preflight(p.path(), &argv(&["python3", "app.py", "--port", "8000"]), Os::Unix).unwrap();
        assert_eq!(
            line(&fix),
            "python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt && .venv/bin/python app.py --port 8000"
        );
        let fix = preflight(p.path(), &argv(&["pytest", "-q"]), Os::Unix).unwrap();
        assert!(line(&fix).ends_with("&& .venv/bin/pytest -q"));
        assert_eq!(preflight(project(&[("app.py", "")]).path(), &argv(&["python3", "app.py"]), Os::Unix), None);
    }

    #[test]
    fn fixes_a_missing_python_module_by_package_name() {
        let p = project(&[("app.py", "")]);
        let out = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'cv2'\n";
        let fix = after_failure(p.path(), &argv(&["python3", "app.py"]), out, Os::Unix).unwrap();
        assert_eq!(line(&fix), "python3 -m venv .venv && .venv/bin/pip install -q opencv-python && .venv/bin/python app.py");
        assert!(fix.reason.contains("opencv-python"));
    }

    #[test]
    fn fixes_missing_node_packages_build_output_and_ts() {
        let pkg = r#"{"scripts":{"build":"vite build"}}"#;
        let p = project(&[("package.json", pkg)]);
        let fix = after_failure(p.path(), &argv(&["node", "index.js"]), "Error: Cannot find module 'express'", Os::Unix).unwrap();
        assert_eq!(line(&fix), "npm install && node index.js");

        let out = "Error: Cannot find module '/work/project/dist/server.js'";
        let fix = after_failure(p.path(), &argv(&["node", "dist/server.js"]), out, Os::Unix).unwrap();
        assert_eq!(line(&fix), "npm install && npm run build && node dist/server.js");

        let out = "TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension \".ts\"";
        let fix = after_failure(p.path(), &argv(&["node", "main.ts", "--fast"]), out, Os::Unix).unwrap();
        assert_eq!(fix.argv, argv(&["npx", "--yes", "tsx", "main.ts", "--fast"]));

        assert_eq!(after_failure(p.path(), &argv(&["node", "x.js"]), "SyntaxError: oops", Os::Unix), None);
    }

    #[test]
    fn a_failed_wheel_tries_older_pythons_then_the_newest_package() {
        let p = project(&[("requirements.txt", "psycopg2-binary==2.9.9\nflask")]);
        let cmd = argv(&["sh", "-c", "python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest && .venv/bin/python -m pytest"]);
        let out = "creating build/lib.macosx-27.0-arm64-cpython-315/psycopg2\n  ERROR: Failed building wheel for psycopg2-binary\n";
        let fix = after_failure(p.path(), &cmd, out, Os::Unix).unwrap();
        assert!(line(&fix).starts_with("PY=$(for v in 14 13 12 11 10 9; do if command -v python3.$v; then break; fi; done);"), "{}", line(&fix));
        assert!(line(&fix).ends_with("\"$PY\" -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest && .venv/bin/python -m pytest"));

        // Python 3.12 fails too: the next round starts below it.
        let out = "include/python3.12/Python.h\nFailed building wheel for psycopg2-binary\n";
        let again = after_failure(p.path(), &fix.argv, out, Os::Unix).unwrap();
        assert!(line(&again).starts_with("PY=$(for v in 11 10 9;"), "{}", line(&again));

        // No older Python left: the newest release of the package instead of the pin.
        let out = "bc: no older Python on this device for psycopg2-binary\n";
        let last = after_failure(p.path(), &again.argv, out, Os::Unix).unwrap();
        assert_eq!(
            line(&last),
            "python3 -m venv .venv && grep -iv '^psycopg2-binary' requirements.txt > .bc-requirements.txt; .venv/bin/pip install -q -r .bc-requirements.txt pytest && .venv/bin/pip install -q psycopg2-binary && .venv/bin/python -m pytest"
        );
        assert_eq!(after_failure(p.path(), &last.argv, out, Os::Unix), None, "nothing new left to try");

        // Names that are not plain package names never reach the shell.
        let out = "cpython-315\nFailed building wheel for x;rm -rf ~\n";
        assert_eq!(after_failure(p.path(), &cmd, out, Os::Unix), None);
    }

    #[test]
    fn a_missing_module_keeps_the_chosen_python_and_reads_subproject_requirements() {
        let p = project(&[("requirements-shared.txt", ""), ("gateway/", ""), ("gateway/requirements.txt", "flask-login"), ("services/", ""), ("services/geo/", ""), ("services/geo/requirements.txt", ""), ("venv/", "")]);
        let cmd = argv(&["sh", "-c", "PY=x; \"$PY\" -m venv .venv && .venv/bin/pip install -q -r requirements-shared.txt pytest && .venv/bin/python -m pytest"]);
        let fix = after_failure(p.path(), &cmd, "E   ModuleNotFoundError: No module named 'flask_login'\n", Os::Unix).unwrap();
        assert_eq!(
            line(&fix),
            "PY=x; \"$PY\" -m venv .venv && .venv/bin/pip install -q -r requirements-shared.txt pytest && .venv/bin/pip install -q -r gateway/requirements.txt -r services/geo/requirements.txt flask_login && .venv/bin/python -m pytest"
        );
    }

    #[test]
    fn quotes_arguments() {
        let p = project(&[("package.json", "{}")]);
        let fix = preflight(p.path(), &argv(&["node", "-e", "console.log('hi there')"]), Os::Unix).unwrap();
        assert_eq!(line(&fix), "npm install && node -e 'console.log('\\''hi there'\\'')'");
    }
}
