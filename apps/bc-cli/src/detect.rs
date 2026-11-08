//! Recognizes a project from the files at its root and suggests commands to run it.
//!
//! The first suggestion is the default. Commands that need dependencies install them
//! inside the job's workspace (`npm ci`, a Python `.venv`), so they are deleted with it
//! and nothing from the project stays on the device. A `bc.toml` always wins over
//! detection. Commands are written for the device that runs them ([`Os`]): `cmd /c` and
//! `.venv\Scripts` on a Windows PC, `sh -c` and `.venv/bin` elsewhere.

use std::path::Path;

use bc_proto::cloud::DeviceKind;
use serde::Serialize;

/// How commands are written for the device a job runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Os {
    /// A Mac (or Linux): `sh`, `python3`, `.venv/bin`. Also the guess for a device of
    /// unknown kind, such as a shared Mac.
    #[default]
    Unix,
    Windows,
}

impl Os {
    pub fn of(kind: Option<DeviceKind>) -> Self {
        if kind == Some(DeviceKind::Windows) { Os::Windows } else { Os::Unix }
    }

    /// A command that needs a shell (`&&` between steps), still as an argv.
    pub fn shell(self, script: &str) -> Vec<String> {
        match self {
            Os::Unix => vec!["sh".into(), "-c".into(), script.into()],
            Os::Windows => vec!["cmd".into(), "/d".into(), "/c".into(), script.into()],
        }
    }

    /// The script of a command [`Os::shell`] wrapped, from either system.
    pub fn shell_script(argv: &[String]) -> Option<&str> {
        match argv {
            [sh, c, script] if (sh == "sh" || sh == "bash") && c == "-c" => Some(script),
            [cmd, d, c, script] if cmd == "cmd" && d == "/d" && c == "/c" => Some(script),
            [cmd, c, script] if cmd == "cmd" && c.eq_ignore_ascii_case("/c") => Some(script),
            _ => None,
        }
    }

    pub fn python(self) -> &'static str {
        match self {
            Os::Unix => "python3",
            Os::Windows => "python",
        }
    }

    /// Where a virtualenv keeps its programs, relative to the project.
    pub fn venv_bin(self) -> &'static str {
        match self {
            Os::Unix => ".venv/bin/",
            Os::Windows => ".venv\\Scripts\\",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Suggestion {
    /// What the command does, for a button: "Build", "Test", "Run main.py".
    pub label: String,
    pub argv: Vec<String>,
    /// A server or watcher that runs until stopped, so it should run as a service.
    pub service: bool,
}

impl Suggestion {
    fn service(mut self) -> Self {
        self.service = true;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Detected {
    /// "Rust", "Node (pnpm)", "Python", …
    pub kind: String,
    /// The file that gave it away, such as "Cargo.toml".
    pub reason: String,
    /// The first entry is the default command.
    pub suggestions: Vec<Suggestion>,
}

const MAX_SUGGESTIONS: usize = 6;

/// The project at `root` and commands for it, written for a device running `os`.
pub fn detect(root: &Path, os: Os) -> Option<Detected> {
    let files = RootFiles::read(root);
    let apple = os == Os::Unix;
    let mut found = rust(&files)
        .or_else(|| swift(&files).filter(|_| apple))
        .or_else(|| xcode(&files).filter(|_| apple))
        .or_else(|| node(root, &files, os))
        .or_else(|| typescript(&files))
        .or_else(|| go(&files))
        .or_else(|| python(root, &files, os))
        .or_else(|| cmake(&files, os))
        .or_else(|| jvm(&files, os))
        .or_else(|| dotnet(&files))
        .or_else(|| make(root, &files))
        .or_else(|| html(&files, os))
        .or_else(|| scripts(&files, os))?;
    // A Makefile next to another project usually holds its real entry points.
    if found.kind != "Make" && files.has("Makefile") {
        if let Some(m) = make(root, &files) {
            found.suggestions.extend(m.suggestions);
        }
    }
    found.suggestions.dedup_by(|a, b| a.argv == b.argv);
    found.suggestions.truncate(MAX_SUGGESTIONS);
    Some(found)
}

/// Names of the files and folders directly inside the project root.
struct RootFiles(Vec<String>);

impl RootFiles {
    fn read(root: &Path) -> Self {
        let mut names: Vec<String> = std::fs::read_dir(root)
            .map(|dir| dir.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        names.sort();
        RootFiles(names)
    }

    fn has(&self, name: &str) -> bool {
        self.0.iter().any(|n| n == name)
    }

    fn with_ext<'a>(&'a self, ext: &'a str) -> impl Iterator<Item = &'a String> + 'a {
        self.0.iter().filter(move |n| n.ends_with(ext) && !n.starts_with('.'))
    }
}

fn s(label: &str, argv: &[&str]) -> Suggestion {
    Suggestion { label: label.to_string(), argv: argv.iter().map(|a| a.to_string()).collect(), service: false }
}

/// A command that needs a shell (`&&` between steps). Still an argv: `sh -c <script>`, or
/// `cmd /d /c <script>` on Windows.
fn shell(label: &str, script: &str, os: Os) -> Suggestion {
    Suggestion { label: label.to_string(), argv: os.shell(script), service: false }
}

fn found(kind: &str, reason: &str, suggestions: Vec<Suggestion>) -> Option<Detected> {
    Some(Detected { kind: kind.to_string(), reason: reason.to_string(), suggestions })
}

fn rust(f: &RootFiles) -> Option<Detected> {
    f.has("Cargo.toml").then_some(())?;
    found("Rust", "Cargo.toml", vec![
        s("Build", &["cargo", "build", "--release"]),
        s("Test", &["cargo", "test"]),
        s("Run", &["cargo", "run", "--release"]),
    ])
}

/// SwiftPM and Xcode keep their caches in ~/Library no matter what the environment says,
/// so their commands name a folder inside the project instead. The project is deleted
/// with the job's workspace, so these caches are too.
const SWIFTPM_CACHE: &str = ".bc/swiftpm-cache";
const XCODE_DERIVED_DATA: &str = ".bc/DerivedData";
const XCODE_PACKAGES: &str = ".bc/SourcePackages";

fn swift(f: &RootFiles) -> Option<Detected> {
    f.has("Package.swift").then_some(())?;
    found("Swift", "Package.swift", vec![
        s("Build", &["swift", "build", "-c", "release", "--cache-path", SWIFTPM_CACHE]),
        s("Test", &["swift", "test", "--cache-path", SWIFTPM_CACHE]),
    ])
}

fn xcode(f: &RootFiles) -> Option<Detected> {
    let project = f.with_ext(".xcodeproj").next()?;
    found("Xcode", project, vec![s("Build", &[
        "xcodebuild", "-project", project,
        "-derivedDataPath", XCODE_DERIVED_DATA,
        "-clonedSourcePackagesDirPath", XCODE_PACKAGES,
        "build",
    ])])
}

/// The package manager a Node project uses, from its lockfile, and its install command.
/// `None` without a package.json.
pub fn node_package_manager(root: &Path) -> Option<(&'static str, &'static str)> {
    let f = RootFiles::read(root);
    f.has("package.json").then_some(())?;
    Some(if f.has("pnpm-lock.yaml") {
        ("pnpm", "pnpm install --frozen-lockfile")
    } else if f.has("yarn.lock") {
        ("yarn", "yarn install --frozen-lockfile")
    } else if f.has("bun.lockb") || f.has("bun.lock") {
        ("bun", "bun install")
    } else if f.has("package-lock.json") {
        ("npm", "npm ci")
    } else {
        ("npm", "npm install")
    })
}

/// The virtualenv setup for a Python project with declared dependencies, run inside
/// the job's workspace. `None` when it declares none.
pub fn python_setup(root: &Path, os: Os) -> Option<String> {
    let deps = if let Some(r) = requirements(root) {
        format!("-r {r}")
    } else if root.join("pyproject.toml").is_file() || root.join("setup.py").is_file() {
        ".".to_string()
    } else {
        return None;
    };
    Some(format!("{} -m venv .venv && {}pip install -q {deps}", os.python(), os.venv_bin()))
}

/// `requirements.txt`, else another `requirements*.txt` (`requirements-shared.txt`).
fn requirements(root: &Path) -> Option<String> {
    let f = RootFiles::read(root);
    if f.has("requirements.txt") {
        return Some("requirements.txt".into());
    }
    f.0.into_iter().find(|n| n.starts_with("requirements") && n.ends_with(".txt"))
}

fn node(root: &Path, f: &RootFiles, os: Os) -> Option<Detected> {
    let (pm, install) = node_package_manager(root)?;
    let package: serde_json::Value = std::fs::read(root.join("package.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let scripts = package.get("scripts").and_then(|v| v.as_object());
    let mut suggestions = Vec::new();
    let servers = ["dev", "start", "serve", "preview"];
    for (name, label) in
        [("build", "Build"), ("test", "Test"), ("dev", "Dev server"), ("start", "Start"), ("serve", "Serve"), ("preview", "Preview"), ("lint", "Lint")]
    {
        if scripts.is_some_and(|s| s.contains_key(name)) {
            let suggestion = shell(label, &format!("{install} && {pm} run {name}"), os);
            suggestions.push(if servers.contains(&name) { suggestion.service() } else { suggestion });
        }
    }
    // A TypeScript project with no build script still has a compiler to run.
    let ts = f.has("tsconfig.json");
    if ts && !scripts.is_some_and(|s| s.contains_key("build")) {
        suggestions.insert(0, shell("Build", &format!("{install} && npx --yes -p typescript tsc -p ."), os));
    }
    if suggestions.is_empty() {
        let main = package.get("main").and_then(|v| v.as_str()).unwrap_or("index.js");
        suggestions.push(shell("Run", &format!("{install} && node {main}"), os));
    }
    let lang = if ts { "TypeScript" } else { "Node" };
    let kind = if pm == "npm" { lang.to_string() } else { format!("{lang} ({pm})") };
    found(&kind, "package.json", suggestions)
}

/// TypeScript without a package.json: the compiler and tsx come from npx, into the
/// job's isolated npm cache.
fn typescript(f: &RootFiles) -> Option<Detected> {
    let scripts: Vec<&String> = f.with_ext(".ts").filter(|n| !n.ends_with(".d.ts")).collect();
    let mut suggestions = Vec::new();
    if f.has("tsconfig.json") {
        suggestions.push(s("Build", &["npx", "--yes", "-p", "typescript", "tsc", "-p", "."]));
    }
    for script in &scripts {
        suggestions.push(s(&format!("Run {script}"), &["npx", "--yes", "tsx", script]));
    }
    let reason = if f.has("tsconfig.json") { "tsconfig.json".to_string() } else { scripts.first()?.to_string() };
    found("TypeScript", &reason, suggestions)
}

fn go(f: &RootFiles) -> Option<Detected> {
    f.has("go.mod").then_some(())?;
    found("Go", "go.mod", vec![s("Build", &["go", "build", "./..."]), s("Test", &["go", "test", "./..."])])
}

fn python(root: &Path, f: &RootFiles, os: Os) -> Option<Detected> {
    let scripts: Vec<&String> = f.with_ext(".py").collect();
    let reason = Some("pyproject.toml".to_string())
        .filter(|n| f.has(n))
        .or_else(|| requirements(root))
        .or_else(|| Some("setup.py".to_string()).filter(|n| f.has(n)))
        .or_else(|| scripts.first().map(|n| n.to_string()))?;

    // Dependencies go into a virtualenv inside the workspace, never into the device's Python.
    let setup = python_setup(root, os);
    let (python, bin) = (os.python(), os.venv_bin());
    let run = |label: &str, args: &str| match &setup {
        Some(setup) => shell(label, &format!("{setup} && {bin}python {args}"), os),
        None => s(label, &std::iter::once(python).chain(args.split(' ')).collect::<Vec<_>>()),
    };

    let mut suggestions = Vec::new();
    // Likely entry points first, then every other script at the root.
    let mut entries: Vec<&String> = ["main.py", "app.py", "__main__.py"]
        .iter()
        .filter_map(|name| scripts.iter().find(|n| n.as_str() == *name).copied())
        .collect();
    let rest: Vec<&String> = scripts.iter().filter(|n| !entries.contains(n)).copied().collect();
    entries.extend(rest);
    for script in entries {
        suggestions.push(run(&format!("Run {script}"), script));
    }
    let has_tests = f.has("tests") || f.has("test") || f.has("conftest.py") || f.has("pytest.ini")
        || std::fs::read_to_string(root.join("pyproject.toml")).is_ok_and(|t| t.contains("pytest"));
    if has_tests {
        let test = match &setup {
            Some(setup) => shell("Test", &format!("{setup} pytest && {bin}python -m pytest"), os),
            None => shell("Test", &format!("{python} -m venv .venv && {bin}pip install -q pytest && {bin}python -m pytest"), os),
        };
        // With no obvious entry point, tests are the better default.
        if !["main.py", "app.py", "__main__.py"].iter().any(|n| f.has(n)) {
            suggestions.insert(0, test);
        } else {
            suggestions.push(test);
        }
    }
    if suggestions.is_empty() {
        return None;
    }
    found("Python", &reason, suggestions)
}

fn cmake(f: &RootFiles, os: Os) -> Option<Detected> {
    f.has("CMakeLists.txt").then_some(())?;
    found("C/C++ (CMake)", "CMakeLists.txt", vec![
        shell("Build", "cmake -S . -B build && cmake --build build --parallel", os),
        shell("Test", "cmake -S . -B build && cmake --build build --parallel && ctest --test-dir build", os),
    ])
}

fn jvm(f: &RootFiles, os: Os) -> Option<Detected> {
    // The wrapper is a shell script, with a batch file beside it for Windows.
    let wrapper = match os {
        Os::Unix => ("gradlew", "./gradlew"),
        Os::Windows => ("gradlew.bat", "./gradlew.bat"),
    };
    if f.has(wrapper.0) {
        return found("Gradle", wrapper.0, vec![s("Build", &[wrapper.1, "build"]), s("Test", &[wrapper.1, "test"])]);
    }
    if f.has("build.gradle") || f.has("build.gradle.kts") {
        return found("Gradle", "build.gradle", vec![s("Build", &["gradle", "build"]), s("Test", &["gradle", "test"])]);
    }
    f.has("pom.xml").then_some(())?;
    found("Maven", "pom.xml", vec![s("Build", &["mvn", "-B", "package"]), s("Test", &["mvn", "-B", "test"])])
}

fn dotnet(f: &RootFiles) -> Option<Detected> {
    let file = f.with_ext(".sln").chain(f.with_ext(".csproj")).next()?;
    found(".NET", file, vec![s("Build", &["dotnet", "build"]), s("Test", &["dotnet", "test"])])
}

fn make(root: &Path, f: &RootFiles) -> Option<Detected> {
    f.has("Makefile").then_some(())?;
    let text = std::fs::read_to_string(root.join("Makefile")).unwrap_or_default();
    let has_target = |t: &str| text.lines().any(|l| l.starts_with(&format!("{t}:")));
    let mut suggestions = vec![s("make", &["make"])];
    for target in ["build", "test", "run"] {
        if has_target(target) {
            suggestions.push(s(&format!("make {target}"), &["make", target]));
        }
    }
    found("Make", "Makefile", suggestions)
}

/// A static site: nothing to compile or run as a batch job, so it gets a check and an
/// optional bundle. html-validate and Vite come from npx, into the job's npm cache.
fn html(f: &RootFiles, os: Os) -> Option<Detected> {
    let page = if f.has("index.html") { "index.html".to_string() } else { f.with_ext(".html").next()?.clone() };
    found("HTML", &page, vec![
        // The standard preset checks conformance to the HTML spec, not house style.
        s("Check HTML", &["npx", "--yes", "html-validate", "--preset", "standard", "**/*.html"]),
        s("Build with Vite", &["npx", "--yes", "vite", "build"]),
        // Port forwarding brings it to this computer's http://localhost:8000.
        s("Serve", &[os.python(), "-m", "http.server", "8000", "-b", "127.0.0.1"]).service(),
    ])
}

fn scripts(f: &RootFiles, os: Os) -> Option<Detected> {
    let mut suggestions = Vec::new();
    if os == Os::Unix {
        for script in f.with_ext(".sh") {
            suggestions.push(s(&format!("Run {script}"), &["sh", script]));
        }
    } else {
        for script in f.with_ext(".cmd").chain(f.with_ext(".bat")) {
            suggestions.push(s(&format!("Run {script}"), &[&format!("./{script}")]));
        }
        for script in f.with_ext(".ps1") {
            suggestions.push(s(&format!("Run {script}"), &["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script]));
        }
    }
    for script in f.with_ext(".js").chain(f.with_ext(".mjs")) {
        suggestions.push(s(&format!("Run {script}"), &["node", script]));
    }
    let first = &suggestions.first()?.argv;
    let reason = first.last().unwrap_or(&first[0]).trim_start_matches("./").to_string();
    found("Scripts", &reason, suggestions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            let path = dir.path().join(name);
            if name.ends_with('/') {
                std::fs::create_dir_all(&path).unwrap();
            } else {
                std::fs::write(path, body).unwrap();
            }
        }
        dir
    }

    fn argv(d: &Detected, i: usize) -> String {
        d.suggestions[i].argv.join(" ")
    }

    #[test]
    fn commands_fit_a_windows_pc() {
        let win = |files: &[(&str, &str)]| detect(project(files).path(), Os::Windows).unwrap();
        let pkg = r#"{"scripts":{"build":"vite build"}}"#;
        assert_eq!(argv(&win(&[("package.json", pkg), ("package-lock.json", "")]), 0), "cmd /d /c npm ci && npm run build");
        let d = win(&[("requirements.txt", "numpy"), ("main.py", "")]);
        assert_eq!(argv(&d, 0), "cmd /d /c python -m venv .venv && .venv\\Scripts\\pip install -q -r requirements.txt && .venv\\Scripts\\python main.py");
        assert_eq!(argv(&win(&[("gradlew", ""), ("gradlew.bat", "")]), 0), "./gradlew.bat build");
        let d = win(&[("build.cmd", ""), ("run.sh", "")]);
        assert_eq!((d.reason.as_str(), argv(&d, 0).as_str()), ("build.cmd", "./build.cmd"));
        assert_eq!(win(&[("Package.swift", ""), ("main.py", "")]).kind, "Python");
        assert_eq!(Os::shell_script(&Os::Windows.shell("a && b")), Some("a && b"));
    }

    #[test]
    fn rust_builds_in_release() {
        let d = detect(project(&[("Cargo.toml", "")]).path(), Os::Unix).unwrap();
        assert_eq!((d.kind.as_str(), d.reason.as_str()), ("Rust", "Cargo.toml"));
        assert_eq!(argv(&d, 0), "cargo build --release");
        assert_eq!(d.suggestions[1].label, "Test");
    }

    #[test]
    fn node_uses_lockfile_manager_and_scripts() {
        let pkg = r#"{"scripts":{"test":"vitest","build":"vite build"}}"#;
        let d = detect(project(&[("package.json", pkg), ("pnpm-lock.yaml", "")]).path(), Os::Unix).unwrap();
        assert_eq!(d.kind, "Node (pnpm)");
        assert_eq!(argv(&d, 0), "sh -c pnpm install --frozen-lockfile && pnpm run build");
        assert_eq!(d.suggestions[1].label, "Test");

        let d = detect(project(&[("package.json", "{}"), ("package-lock.json", "")]).path(), Os::Unix).unwrap();
        assert_eq!(argv(&d, 0), "sh -c npm ci && node index.js");
    }

    #[test]
    fn python_scripts_without_dependencies_run_directly() {
        let d = detect(project(&[("heavy_benchmark.py", ""), ("palindrome_checker.py", "")]).path(), Os::Unix).unwrap();
        assert_eq!(d.kind, "Python");
        assert_eq!(argv(&d, 0), "python3 heavy_benchmark.py");
        assert_eq!(d.suggestions[1].label, "Run palindrome_checker.py");
    }

    #[test]
    fn python_dependencies_go_into_a_workspace_venv() {
        let d = detect(project(&[("requirements.txt", "numpy"), ("main.py", ""), ("tests/", "")]).path(), Os::Unix).unwrap();
        assert_eq!(d.reason, "requirements.txt");
        assert_eq!(d.suggestions[0].label, "Run main.py");
        assert!(argv(&d, 0).contains("python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt"));
        assert!(d.suggestions.iter().any(|s| s.label == "Test"));
    }

    #[test]
    fn python_with_other_requirements_name_is_found() {
        let d = detect(project(&[("requirements-shared.txt", "flask"), ("tests/", "")]).path(), Os::Unix).unwrap();
        assert_eq!(d.reason, "requirements-shared.txt");
        assert!(argv(&d, 0).contains("pip install -q -r requirements-shared.txt pytest"));
    }

    #[test]
    fn python_tests_lead_when_there_is_no_entry_point() {
        let d = detect(project(&[("pyproject.toml", "[tool.pytest.ini_options]"), ("lib.py", "")]).path(), Os::Unix).unwrap();
        assert_eq!(d.suggestions[0].label, "Test");
    }

    #[test]
    fn makefile_targets_join_other_projects() {
        let d = detect(project(&[("go.mod", ""), ("Makefile", "build:\n\tgo build\ntest:\n\tgo test\n")]).path(), Os::Unix).unwrap();
        assert_eq!(d.kind, "Go");
        assert!(d.suggestions.iter().any(|s| s.label == "make test"));
    }

    #[test]
    fn other_toolchains() {
        let kind = |files: &[(&str, &str)]| detect(project(files).path(), Os::Unix).map(|d| (d.kind.clone(), argv_of(&d)));
        fn argv_of(d: &Detected) -> String { d.suggestions[0].argv.join(" ") }
        assert_eq!(kind(&[("Package.swift", "")]).unwrap().1, "swift build -c release --cache-path .bc/swiftpm-cache");
        assert_eq!(
            kind(&[("App.xcodeproj/", "")]).unwrap().1,
            "xcodebuild -project App.xcodeproj -derivedDataPath .bc/DerivedData \
             -clonedSourcePackagesDirPath .bc/SourcePackages build"
        );
        assert_eq!(kind(&[("CMakeLists.txt", "")]).unwrap().0, "C/C++ (CMake)");
        assert_eq!(kind(&[("gradlew", "")]).unwrap().1, "./gradlew build");
        assert_eq!(kind(&[("pom.xml", "")]).unwrap().1, "mvn -B package");
        assert_eq!(kind(&[("App.csproj", "")]).unwrap().1, "dotnet build");
        assert_eq!(kind(&[("run.sh", "")]).unwrap().1, "sh run.sh");
    }

    #[test]
    fn typescript_with_and_without_package_json() {
        let d = detect(project(&[("package.json", r#"{"scripts":{"test":"vitest"}}"#), ("tsconfig.json", "")]).path(), Os::Unix).unwrap();
        assert_eq!(d.kind, "TypeScript");
        assert_eq!(argv(&d, 0), "sh -c npm install && npx --yes -p typescript tsc -p .");
        assert_eq!(d.suggestions[1].label, "Test");

        let d = detect(project(&[("main.ts", ""), ("types.d.ts", "")]).path(), Os::Unix).unwrap();
        assert_eq!((d.kind.as_str(), d.reason.as_str()), ("TypeScript", "main.ts"));
        assert_eq!(argv(&d, 0), "npx --yes tsx main.ts");
        assert_eq!(d.suggestions.len(), 1, "declaration files are not scripts");
    }

    #[test]
    fn html_site_gets_check_and_bundle() {
        let d = detect(project(&[("index.html", ""), ("about.html", ""), ("app.js", "")]).path(), Os::Unix).unwrap();
        assert_eq!((d.kind.as_str(), d.reason.as_str()), ("HTML", "index.html"));
        assert_eq!(argv(&d, 0), "npx --yes html-validate --preset standard **/*.html");
        assert_eq!(d.suggestions[1].label, "Build with Vite");
        assert!(d.suggestions[2].service && !d.suggestions[0].service);
    }

    #[test]
    fn node_dev_servers_run_as_services() {
        let pkg = r#"{"scripts":{"build":"vite build","dev":"vite","start":"node server.js"}}"#;
        let d = detect(project(&[("package.json", pkg)]).path(), Os::Unix).unwrap();
        let services: Vec<(&str, bool)> = d.suggestions.iter().map(|s| (s.label.as_str(), s.service)).collect();
        assert_eq!(services, [("Build", false), ("Dev server", true), ("Start", true)]);
    }

    #[test]
    fn unknown_or_empty_folder_detects_nothing() {
        assert!(detect(project(&[("README.md", "")]).path(), Os::Unix).is_none());
        assert!(detect(Path::new("/definitely/not/here"), Os::Unix).is_none());
    }
}
