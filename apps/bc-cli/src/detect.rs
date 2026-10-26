//! Recognizes a project from the files at its root and suggests commands to run it.
//!
//! The first suggestion is the default. Commands that need dependencies install them
//! inside the job's workspace (`npm ci`, a Python `.venv`), so they are deleted with it
//! and nothing from the project stays on the Mac. A `bc.toml` always wins over detection.

use std::path::Path;

use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Suggestion {
    /// What the command does, for a button: "Build", "Test", "Run main.py".
    pub label: String,
    pub argv: Vec<String>,
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

pub fn detect(root: &Path) -> Option<Detected> {
    let files = RootFiles::read(root);
    let mut found = rust(&files)
        .or_else(|| swift(&files))
        .or_else(|| xcode(&files))
        .or_else(|| node(root, &files))
        .or_else(|| typescript(&files))
        .or_else(|| go(&files))
        .or_else(|| python(root, &files))
        .or_else(|| cmake(&files))
        .or_else(|| jvm(&files))
        .or_else(|| dotnet(&files))
        .or_else(|| make(root, &files))
        .or_else(|| html(&files))
        .or_else(|| scripts(&files))?;
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
    Suggestion { label: label.to_string(), argv: argv.iter().map(|a| a.to_string()).collect() }
}

/// A command that needs a shell (`&&` between steps). Still an argv: `sh -c <script>`.
fn shell(label: &str, script: &str) -> Suggestion {
    s(label, &["sh", "-c", script])
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

fn node(root: &Path, f: &RootFiles) -> Option<Detected> {
    f.has("package.json").then_some(())?;
    let (pm, install) = if f.has("pnpm-lock.yaml") {
        ("pnpm", "pnpm install --frozen-lockfile")
    } else if f.has("yarn.lock") {
        ("yarn", "yarn install --frozen-lockfile")
    } else if f.has("bun.lockb") || f.has("bun.lock") {
        ("bun", "bun install")
    } else if f.has("package-lock.json") {
        ("npm", "npm ci")
    } else {
        ("npm", "npm install")
    };
    let package: serde_json::Value = std::fs::read(root.join("package.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let scripts = package.get("scripts").and_then(|v| v.as_object());
    let mut suggestions = Vec::new();
    for (name, label) in [("build", "Build"), ("test", "Test"), ("start", "Start"), ("lint", "Lint")] {
        if scripts.is_some_and(|s| s.contains_key(name)) {
            suggestions.push(shell(label, &format!("{install} && {pm} run {name}")));
        }
    }
    // A TypeScript project with no build script still has a compiler to run.
    let ts = f.has("tsconfig.json");
    if ts && !scripts.is_some_and(|s| s.contains_key("build")) {
        suggestions.insert(0, shell("Build", &format!("{install} && npx --yes -p typescript tsc -p .")));
    }
    if suggestions.is_empty() {
        let main = package.get("main").and_then(|v| v.as_str()).unwrap_or("index.js");
        suggestions.push(shell("Run", &format!("{install} && node {main}")));
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

fn python(root: &Path, f: &RootFiles) -> Option<Detected> {
    let scripts: Vec<&String> = f.with_ext(".py").collect();
    let reason = ["pyproject.toml", "requirements.txt", "setup.py"]
        .into_iter()
        .find(|n| f.has(n))
        .map(str::to_string)
        .or_else(|| scripts.first().map(|n| n.to_string()))?;

    // Dependencies go into a virtualenv inside the workspace, never into the Mac's Python.
    let setup = if f.has("requirements.txt") {
        Some("python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt")
    } else if f.has("pyproject.toml") || f.has("setup.py") {
        Some("python3 -m venv .venv && .venv/bin/pip install -q .")
    } else {
        None
    };
    let run = |label: &str, args: &str| match setup {
        Some(setup) => shell(label, &format!("{setup} && .venv/bin/python {args}")),
        None => s(label, &std::iter::once("python3").chain(args.split(' ')).collect::<Vec<_>>()),
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
        let test = match setup {
            Some(setup) => shell("Test", &format!("{setup} pytest && .venv/bin/python -m pytest")),
            None => shell("Test", "python3 -m venv .venv && .venv/bin/pip install -q pytest && .venv/bin/python -m pytest"),
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

fn cmake(f: &RootFiles) -> Option<Detected> {
    f.has("CMakeLists.txt").then_some(())?;
    found("C/C++ (CMake)", "CMakeLists.txt", vec![
        shell("Build", "cmake -S . -B build && cmake --build build --parallel"),
        shell("Test", "cmake -S . -B build && cmake --build build --parallel && ctest --test-dir build"),
    ])
}

fn jvm(f: &RootFiles) -> Option<Detected> {
    if f.has("gradlew") {
        return found("Gradle", "gradlew", vec![s("Build", &["./gradlew", "build"]), s("Test", &["./gradlew", "test"])]);
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
fn html(f: &RootFiles) -> Option<Detected> {
    let page = if f.has("index.html") { "index.html".to_string() } else { f.with_ext(".html").next()?.clone() };
    found("HTML", &page, vec![
        // The standard preset checks conformance to the HTML spec, not house style.
        s("Check HTML", &["npx", "--yes", "html-validate", "--preset", "standard", "**/*.html"]),
        s("Build with Vite", &["npx", "--yes", "vite", "build"]),
    ])
}

fn scripts(f: &RootFiles) -> Option<Detected> {
    let mut suggestions = Vec::new();
    for script in f.with_ext(".sh") {
        suggestions.push(s(&format!("Run {script}"), &["sh", script]));
    }
    for script in f.with_ext(".js").chain(f.with_ext(".mjs")) {
        suggestions.push(s(&format!("Run {script}"), &["node", script]));
    }
    let reason = suggestions.first()?.argv[1].clone();
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
    fn rust_builds_in_release() {
        let d = detect(project(&[("Cargo.toml", "")]).path()).unwrap();
        assert_eq!((d.kind.as_str(), d.reason.as_str()), ("Rust", "Cargo.toml"));
        assert_eq!(argv(&d, 0), "cargo build --release");
        assert_eq!(d.suggestions[1].label, "Test");
    }

    #[test]
    fn node_uses_lockfile_manager_and_scripts() {
        let pkg = r#"{"scripts":{"test":"vitest","build":"vite build"}}"#;
        let d = detect(project(&[("package.json", pkg), ("pnpm-lock.yaml", "")]).path()).unwrap();
        assert_eq!(d.kind, "Node (pnpm)");
        assert_eq!(argv(&d, 0), "sh -c pnpm install --frozen-lockfile && pnpm run build");
        assert_eq!(d.suggestions[1].label, "Test");

        let d = detect(project(&[("package.json", "{}"), ("package-lock.json", "")]).path()).unwrap();
        assert_eq!(argv(&d, 0), "sh -c npm ci && node index.js");
    }

    #[test]
    fn python_scripts_without_dependencies_run_directly() {
        let d = detect(project(&[("heavy_benchmark.py", ""), ("palindrome_checker.py", "")]).path()).unwrap();
        assert_eq!(d.kind, "Python");
        assert_eq!(argv(&d, 0), "python3 heavy_benchmark.py");
        assert_eq!(d.suggestions[1].label, "Run palindrome_checker.py");
    }

    #[test]
    fn python_dependencies_go_into_a_workspace_venv() {
        let d = detect(project(&[("requirements.txt", "numpy"), ("main.py", ""), ("tests/", "")]).path()).unwrap();
        assert_eq!(d.reason, "requirements.txt");
        assert_eq!(d.suggestions[0].label, "Run main.py");
        assert!(argv(&d, 0).contains("python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt"));
        assert!(d.suggestions.iter().any(|s| s.label == "Test"));
    }

    #[test]
    fn python_tests_lead_when_there_is_no_entry_point() {
        let d = detect(project(&[("pyproject.toml", "[tool.pytest.ini_options]"), ("lib.py", "")]).path()).unwrap();
        assert_eq!(d.suggestions[0].label, "Test");
    }

    #[test]
    fn makefile_targets_join_other_projects() {
        let d = detect(project(&[("go.mod", ""), ("Makefile", "build:\n\tgo build\ntest:\n\tgo test\n")]).path()).unwrap();
        assert_eq!(d.kind, "Go");
        assert!(d.suggestions.iter().any(|s| s.label == "make test"));
    }

    #[test]
    fn other_toolchains() {
        let kind = |files: &[(&str, &str)]| detect(project(files).path()).map(|d| (d.kind.clone(), argv_of(&d)));
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
        let d = detect(project(&[("package.json", r#"{"scripts":{"test":"vitest"}}"#), ("tsconfig.json", "")]).path()).unwrap();
        assert_eq!(d.kind, "TypeScript");
        assert_eq!(argv(&d, 0), "sh -c npm install && npx --yes -p typescript tsc -p .");
        assert_eq!(d.suggestions[1].label, "Test");

        let d = detect(project(&[("main.ts", ""), ("types.d.ts", "")]).path()).unwrap();
        assert_eq!((d.kind.as_str(), d.reason.as_str()), ("TypeScript", "main.ts"));
        assert_eq!(argv(&d, 0), "npx --yes tsx main.ts");
        assert_eq!(d.suggestions.len(), 1, "declaration files are not scripts");
    }

    #[test]
    fn html_site_gets_check_and_bundle() {
        let d = detect(project(&[("index.html", ""), ("about.html", ""), ("app.js", "")]).path()).unwrap();
        assert_eq!((d.kind.as_str(), d.reason.as_str()), ("HTML", "index.html"));
        assert_eq!(argv(&d, 0), "npx --yes html-validate --preset standard **/*.html");
        assert_eq!(d.suggestions[1].label, "Build with Vite");
    }

    #[test]
    fn unknown_or_empty_folder_detects_nothing() {
        assert!(detect(project(&[("README.md", "")]).path()).is_none());
        assert!(detect(Path::new("/definitely/not/here")).is_none());
    }
}
