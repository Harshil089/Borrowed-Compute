// Prints the update manifest the Tauri updater reads:
// node update-manifest.mjs <version> <platform> <package url> <signature file>
import { readFileSync } from "node:fs";

const [version, platform, url, sigFile] = process.argv.slice(2);
const manifest = {
  version,
  pub_date: new Date().toISOString().replace(/\.\d+Z$/, "Z"),
  platforms: { [platform]: { signature: readFileSync(sigFile, "utf8").trim(), url } },
};
console.log(JSON.stringify(manifest, null, 2));
