// Sets the app version in a tauri.conf.json: node set-version.mjs <file> <version>
import { readFileSync, writeFileSync } from "node:fs";

const [file, version] = process.argv.slice(2);
const config = JSON.parse(readFileSync(file, "utf8"));
config.version = version;
writeFileSync(file, JSON.stringify(config, null, 2) + "\n");
