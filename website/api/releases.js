// Release history for the website, read from the public releases feed on the server so
// visitors never leave the site or call GitHub themselves. Notes written for people live in
// notes.json beside this file; a release without one gets a plain line, never a commit hash.
const notes = require("./notes.json");

const FEED = "https://api.github.com/repos/Harshil089/Borrowed-Compute-releases/releases?per_page=100";
const APPS = { agent: "Mac", client: "Windows" };

module.exports = async function handler(req, res) {
  try {
    const r = await fetch(FEED, { headers: { accept: "application/vnd.github+json", "user-agent": "borrowed-compute-site" } });
    if (!r.ok) throw new Error(`feed answered ${r.status}`);
    const list = (await r.json())
      .filter((x) => !x.draft && !x.prerelease && /^(agent|client)-v\d+\.\d+\.\d+$/.test(x.tag_name))
      .map((x) => {
        const [app, version] = x.tag_name.split("-v");
        return { app: APPS[app], key: app, version, date: x.published_at, notes: notes[x.tag_name] ?? ["Fixes and improvements."] };
      })
      .sort((a, b) => b.date.localeCompare(a.date));
    // Fresh for ten minutes at the edge, then served stale while it refreshes.
    res.setHeader("Cache-Control", "public, s-maxage=600, stale-while-revalidate=86400");
    res.status(200).json({ releases: list });
  } catch (e) {
    res.setHeader("Cache-Control", "public, s-maxage=60");
    res.status(502).json({ releases: [], error: "release history is unavailable right now" });
  }
};
