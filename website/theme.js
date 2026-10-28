// Runs before the stylesheet paints, so a saved daylight choice never flashes dark first.
try { document.documentElement.dataset.theme = localStorage.getItem("bc-theme") === "light" ? "light" : "dark"; } catch (_) { document.documentElement.dataset.theme = "dark"; }
