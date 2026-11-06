// Runs before the stylesheet paints: a saved choice wins, otherwise the system's light or dark.
(() => {
  let t = null;
  try { t = localStorage.getItem("bc-theme"); } catch (_) {}
  if (t !== "light" && t !== "dark") t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = t;
})();
