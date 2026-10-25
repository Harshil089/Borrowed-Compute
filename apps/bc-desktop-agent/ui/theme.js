// Applies the saved appearance before the page paints, so a window never flashes the
// wrong theme. Light is the default; "system" follows the Mac or Windows setting.
(() => {
  let theme = "light";
  try { theme = localStorage.getItem("appearance") || "light"; } catch { /* storage off */ }
  document.documentElement.dataset.theme = theme;
})();

// Changes the appearance: the page, the saved choice and the native title bar together.
function setAppearance(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("appearance", theme); } catch { /* storage off */ }
  window.__TAURI__?.window?.getCurrentWindow()
    .setTheme(theme === "system" ? null : theme)
    .catch(() => {});
  document.querySelectorAll("[data-appearance]").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.appearance === theme));
  });
}

document.addEventListener("click", (event) => {
  const choice = event.target.closest("[data-appearance]");
  if (choice) setAppearance(choice.dataset.appearance);
});
document.addEventListener("DOMContentLoaded", () => setAppearance(document.documentElement.dataset.theme));
