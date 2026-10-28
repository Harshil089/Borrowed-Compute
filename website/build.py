"""Assembles the static pages: each src/*.html page gets the shared head, icon sprite,
navigation and footer. Run after editing src/ or partials/; the output is committed and
served as is, so Vercel needs no build step."""
import pathlib, re

HERE = pathlib.Path(__file__).parent
SITE = "https://borrowed-compute.vercel.app"
part = lambda name: (HERE / "partials" / name).read_text()

for src in sorted((HERE / "src").glob("*.html")):
    text = src.read_text()
    meta = dict(re.findall(r"^(\w+): (.*)$", text.split("-->")[0], re.M))
    body = text.split("-->", 1)[1].strip()
    slug = src.stem
    url = SITE + ("/" if slug == "index" else f"/{slug}")
    header = part("header.html").replace(f'data-nav="{meta["nav"]}"', 'aria-current="page"')
    page = f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{meta["title"]}</title>
  <meta name="description" content="{meta["description"]}">
  <link rel="canonical" href="{url}">
  <meta name="theme-color" content="#07080a">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{meta["title"]}">
  <meta property="og:description" content="{meta["description"]}">
  <meta property="og:url" content="{url}">
  <meta property="og:image" content="{SITE}/og.png">
  <meta name="twitter:card" content="summary_large_image">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="preload" href="/fonts/archivo-latin.woff2" as="font" type="font/woff2" crossorigin>
  <script src="/theme.js"></script>
  <link rel="stylesheet" href="/styles.css">
  <script src="/site.js" defer></script>
</head>
<body>
{part("sprite.html")}
{header}
{body}
{part("footer.html")}
</body>
</html>
"""
    (HERE / f"{slug}.html").write_text(page)
    print("built", slug)
