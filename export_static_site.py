"""Export the Flask presentation pages for GitHub Pages hosting."""

from pathlib import Path
import re
import shutil

from app import app


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "docs"
REPOSITORY_PATH = "/tomato-growth-stage-classifier"
PAGES = {
    "/": OUTPUT_ROOT / "index.html",
    "/research": OUTPUT_ROOT / "research/index.html",
    "/dataset": OUTPUT_ROOT / "dataset/index.html",
    "/results": OUTPUT_ROOT / "results/index.html",
    "/predictor": OUTPUT_ROOT / "predictor/index.html",
    "/reports": OUTPUT_ROOT / "reports/index.html",
    "/references": OUTPUT_ROOT / "references/index.html",
}


def add_static_notice(html: str, route: str) -> str:
    if route not in {"/predictor", "/reports"}:
        return html

    notice = (
        '<div style="margin:16px auto;max-width:1180px;padding:12px 18px;'
        'border-radius:14px;background:#fff3cd;color:#664d03;font-weight:700;">'
        "GitHub Pages presentation mode: live prediction and generated downloads "
        "require the local Flask application described in the README.</div>"
    )
    html = html.replace("<main", notice + "<main", 1)
    html = html.replace("<form method=\"post\"", '<form method="post" onsubmit="return false;"')
    html = html.replace("<form method=\"get\"", '<form method="get" onsubmit="return false;"')
    return re.sub(
        rf'href="{REPOSITORY_PATH}/download-[^"]+"',
        'href="#" onclick="alert(\'Available in the local Flask app.\'); return false;"',
        html,
    )


def export_pages() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / ".nojekyll").touch()

    with app.test_client() as client:
        for route, destination in PAGES.items():
            response = client.get(route, environ_overrides={"SCRIPT_NAME": REPOSITORY_PATH})
            if response.status_code != 200:
                raise RuntimeError(f"Could not export {route}: HTTP {response.status_code}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            html = add_static_notice(response.get_data(as_text=True), route)
            destination.write_text(html, encoding="utf-8")

    shutil.copytree(PROJECT_ROOT / "demo_data", OUTPUT_ROOT / "dataset-image")


if __name__ == "__main__":
    export_pages()
    print(f"Static website exported to {OUTPUT_ROOT}")
