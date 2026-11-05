"""Regenerate the README screenshots in docs/screenshots/ (optional, for documentation only).

Needs: pip install playwright && python -m playwright install chromium
Usage: start the app first, then run this script:

    streamlit run ds_app.py --server.headless true --server.port 8599
    python scripts/take_screenshots.py
"""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8599"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1900})
        page.goto(URL, wait_until="networkidle")
        time.sleep(4)
        page.screenshot(path=OUT / "00_start.png")

        page.get_by_role("button", name="Run inference").click()
        time.sleep(6)
        page.screenshot(path=OUT / "01_metrics.png")

        for i, tab in enumerate(["Plots", "By group", "Features", "Predictions", "Training"], start=2):
            page.get_by_role("tab", name=tab).click()
            time.sleep(4)
            page.screenshot(path=OUT / f"{i:02d}_{tab.lower().replace(' ', '_')}.png")
        browser.close()
    print(f"Screenshots saved to {OUT}")


if __name__ == "__main__":
    main()
