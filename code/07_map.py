"""
07_map.py
---------
Supplementary Figure 3: locations of the DC fast charging sessions in the
model sample (n = 1,360 sessions with complete covariates).

Coordinates are the nearest public charger to the location each respondent
reported (data/charger_coordinates.csv; latitude/longitude only, no link to
survey responses).  State and province boundaries come from Natural Earth
(1:50m Admin 1, public domain); the zip is downloaded on first use.

Outputs: outputs/figures/Supplementary_Figure_3.{pdf,png},
         outputs/source_data/suppfig3_charger_coordinates.csv
"""
import io
import ssl
import subprocess
import urllib.request
import zipfile

from common import DATA_DIR, FIG_DIR, PALETTE_SECONDARY, SD_DIR, write_csv

if not (DATA_DIR / "charger_coordinates.csv").exists():
    print("[map] data/charger_coordinates.csv not found (the coordinates are not distributed); "
          "skipping Supplementary Figure 3.")
    raise SystemExit(0)

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

NE_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip"
ne_dir = DATA_DIR / "external" / "ne_50m_admin_1_states_provinces"
shp = ne_dir / "ne_50m_admin_1_states_provinces.shp"
if not shp.exists():
    print(f"[map] downloading Natural Earth boundaries from {NE_URL}")
    ne_dir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(NE_URL, timeout=120, context=ctx) as r:
            payload = r.read()
    except Exception as exc:  # fall back to curl (e.g. missing CA bundle)
        print(f"[map] urllib failed ({exc}); trying curl")
        payload = subprocess.run(["curl", "-sL", NE_URL], check=True, capture_output=True).stdout
    zipfile.ZipFile(io.BytesIO(payload)).extractall(ne_dir)
admin1 = gpd.read_file(shp).to_crs("EPSG:4326")
us = admin1[admin1["admin"] == "United States of America"]
ca = admin1[admin1["admin"] == "Canada"]

pts = pd.read_csv(DATA_DIR / "charger_coordinates.csv")
write_csv(pts, SD_DIR / "suppfig3_charger_coordinates.csv")
g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["longitude"], pts["latitude"]), crs="EPSG:4326")

fig, ax = plt.subplots(figsize=(15, 8))
us.plot(ax=ax, facecolor="#EFEFEF", edgecolor="gray", linewidth=0.8, zorder=1)
ca.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.8, zorder=2)
g.plot(ax=ax, marker="o", color=PALETTE_SECONDARY, edgecolor="black", linewidth=0.4, alpha=0.75, markersize=28,
       zorder=10, label=f"Charger location ($n$ = {len(g):,})")
ax.set_xlim(-130, -66); ax.set_ylim(24, 52)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right")
ax_h = inset_axes(ax, width="34%", height="46%", bbox_to_anchor=(0.02, 0.02, 0.45, 0.46),
                  bbox_transform=ax.transAxes, loc="lower left", borderpad=1.1)
us.plot(ax=ax_h, facecolor="#EFEFEF", edgecolor="gray", linewidth=0.6, zorder=1)
g.plot(ax=ax_h, marker="o", color=PALETTE_SECONDARY, edgecolor="black", linewidth=0.4, alpha=0.85, markersize=28, zorder=10)
ax_h.set_xlim(-160.5, -154.5); ax_h.set_ylim(18.5, 22.5)
ax_h.set_title("Hawaii", fontsize=10)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"Supplementary_Figure_3.{ext}", dpi=200, bbox_inches="tight")
print(f"[fig]  outputs/figures/Supplementary_Figure_3.pdf (+png); n = {len(g)} charger locations")
print("[done] 07_map.py")
