#!/usr/bin/env python3
"""
plot_minmax_temp_jk_ladakh.py
==============================
Min & Max 2m temperature for Jammu & Kashmir + Ladakh, in a single figure
(two side-by-side panels), using the combined JK+Ladakh GeoJSON boundary
approach (same fix applied to the rainfall script).

Usage:
  python plot_minmax_temp_jk_ladakh.py --input wrfout_d01_2024-06-01_00:00:00 --outdir ./output
"""

import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.path import shapely_to_path
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np
from datetime import timedelta

# ─── PATHS ─────────────────────────────────────────────────────────────────
SHAPEFILE_PATH  = "/home/ras_08/WEATHER/INDIA/WORKFLOW/JK&LADAKH/LADAKH_DISTRICTS.geojson"
SHAPEFILE_PATH2 = "/home/ras_08/WEATHER/INDIA/WORKFLOW/JK&LADAKH/JAMMU & KASHMIR_DISTRICTS.geojson"
LOGO_PATH       = "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"

# ─── DOMAIN ────────────────────────────────────────────────────────────────
LON_MIN, LON_MAX = 71.0, 80.5
LAT_MIN, LAT_MAX = 32.0, 37.0

# ─── COLOR SCALES ──────────────────────────────────────────────────────────
MIN_COLORS = [
    "#08306b", "#08519c", "#2171b5", "#4292c6",
    "#6baed6", "#9ecae1", "#c6dbef", "#deebf7",
    "#f7fbff", "#e0f3f8", "#abd9e9",
    "#74add1", "#4575b4", "#313695",
]
MIN_LEVELS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

MAX_COLORS = [
    "#4d004b", "#810f7c", "#88419d", "#8c6bb1", "#8c96c6",
    "#9ebcda", "#bfd3e6", "#edf8fb",
    "#ffffcc", "#ffeda0", "#fed976",
    "#feb24c", "#fd8d3c", "#fc4e2a",
    "#e31a1c", "#bd0026", "#800026",
]
MAX_LEVELS = [10, 14, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 42]


# ─── SHAPEFILE HELPERS (merged, per the earlier fix) ───────────────────────
def load_combined_shapefile():
    gdf1 = gpd.read_file(SHAPEFILE_PATH)   # Ladakh
    gdf2 = gpd.read_file(SHAPEFILE_PATH2)  # J&K

    for g in (gdf1, gdf2):
        if g.crs is None:
            g.set_crs(epsg=4326, inplace=True, allow_override=True)
        elif g.crs.to_epsg() != 4326:
            g.to_crs(epsg=4326, inplace=True)

    combined = pd.concat([gdf1, gdf2], ignore_index=True)
    return gpd.GeoDataFrame(combined, crs="EPSG:4326")


def add_logo(fig, x=0.01, y=0.85, w=0.10, h=0.10):
    logo = mpimg.imread(LOGO_PATH)
    logo_ax = fig.add_axes([x, y, w, h])
    logo_ax.imshow(logo)
    logo_ax.axis("off")


def apply_map_features(ax, gdf, india_path):
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#cce6ff", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=11)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.0, zorder=11)

    gdf.boundary.plot(ax=ax, edgecolor="#880000", linewidth=1.2, zorder=6)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.3, linestyle="--", zorder=14)
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = {"fontsize": 8}
    gl.ylabel_style = {"fontsize": 8}


def fmt_time(ts):
    return pd.to_datetime(str(ts)).strftime("%d %b %Y, %H:%M UTC")


def load_all_timestamps(ncfile, ntimes):
    return [
        pd.to_datetime(str(getvar(ncfile, "Times", timeidx=i).values))
        for i in range(ntimes)
    ]


def get_day_intervals(all_ts, cycle_hour=None):
    """Split into 24h buckets starting from the model's cycle hour."""
    if cycle_hour is None:
        cycle_hour = all_ts[0].hour

    bucket_idx = [i for i, ts in enumerate(all_ts) if ts.hour == cycle_hour]

    if not bucket_idx or bucket_idx[0] != 0:
        bucket_idx = [0] + bucket_idx

    bounds = bucket_idx + [len(all_ts) - 1]

    intervals = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if s == e:
            continue
        intervals.append((k + 1, s, e, all_ts[s], all_ts[e]))
    return intervals


def make_minmax_panel(fig, gdf, clip_path, lons, lats, min_data, max_data,
                       day_num, s_ts, e_ts):
    ax_min = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax_max = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

    min_cmap = ListedColormap(MIN_COLORS)
    min_norm = BoundaryNorm(MIN_LEVELS, min_cmap.N)
    max_cmap = ListedColormap(MAX_COLORS)
    max_norm = BoundaryNorm(MAX_LEVELS, max_cmap.N)

    for ax, data, cmap, norm, levels, label in [
        (ax_min, min_data, min_cmap, min_norm, MIN_LEVELS, "Minimum Temperature (°C)"),
        (ax_max, max_data, max_cmap, max_norm, MAX_LEVELS, "Maximum Temperature (°C)"),
    ]:
        apply_map_features(ax, gdf, clip_path)
        cf = ax.contourf(
            to_np(lons), to_np(lats), data,
            levels=levels, cmap=cmap, norm=norm, extend="both",
            transform=ccrs.PlateCarree(), zorder=2,
        )
        if hasattr(cf, "collections"):
            for col in cf.collections:
                col.set_clip_path(clip_path, transform=ax.transData)
        else:
            cf.set_clip_path(clip_path, transform=ax.transData)

        plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.06, shrink=0.85, label=label)
        ax.set_title(label, fontsize=12, fontweight="bold")

    fig.suptitle(
        f"J&K + Ladakh — Day {day_num} Min/Max Temperature\n"
        f"Valid: {fmt_time(s_ts)} to {fmt_time(e_ts)}",
        fontsize=14, fontweight="bold",
    )


def run(wrf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    daily_dir = os.path.join(output_dir, "jk_ladakh_daily_minmax")
    os.makedirs(daily_dir, exist_ok=True)

    ncfile = Dataset(wrf_path)

    gdf = load_combined_shapefile()
    india_border = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    paths = shapely_to_path(india_border)
    clip_path = paths[0] if isinstance(paths, list) else paths

    t2_raw = getvar(ncfile, "T2", timeidx=0)
    lats, lons = latlon_coords(t2_raw)

    ntimes = ncfile.dimensions["Time"].size
    all_ts = load_all_timestamps(ncfile, ntimes)
    day_intervals = get_day_intervals(all_ts, cycle_hour=all_ts[0].hour)

    for day_num, s_idx, e_idx, s_ts, e_ts in day_intervals:
        t2_block = getvar(ncfile, "T2", timeidx=None) - 273.15  # (time, y, x), deg C
        t2_block = to_np(t2_block)[s_idx:e_idx + 1]

        min_t2 = np.min(t2_block, axis=0)
        max_t2 = np.max(t2_block, axis=0)

        fig = plt.figure(figsize=(18, 10))
        add_logo(fig)
        make_minmax_panel(fig, gdf, clip_path, lons, lats, min_t2, max_t2,
                           day_num, s_ts, e_ts)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fname = f"jk_ladakh_minmax_Day{day_num}_{e_ts.strftime('%d%b%Y').upper()}.png"
        out_path = os.path.join(daily_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Day {day_num} -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to wrfout file")
    p.add_argument("--outdir", required=True, help="Output directory")
    args = p.parse_args()
    run(args.input, args.outdir)
