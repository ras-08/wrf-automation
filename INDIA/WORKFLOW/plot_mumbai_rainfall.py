#!/usr/bin/env python3
"""
plot_mumbai_rainfall.py
========================
Standalone WRF rainfall plotting script for Mumbai using a local shapefile.

Outputs (saved to --outdir):
  • mumbai_cumulative_rainfall.png
  • mumbai_daily_rainfall/mumbai_daily_DayN_DDMONYYYY.png   (one per calendar day)
  • mumbai_hourly_animation.mp4

Usage: 
  python plot_mumbai_rainfall.py --input /path/to/wrfout_d01_* --outdir ./output
  python plot_mumbai_rainfall.py --input /path/to/wrfout_d01_* --outdir ./output --no-animation

Dependencies:
  pip install netCDF4 wrf-python geopandas shapely cartopy matplotlib pandas numpy

Author : ras_08
Version: 1.0
"""

__version__ = "1.0"
__author__  = "ras_08"

import os
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.geometry import box
from shapely.ops import unary_union
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np
import pandas as pd
from scipy.ndimage import zoom
import matplotlib.image as mpimg


# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── PATHS ────────────────────────────────────────────────────────────────────
# Update this to match where you placed the shapefile on your machine
SHAPEFILE_PATH = os.path.join(
    os.path.dirname(__file__),
     "/home/ras_08/WEATHER/INDIA/WORKFLOW/MMRDA/MMRDA_Manual.shp"
)
SHAPEFILE_PATH2 = "/home/ras_08/WEATHER/INDIA/WORKFLOW/mumbai_ward/mumbai.shp"

# ─── MAP DOMAIN — tight around Mumbai ─────────────────────────────────────────
LON_MIN, LON_MAX = 72.70, 73.10
LAT_MIN, LAT_MAX = 18.85, 19.40

# ─── NWP METADATA ─────────────────────────────────────────────────────────────
NWP_RES    = "9 Km"
MODEL_NAME = "WFS (WeatherEx Forecasting System)"

# ─── IMD RAINFALL COLOR SCALE ─────────────────────────────────────────────────
COLORS = [
    "#ffff99",  # 1   – 2.5  mm
    "#CECE0C",  # 2.5 – 10   mm
    "#0BFF0B",  # 10  – 20   mm
    "#1B5E20",  # 20  – 40   mm
    "#00B7FF",  # 40  – 70   mm
    "#0A2683",  # 70  – 130  mm
    "#F3931D",  # 130 – 200  mm
    "#FF0000",  # 200 – 300  mm
]
LEVELS = [1, 2.5, 10, 20, 40, 70, 130, 200, 300]

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_colormap():
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    return cmap, norm


def fmt_time(ts: pd.Timestamp) -> str:
    return ts.strftime("%H UTC %d-%b-%Y")


def load_shapefile() -> gpd.GeoDataFrame:
    """Load Mumbai shapefile. Reprojects to EPSG:4326 if needed."""
    if not os.path.exists(SHAPEFILE_PATH):
        raise FileNotFoundError(
            f"Shapefile not found: {SHAPEFILE_PATH}\n"
            f"Please update SHAPEFILE_PATH in the script, or place\n"
            f"/home/ras_08/WEATHER/INDIA/WORKFLOW/mumbai/Mumbai_Shapefile.shp in the same folder as this script."
        )
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if gdf.crs is None:
        log.warning("Shapefile has no CRS — assuming EPSG:4326")
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        log.info(f"Reprojecting from {gdf.crs} → EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)
    log.info(f"Shapefile loaded: {len(gdf)} feature(s), columns: {list(gdf.columns)}")
    return gdf


def make_outer_mask(gdf: gpd.GeoDataFrame):
    """Returns the polygon OUTSIDE the Mumbai boundary (for masking)."""
    outer = box(60.0, 5.0, 100.0, 40.0)
    union = unary_union(gdf.geometry.values)
    return outer.difference(union)


# ══════════════════════════════════════════════════════════════════════════════
#  MAP DECORATION
# ══════════════════════════════════════════════════════════════════════════════

def apply_map_features(ax, gdf: gpd.GeoDataFrame, outer_mask):
    """Apply coastlines, mask, district/state boundaries, gridlines."""

    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    # Background
    ax.add_feature(cfeature.LAND,      facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce6ff",    zorder=0)
    #ax.add_feature(cfeature.COASTLINE, linewidth=1.2,           zorder=5)
    #ax.add_feature(cfeature.RIVERS,    linewidth=0.5, edgecolor="#7ec8e3", zorder=4)

    # White mask outside Mumbai boundary
    ax.add_geometries(
        [outer_mask], crs=ccrs.PlateCarree(),
        facecolor="white", edgecolor="none", zorder=3,
    )

    # Internal boundaries (districts / wards if present in shapefile)
    gdf.boundary.plot(
        ax=ax, edgecolor="#CC00CC", linewidth=0.6, zorder=6, aspect=None
    )
    # Second shapefile overlay
    gdf2 = gpd.read_file(SHAPEFILE_PATH2)
    if gdf2.crs is None:
        gdf2 = gdf2.set_crs(epsg=4326, allow_override=True)
    elif gdf2.crs.to_epsg() != 4326:
        gdf2 = gdf2.to_crs(epsg=4326)
    # Clip gdf2 to MMRDA boundary so outside areas don't show
    mmrda_union = unary_union(gdf.geometry.values)
    gdf2 = gdf2.clip(mmrda_union)
    gdf2.boundary.plot(
    ax=ax, edgecolor="#0000CC", linewidth=0.8, zorder=8, aspect=None
    )

    # Outer Mumbai boundary (thicker)
    outer_union = unary_union(gdf.geometry.values)
    outer_gdf   = gpd.GeoDataFrame({"geometry": [outer_union]}, crs=gdf.crs)
    outer_gdf.boundary.plot(
        ax=ax, edgecolor="#880000", linewidth=2.0, zorder=7, aspect=None
    )

    # Feature labels (use first text column found in the shapefile)
    '''label_col = _find_label_col(gdf)
    if label_col:
        for _, row in gdf.iterrows():
            pt   = row.geometry.representative_point()
            name = str(row[label_col])
            ax.text(
                pt.x, pt.y, name,
                transform=ccrs.PlateCarree(),
                fontsize=6, color="#660066", fontweight="bold",
                ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            )'''

    # Region label
    centre_x = (LON_MIN + LON_MAX) / 2
    centre_y = (LAT_MIN + LAT_MAX) / 2
    ax.text(
        centre_x, centre_y,
        "MUMBAI",
        transform=ccrs.PlateCarree(),
        fontsize=11, color="#CC0000", fontweight="bold",
        ha="center", va="center", zorder=10,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.7, color="gray", alpha=0.4, linestyle="--",
    )
    
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {"fontsize": 8}
    gl.ylabel_style = {"fontsize": 8}


def _find_label_col(gdf: gpd.GeoDataFrame):
    """Guess which column holds place names."""
    candidates = ["TALUK", "DISTRICT", "name", "NAME", "Name", "WARD_NAME",
                  "DIST_NAME", "dtname", "DTNAME", "label", "LABEL", "NAME_1", "NAME_2"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None   # ← no fallback, returns None if nothing matches


def add_colorbar(fig, ax, mappable):
    cbar = fig.colorbar(
        mappable, ax=ax, orientation="vertical",
        pad=0.02, shrink=0.55, fraction=0.03,
        extend="max", ticks=LEVELS,
    )
    cbar.ax.set_yticklabels(
        [f"{l:g}" for l in LEVELS], fontsize=8
    )
    cbar.set_label("Rainfall (mm)", fontsize=9)
    return cbar


def set_title(ax, param_label, forecast_hours, based_on_str, valid_for_str):
    ax.set_title(
        f"{MODEL_NAME} ({NWP_RES}) Mumbai {param_label} ({forecast_hours}-HR FCST)\n"
        f"Based on: {based_on_str}    Valid for: {valid_for_str}",
        fontsize=10, fontweight="bold", loc="center",
        pad=8, linespacing=1.6, family="monospace",
    )
def add_weatherex_logo(fig):
    logo = mpimg.imread(
        "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"
    )

    logo_ax = fig.add_axes(
        [0.01, 0.80, 0.15, 0.15]   # left, bottom, width, height
    )

    logo_ax.imshow(logo)
    logo_ax.axis("off")

def add_footer(fig):
    fig.text(
        0.5, 0.005,
        "(Background does not depict political boundary)",
        ha="center", fontsize=7.5, style="italic", color="#555555",
    )
    fig.text(
        0.01, 0.005,
        "NWP MODEL OUTPUT | WeatherEx.Ai",
        ha="left", fontsize=7, color="#555555",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PLOT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def make_plot(lons, lats, rain_data, cmap, norm,
              gdf, outer_mask,
              param_label, forecast_hours, based_on_str, valid_for_str,
              figsize=(9, 9)):
    """
    Returns (fig, ax, cf) — caller is responsible for saving and closing.
    """
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    apply_map_features(ax, gdf, outer_mask)
    add_weatherex_logo(fig)

    cf = ax.pcolormesh(
        to_np(lons), to_np(lats), rain_data,
        cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(), zorder=2, alpha=0.90,
    )
    
    #factor = 4

    #rain_fine = zoom(to_np(rain_data), factor, order=3)
    #lons_fine = zoom(to_np(lons), factor, order=3)
    #lats_fine = zoom(to_np(lats), factor, order=3)
    #cf = ax.contourf(
    #     lons_fine,
    #     lats_fine,
    #     rain_fine,
    #     levels=LEVELS,
    #     cmap=cmap,
    #     transform=ccrs.PlateCarree(),
    #     extend="max"
    #)
    add_colorbar(fig, ax, cf)
    set_title(ax, param_label, forecast_hours, based_on_str, valid_for_str)
    add_footer(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return fig, ax, cf


# ══════════════════════════════════════════════════════════════════════════════
#  TIME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_all_timestamps(ncfile, ntimes):
    return [
        pd.to_datetime(str(getvar(ncfile, "Times", timeidx=i).values))
        for i in range(ntimes)
    ]


def get_day_intervals(all_ts, cycle_hour=None):
    """
    Split into 24h buckets starting from the cycle hour.
    e.g. 12z run: 12 UTC → next day 12 UTC → day after 12 UTC
    """
    if cycle_hour is None:
        cycle_hour = all_ts[0].hour

    # Find indices where hour matches cycle start hour
    bucket_idx = [
        i for i, ts in enumerate(all_ts)
        if ts.hour == cycle_hour
    ]

    intervals = []

    # If first timestep is not on the bucket hour, start from 0
    if not bucket_idx or bucket_idx[0] != 0:
        bucket_idx = [0] + bucket_idx

    # Add final index as closing bound
    bounds = bucket_idx + [len(all_ts) - 1]

    for k in range(len(bounds) - 1):
        s = bounds[k]
        e = bounds[k + 1]
        if s == e:
            continue
        intervals.append((k + 1, s, e, all_ts[s], all_ts[e]))

    return intervals


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(wrf_path: str, output_dir: str, no_animation: bool = False):

    os.makedirs(output_dir, exist_ok=True)
    daily_dir = os.path.join(output_dir, "mumbai_daily_rainfall")
    os.makedirs(daily_dir, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    log.info("Loading WRF file ...")
    ncfile = Dataset(wrf_path)

    log.info(f"Loading shapefile: {SHAPEFILE_PATH}")
    gdf         = load_shapefile()
    outer_mask  = make_outer_mask(gdf)
    cmap, norm  = build_colormap()

    rain_t0      = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons   = latlon_coords(rain_t0)
    ntimes       = ncfile.dimensions["Time"].size
    all_ts       = load_all_timestamps(ncfile, ntimes)
    start_ts, end_ts = all_ts[0], all_ts[-1]
    based_on_str = fmt_time(start_ts)
    total_hours  = int((end_ts - start_ts).total_seconds() // 3600)

    log.info(f"WRF period: {start_ts} → {end_ts}  ({total_hours}h, {ntimes} timesteps)")

    # ── 1. Cumulative rainfall ─────────────────────────────────────────────────
    log.info("[1/3] Generating cumulative rainfall map ...")
    rain_end   = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = np.ma.masked_less(to_np(rain_end) - to_np(rain_t0), 1.0)

    fig, ax, cf = make_plot(
        lons, lats, rain_total, cmap, norm, gdf, outer_mask,
        param_label  = "CUMULATIVE RAINFALL (mm)",
        forecast_hours=total_hours,
        based_on_str = based_on_str,
        valid_for_str= fmt_time(end_ts),
    )
    add_weatherex_logo(fig)
    out_path = os.path.join(output_dir, "mumbai_cumulative_rainfall.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out_path}")

    # ── 2. Daily rainfall ──────────────────────────────────────────────────────
    log.info("[2/3] Generating daily rainfall maps ...")
    day_intervals = get_day_intervals(all_ts, cycle_hour=start_ts.hour)

    for day_num, s_idx, e_idx, s_ts, e_ts in day_intervals:
        r_start = getvar(ncfile, "RAINC", timeidx=s_idx) + getvar(ncfile, "RAINNC", timeidx=s_idx)
        r_end   = getvar(ncfile, "RAINC", timeidx=e_idx) + getvar(ncfile, "RAINNC", timeidx=e_idx)
        r_day   = np.ma.masked_less(to_np(r_end) - to_np(r_start), 1.0)
        day_hrs = int((e_ts - s_ts).total_seconds() // 3600)

        fig, ax, cf = make_plot(
            lons, lats, r_day, cmap, norm, gdf, outer_mask,
            param_label   = f"DAY {day_num} RAINFALL (mm)",
            forecast_hours= day_hrs,
            based_on_str  = based_on_str,
            valid_for_str = f"{fmt_time(s_ts)} to {fmt_time(e_ts)}",
        )
        add_weatherex_logo(fig)
        fname    = f"mumbai_daily_Day{day_num}_{e_ts.strftime('%d%b%Y').upper()}.png"
        out_path = os.path.join(daily_dir, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Day {day_num} → {out_path}")

    # ── 3. Hourly animation ────────────────────────────────────────────────────
    if no_animation:
        log.info("[3/3] Skipping animation (--no-animation flag set)")
    else:
        log.info("[3/3] Generating hourly animation ...")
        try:
            writer = FFMpegWriter(fps=4, metadata={"title": "Mumbai WRF Rainfall"})
        except Exception:
            log.warning("ffmpeg not found — skipping animation. Install ffmpeg to enable.")
            return

        fig, ax = plt.subplots(
            figsize=(9, 9), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        apply_map_features(ax, gdf, outer_mask)
        add_weatherex_logo(fig)

        cf_anim  = None
        anim_out = os.path.join(output_dir, "mumbai_hourly_animation.mp4")

        with writer.saving(fig, anim_out, dpi=120):
            for i in range(1, ntimes):
                r_s  = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
                r_e  = getvar(ncfile, "RAINC", timeidx=i) + getvar(ncfile, "RAINNC", timeidx=i)
                r_hr = np.ma.masked_less(to_np(r_e) - to_np(r_s), 1.0)

                if cf_anim is not None:
                    cf_anim.remove()

                cf_anim = ax.pcolormesh(
                    to_np(lons), to_np(lats), r_hr,
                    cmap=cmap, norm=norm, shading="auto",
                   transform=ccrs.PlateCarree(), zorder=2, alpha=0.90,
                )
                #factor = 4

                #rain_fine = zoom(to_np(r_hr), factor, order=3)
                #lons_fine = zoom(to_np(lons), factor, order=3)
                #lats_fine = zoom(to_np(lats), factor, order=3)
                #cf_anim = ax.contourf(
                #     lons_fine,
                 #    lats_fine,
                  #   rain_fine,
                   #  levels=LEVELS,
                    # cmap=cmap,
                     #transform=ccrs.PlateCarree(),
                     #extend="max"
               #)

                hr_elapsed = int((all_ts[i] - start_ts).total_seconds() // 3600)
                set_title(ax,
                    param_label   = f"CUMULATIVE RAINFALL (mm)  [+{hr_elapsed:03d}h]",
                    forecast_hours= hr_elapsed,
                    based_on_str  = based_on_str,
                    valid_for_str = fmt_time(all_ts[i]),
                )
                add_footer(fig)
                writer.grab_frame()

                if i == 1:
                    add_colorbar(fig, ax, cf_anim)

                if (i % 10) == 0:
                    log.info(f"  Frame {i}/{ntimes-1}")

        plt.close(fig)
        log.info(f"  Animation → {anim_out}")

    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Mumbai WRF rainfall plots using /home/ras_08/WEATHER/INDIA/WORKFLOW/mumbai/Mumbai_Shapefile.shp"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to a single wrfout file, e.g. wrfout_d01_2024-06-01_00:00:00"
    )
    p.add_argument(
        "--outdir", default="./mumbai_output",
        help="Directory for output images and animation (default: ./mumbai_output)"
    )
    p.add_argument(
        "--shapefile", default=None,
        help="Override path to /home/ras_08/WEATHER/INDIA/WORKFLOW/mumbai/Mumbai_Shapefile.shp (optional)"
    )
    p.add_argument(
        "--no-animation", action="store_true",
        help="Skip the hourly animation (faster, no ffmpeg needed)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Allow command-line shapefile override
    if args.shapefile:
        SHAPEFILE_PATH = args.shapefile

    run(
        wrf_path     = args.input,
        output_dir   = args.outdir,
        no_animation = args.no_animation,
    )
