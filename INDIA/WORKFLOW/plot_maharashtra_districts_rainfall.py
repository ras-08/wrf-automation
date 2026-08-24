#!/usr/bin/env python3
"""
plot_maharashtra_districts_rainfall.py
=======================================
NWP WRF Rainfall plots for Maharashtra — state-wide or district level.

Regions available via --region flag:
  maharashtra     — full state
  mumbai          — Mumbai district only
  mumbai_suburban — Mumbai Suburban district only
  greater_mumbai  — Mumbai + Mumbai Suburban combined

Outputs (per region):
  • REGION_cumulative_rainfall.png
  • daily_rainfall_REGION/REGION_daily_DayN_DDMONYYYY.png
  • REGION_hourly_animation.mp4

Usage:
  python plot_maharashtra_districts_rainfall.py --input wrfout_d01_* --outdir /path --region maharashtra
  python plot_maharashtra_districts_rainfall.py --input wrfout_d01_* --outdir /path --region mumbai
  python plot_maharashtra_districts_rainfall.py --input wrfout_d01_* --outdir /path --region greater_mumbai
  python plot_maharashtra_districts_rainfall.py --input wrfout_d01_* --outdir /path --region all
"""

__version__ = "1.0"
__author__  = "ras_08 / NCMRWF"

import os
import argparse
import logging
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
from datetime import datetime
import matplotlib.image as mpimg


# =============================================================================
#  LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


# =============================================================================
#  SHAPEFILE & LOGO PATHS
# =============================================================================
SHAPEFILE_PATH = (
    "/home/ras_08/WEATHER/INDIA/WORKFLOW/MAHARASHTRA/MAHARASHTRA_DISTRICTS.geojson"
)
LOGO_PATH = "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"
NWP_RES   = "27 Km"

# Column names in GeoJSON
STATE_COL = "stname"
DIST_COL  = "dtname"
STATE_IN_FILE = "MAHARASHTRA"


# =============================================================================
#  REGION DEFINITIONS
#  Add any new district here — no other code changes needed
# =============================================================================
REGIONS = {
    "maharashtra": {
        "label":           "Maharashtra",
        "code":            "MH",
        "district_filter": False,          # False = use full state
        "districts":       [],             # ignored when filter=False
        "lon_min": 72.5,  "lon_max": 80.9,
        "lat_min": 15.6,  "lat_max": 22.1,
        "figsize":         (12, 8),
        "label_offset":    0.5,            # y offset for state label
    },
    "mumbai": {
        "label":           "Mumbai",
        "code":            "MUM",
        "district_filter": True,
        "districts":       ["Mumbai"],
        "lon_min": 72.75, "lon_max": 72.95,
        "lat_min": 18.85, "lat_max": 19.10,
        "figsize":         (8, 8),
        "label_offset":    0.02,
    },
    "mumbai_suburban": {
        "label":           "Mumbai Suburban",
        "code":            "MUM_S",
        "district_filter": True,
        "districts":       ["Mumbai Suburban"],
        "lon_min": 72.75, "lon_max": 73.00,
        "lat_min": 19.05, "lat_max": 19.35,
        "figsize":         (8, 8),
        "label_offset":    0.02,
    },
    "greater_mumbai": {
        "label":           "Greater Mumbai",
        "code":            "GMUM",
        "district_filter": True,
        "districts":       ["Mumbai", "Mumbai Suburban"],
        "lon_min": 72.75, "lon_max": 73.05,
        "lat_min": 18.85, "lat_max": 19.35,
        "figsize":         (8, 10),
        "label_offset":    0.03,
    },
}


# =============================================================================
#  RAINFALL COLOR SCALE (IMD style)
# =============================================================================
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


# =============================================================================
#  GEODATA LOADER
# =============================================================================

def load_geodata(region: dict):
    """
    Load GeoJSON and return (districts_gdf, state_gdf) for the given region.
    If district_filter=True, filters to the specified districts only.
    If district_filter=False, returns full Maharashtra.
    """
    log.info(f"Loading shapefile: {SHAPEFILE_PATH}")
    if not os.path.exists(SHAPEFILE_PATH):
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")

    counties = gpd.read_file(SHAPEFILE_PATH)

    if counties.crs is None or counties.crs.to_epsg() != 4326:
        log.warning("CRS not EPSG:4326 — overriding")
        counties = counties.set_crs(epsg=4326, allow_override=True)

    if region["district_filter"]:
        # Filter to specific districts
        districts = counties[
            counties[DIST_COL].isin(region["districts"])
        ].copy()

        if districts.empty:
            available = counties[DIST_COL].unique().tolist()
            raise ValueError(
                f"Districts {region['districts']} not found.\n"
                f"Available: {available}"
            )

        # State boundary = union of selected districts
        state_geom = unary_union(districts.geometry.values)
        state_gdf  = gpd.GeoDataFrame(
            {"geometry": [state_geom]},
            crs=districts.crs
        )
        log.info(
            f"  Region    : {region['label']}  "
            f"({len(districts)} district(s))"
        )

    else:
        # Full Maharashtra
        all_mh   = counties[counties[STATE_COL] == STATE_IN_FILE].copy()
        districts = all_mh
        state_gdf = gpd.GeoDataFrame(
            {"geometry": [unary_union(all_mh.geometry.values)]},
            crs=all_mh.crs
        )
        if districts.empty:
            raise ValueError(
                f"State '{STATE_IN_FILE}' not found. "
                f"Available: {counties[STATE_COL].unique().tolist()}"
            )
        log.info(
            f"  Region    : {region['label']}  "
            f"({len(districts)} districts)"
        )

    return districts, state_gdf


# =============================================================================
#  COLORMAP
# =============================================================================

def build_colormap():
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    return cmap, norm


# =============================================================================
#  TIME FORMATTER
# =============================================================================

def fmt_time(ts: pd.Timestamp) -> str:
    return ts.strftime('%H UTC of %d-%m-%Y')


# =============================================================================
#  MAP FEATURES
# =============================================================================

def add_state_mask(ax, state_gdf):
    """White mask outside the region boundary."""
    outer   = box(50.0, -10.0, 110.0, 45.0)
    union   = unary_union(state_gdf.geometry.values)
    outside = outer.difference(union)
    ax.add_geometries(
        [outside], crs=ccrs.PlateCarree(),
        facecolor="white", edgecolor="none", zorder=3
    )


def apply_map_features(axis, districts_gdf, state_gdf, region: dict):
    """Apply all map decorations for the given region."""

    axis.set_extent(
        [region["lon_min"], region["lon_max"],
         region["lat_min"], region["lat_max"]],
        crs=ccrs.PlateCarree()
    )

    axis.add_feature(cfeature.LAND,      facecolor="white",   zorder=0)
    axis.add_feature(cfeature.OCEAN,     facecolor="#cce6ff", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=1.5,        zorder=5)

    # White mask outside region
    add_state_mask(axis, state_gdf)

    # District boundaries
    districts_gdf.boundary.plot(
        ax=axis, edgecolor="#FF00FF", linewidth=0.5, zorder=6, aspect=None
    )

    # Region outer boundary
    state_gdf.boundary.plot(
        ax=axis, edgecolor="#FF00FF", linewidth=2.0, zorder=7, aspect=None
    )

    # District name labels
    for _, row in districts_gdf.iterrows():
        point     = row.geometry.representative_point()
        dist_name = row.get(DIST_COL, "")
        if dist_name:
            axis.text(
                point.x, point.y, dist_name,
                transform=ccrs.PlateCarree(),
                fontsize=5.5, color="#880088", fontweight="bold",
                ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )

    # Region label
    centre = state_gdf.geometry.iloc[0].representative_point()
    axis.text(
        centre.x, centre.y + region["label_offset"],
        region["label"].upper(),
        transform=ccrs.PlateCarree(),
        fontsize=10, color="#CC0000", fontweight="bold",
        ha="center", va="center", zorder=10,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
    )

    # Gridlines
    gl = axis.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.8, color="gray", alpha=0.4, linestyle="--"
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {"fontsize": 9, "color": "black"}
    gl.ylabel_style = {"fontsize": 9, "color": "black"}


# =============================================================================
#  PLOT DECORATIONS
# =============================================================================

def add_logo(fig):
    if os.path.exists(LOGO_PATH):
        logo    = mpimg.imread(LOGO_PATH)
        logo_ax = fig.add_axes([0.01, 0.80, 0.15, 0.15])
        logo_ax.imshow(logo)
        logo_ax.axis("off")
    else:
        log.warning(f"Logo not found: {LOGO_PATH}")


def add_colorbar(fig, ax, mappable):
    cbar = fig.colorbar(
        mappable, ax=ax, orientation="vertical",
        pad=0.02, shrink=0.50, fraction=0.03,
        extend="max", ticks=LEVELS
    )
    cbar.ax.set_yticklabels(
        [str(int(l)) if l == int(l) else str(l) for l in LEVELS],
        fontsize=9
    )
    cbar.ax.tick_params(length=4)
    return cbar


def set_title(ax, region, param_label, forecast_hours, based_on_str, valid_for_str):
    line1 = f"NWP ({NWP_RES}) {region['label']} {param_label} FORECAST ({forecast_hours} HR)"
    line2 = f"based on {based_on_str} valid for {valid_for_str}"
    ax.set_title(
        f"{line1}\n{line2}",
        fontsize=11, fontweight="bold",
        loc="center", pad=8, linespacing=1.6, family="monospace"
    )


def add_footer(fig):
    fig.text(
        0.5, 0.005,
        "(Background does not depict political boundary)",
        ha="center", fontsize=8, style="italic", color="#444444"
    )


def add_imd_credit(fig):
    fig.text(
        0.01, 0.005,
        "IMD OPERATIONAL GLOBAL MODEL COURTESY : BTM, NCMRWF",
        ha="left", fontsize=6.5, color="#555555"
    )


def remove_pcolormesh(cf):
    if cf is not None:
        try:
            cf.remove()
        except Exception as e:
            log.debug(f"remove_pcolormesh: {e}")


# =============================================================================
#  PLOT BUILDER
# =============================================================================

def make_plot(fig, ax, lons, lats, data, cmap, norm,
              region, param_label, forecast_hours,
              based_on_str, valid_for_str,
              districts_gdf, state_gdf):
    """Render one rainfall map for the given region."""
    apply_map_features(ax, districts_gdf, state_gdf, region)
    cf = ax.pcolormesh(
        to_np(lons), to_np(lats), data,
        cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(), zorder=2, alpha=0.9
    )
    add_colorbar(fig, ax, cf)
    set_title(ax, region,
        param_label=param_label,
        forecast_hours=forecast_hours,
        based_on_str=based_on_str,
        valid_for_str=valid_for_str
    )
    add_footer(fig)
    add_imd_credit(fig)
    return cf


# =============================================================================
#  TIME INTERVAL HELPER
# =============================================================================

def get_day_intervals(ncfile, ntimes):
    all_ts = [
        pd.to_datetime(str(getvar(ncfile, "Times", timeidx=i).values))
        for i in range(ntimes)
    ]
    midnight_indices = [i for i, ts in enumerate(all_ts) if ts.hour == 0]
    intervals = []

    if not midnight_indices:
        intervals.append({
            "start_idx": 0, "end_idx": ntimes - 1,
            "start_ts": all_ts[0], "end_ts": all_ts[-1],
        })
        return intervals

    if midnight_indices[0] != 0:
        intervals.append({
            "start_idx": 0, "end_idx": midnight_indices[0],
            "start_ts": all_ts[0], "end_ts": all_ts[midnight_indices[0]],
        })

    for k in range(len(midnight_indices) - 1):
        s, e = midnight_indices[k], midnight_indices[k + 1]
        intervals.append({
            "start_idx": s, "end_idx": e,
            "start_ts": all_ts[s], "end_ts": all_ts[e],
        })

    last_mid = midnight_indices[-1]
    if last_mid < ntimes - 1:
        intervals.append({
            "start_idx": last_mid, "end_idx": ntimes - 1,
            "start_ts": all_ts[last_mid], "end_ts": all_ts[-1],
        })

    return intervals


# =============================================================================
#  SINGLE REGION PIPELINE
# =============================================================================

def run_region(wrf_path: str, output_dir: str, region_key: str):
    """Run the full rainfall pipeline for one region."""

    region = REGIONS[region_key]
    code   = region["code"].lower()

    # Per-region output subdirectory
    region_out = os.path.join(output_dir, region_key)
    daily_dir  = os.path.join(region_out, f"daily_rainfall_{code}")
    os.makedirs(region_out, exist_ok=True)
    os.makedirs(daily_dir,  exist_ok=True)

    log.info("=" * 60)
    log.info(f"Region: {region['label']}  [{region_key}]")
    log.info(f"Output: {region_out}")
    log.info("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────────
    log.info("[1/4] Loading WRF file and geodata ...")
    ncfile = Dataset(wrf_path)
    districts_gdf, state_gdf = load_geodata(region)
    cmap, norm = build_colormap()

    rain_t0      = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons   = latlon_coords(rain_t0)
    ntimes       = ncfile.dimensions["Time"].size
    start_ts     = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=0).values))
    end_ts       = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=-1).values))
    based_on_str = fmt_time(start_ts)
    delta_hours  = int((end_ts - start_ts).total_seconds() // 3600)

    log.info(f"  WRF: {start_ts} → {end_ts}  ({delta_hours}h, {ntimes} steps)")

    # ── Cumulative ────────────────────────────────────────────────────────────
    log.info(f"[2/4] Cumulative rainfall map ...")
    rain_end   = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = np.ma.masked_less(to_np(rain_end) - to_np(rain_t0), 1.0)

    fig, ax = plt.subplots(
        figsize=region["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    make_plot(
        fig, ax, lons, lats, rain_total, cmap, norm,
        region=region,
        param_label="RAINFALL (mm)",
        forecast_hours=delta_hours,
        based_on_str=based_on_str,
        valid_for_str=fmt_time(end_ts),
        districts_gdf=districts_gdf,
        state_gdf=state_gdf
    )
    add_logo(fig)
    out_cum = os.path.join(region_out, f"{code}_cumulative_rainfall.png")
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out_cum}")

    # ── Per-day plots ─────────────────────────────────────────────────────────
    day_intervals = get_day_intervals(ncfile, ntimes)
    log.info(f"[3/4] Generating {len(day_intervals)} daily plot(s) ...")

    for day_num, interval in enumerate(day_intervals, start=1):
        s_idx   = interval["start_idx"]
        e_idx   = interval["end_idx"]
        s_ts    = interval["start_ts"]
        e_ts    = interval["end_ts"]
        day_hrs = int((e_ts - s_ts).total_seconds() // 3600)

        r_start = to_np(getvar(ncfile, "RAINC",  timeidx=s_idx) +
                        getvar(ncfile, "RAINNC", timeidx=s_idx))
        r_end   = to_np(getvar(ncfile, "RAINC",  timeidx=e_idx) +
                        getvar(ncfile, "RAINNC", timeidx=e_idx))
        daily   = np.ma.masked_less(r_end - r_start, 1.0)

        fig, ax = plt.subplots(
            figsize=region["figsize"],
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        make_plot(
            fig, ax, lons, lats, daily, cmap, norm,
            region=region,
            param_label="RAINFALL (mm)",
            forecast_hours=day_hrs,
            based_on_str=fmt_time(s_ts),
            valid_for_str=fmt_time(e_ts),
            districts_gdf=districts_gdf,
            state_gdf=state_gdf
        )
        add_logo(fig)
        fname    = f"{code}_daily_Day{day_num}_{s_ts.strftime('%d%b%Y')}.png"
        out_path = os.path.join(daily_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        log.info(f"  Day {day_num}: {s_ts.strftime('%d %b %Y %H UTC')} → "
                 f"{e_ts.strftime('%d %b %Y %H UTC')}  →  {fname}")

    # ── Animation ─────────────────────────────────────────────────────────────
    log.info(f"[4/4] Generating hourly animation ...")

    fig2, ax2 = plt.subplots(
        figsize=region["figsize"],
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    apply_map_features(ax2, districts_gdf, state_gdf, region)
    add_logo(fig2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_colorbar(fig2, ax2, sm)
    add_footer(fig2)
    add_imd_credit(fig2)

    _cf = [None]

    def update(frame):
        remove_pcolormesh(_cf[0])
        r_prev = to_np(getvar(ncfile, "RAINC",  timeidx=frame) +
                       getvar(ncfile, "RAINNC", timeidx=frame))
        r_now  = to_np(getvar(ncfile, "RAINC",  timeidx=frame+1) +
                       getvar(ncfile, "RAINNC", timeidx=frame+1))
        hourly = np.ma.masked_less_equal(r_now - r_prev, 0.1)
        cf_new = ax2.pcolormesh(
            to_np(lons), to_np(lats), hourly,
            cmap=cmap, norm=norm, shading="auto",
            transform=ccrs.PlateCarree(), zorder=2, alpha=0.85
        )
        _cf[0] = cf_new
        t_prev = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame).values))
        t_now  = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame+1).values))
        set_title(ax2, region,
            param_label="RAINFALL (mm)",
            forecast_hours=1,
            based_on_str=fmt_time(t_prev),
            valid_for_str=fmt_time(t_now)
        )
        return []

    ani = animation.FuncAnimation(
        fig2, update, frames=ntimes - 1, interval=1500, blit=False
    )
    out_ani = os.path.join(region_out, f"{code}_hourly_animation.mp4")
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    log.info(f"  Saved: {out_ani}")

    ncfile.close()

    log.info("=" * 60)
    log.info(f"DONE — {region['label']} → {region_out}")
    log.info(f"  • {code}_cumulative_rainfall.png")
    log.info(f"  • {code}_hourly_animation.mp4")
    log.info(f"  • daily_rainfall_{code}/  ({len(day_intervals)} files)")
    log.info("=" * 60)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    valid_regions = list(REGIONS.keys()) + ["all"]

    parser = argparse.ArgumentParser(
        description=f"NWP Maharashtra District Rainfall Plotter v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    today = datetime.now().strftime("%Y-%m-%d")

    parser.add_argument(
        "--input",
        default=(
            f"/home/ras_08/Models/WRF_TUTORIAL/WRFV4.5/run/"
            f"wrfout_d01_{today}_00:00:00"
        ),
        help="Path to WRF output file"
    )
    parser.add_argument(
        "--outdir",
        default=(
            f"/home/ras_08/WEATHER/INDIA/Data/"
            f"{datetime.now().strftime('%Y%m%d')}/maharashtra"
        ),
        help="Root output directory"
    )
    parser.add_argument(
        "--region",
        default="all",
        choices=valid_regions,
        help=(
            "Region to plot: maharashtra | mumbai | mumbai_suburban | "
            "greater_mumbai | all  (default: all)"
        )
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"WRF file not found: {args.input}")

    log.info(f"Script  : plot_maharashtra_districts_rainfall.py v{__version__}")
    log.info(f"Input   : {args.input}")
    log.info(f"Output  : {args.outdir}")
    log.info(f"Region  : {args.region}")

    # Determine which regions to run
    regions_to_run = list(REGIONS.keys()) if args.region == "all" else [args.region]

    for rkey in regions_to_run:
        log.info(f"\n>>> Running region: {rkey}")
        run_region(
            wrf_path=args.input,
            output_dir=args.outdir,
            region_key=rkey
        )

    log.info("\nAll regions complete.")
