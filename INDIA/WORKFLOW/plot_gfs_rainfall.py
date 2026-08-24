#!/usr/bin/env python3
"""
plot_gfs_rainfall.py
====================
Plot GFS forecast precipitation directly from GRIB2 files.

Outputs:
  • gfs_cumulative_rainfall.png       — total accumulated over all files
  • gfs_stepwise_rainfall/            — per-step 3-hourly PNGs
  • gfs_rainfall_animation.mp4        — animated 3-hourly rainfall

Usage:
  python plot_gfs_rainfall.py --gfsdir /path/to/GFS-DATA/20260619/00z --outdir /path/to/output
  python plot_gfs_rainfall.py --gfsdir /path/to/dir --outdir /path/to/output --cycle 00 --date 20260619
"""

__version__ = "1.0"
__author__  = "ras_08 / WeatherEx.Ai"

import os
import glob
import argparse
import logging
import warnings
import numpy as np
import xarray as xr
import cfgrib
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.image as mpimg
from datetime import datetime

warnings.filterwarnings("ignore")

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
#  CONFIGURATION
# =============================================================================

SHAPEFILE_PATH = "/home/ras_08/WEATHER/INDIA/WORKFLOW/shapefiles/district.shp"
LOGO_PATH      = "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"
NWP_RES        = "0.25 deg"

# Rainfall color scale (IMD style)
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

# GFS 3-hourly step in seconds
STEP_SECONDS = 3 * 3600


# =============================================================================
#  DATA LOADER
# =============================================================================

def load_prate_from_file(filepath: str) -> xr.DataArray:
    """
    Load precipitation rate (prate) from a single GFS GRIB2 file.
    prate is in Dataset 34 — use filter_by_keys to target it directly.
    Returns a DataArray on lat/lon grid.
    """
    datasets = cfgrib.open_datasets(filepath)

    for ds in datasets:
        if "prate" in ds.data_vars:
            prate = ds["prate"]
            log.debug(f"  Found prate in {os.path.basename(filepath)}: {prate.shape}")
            return prate

    raise KeyError(f"'prate' not found in any dataset in {filepath}")


def load_all_files(gfs_dir: str) -> list[dict]:
    """
    Load all GFS GRIB2 files in the directory sorted by forecast hour.
    Returns list of dicts: { fhr, filepath, prate, valid_time }
    """
    pattern = os.path.join(gfs_dir, "gfs.t??z.pgrb2.0p25.f*.grib2")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No GFS GRIB2 files found in {gfs_dir}\n"
            f"Expected pattern: gfs.tHHz.pgrb2.0p25.fNNN.grib2"
        )

    log.info(f"Found {len(files)} GRIB2 files in {gfs_dir}")

    records = []
    for f in files:
        # Extract forecast hour from filename (e.g. f012 → 12)
        fname = os.path.basename(f)
        fhr   = int(fname.split(".f")[-1].replace(".grib2", ""))

        try:
            prate = load_prate_from_file(f)
            records.append({
                "fhr":        fhr,
                "filepath":   f,
                "prate":      prate,
                "valid_time": str(prate.valid_time.values)[:16] if hasattr(prate, "valid_time") else f"f{fhr:03d}",
            })
            log.info(f"  f{fhr:03d}: loaded  valid={records[-1]['valid_time']}")
        except Exception as e:
            log.warning(f"  f{fhr:03d}: FAILED — {e}")

    if not records:
        raise RuntimeError("No prate data could be loaded from any file.")

    return records


# =============================================================================
#  PRECIPITATION ACCUMULATION
# =============================================================================

def compute_accumulated_rainfall(records: list[dict]) -> tuple[np.ndarray, list]:
    """
    Convert prate (kg/m2/s) to 3-hourly accumulated rainfall (mm).
    For each step: rain_mm = prate * STEP_SECONDS
    f000 is the analysis — its prate is 0, so we start accumulating from f003.

    Returns:
      cumulative   — 2D array of total accumulated rainfall
      step_rains   — list of 2D arrays, one per step (3-hourly)
    """
    step_rains = []
    cumulative = None

    for i, rec in enumerate(records):
        prate_np = rec["prate"].values.astype(np.float32)

        # Convert rate to 3h accumulation (mm = kg/m2)
        rain_3h = prate_np * STEP_SECONDS

        # Mask negative/tiny values
        rain_3h = np.where(rain_3h < 0.01, np.nan, rain_3h)

        step_rains.append(rain_3h)

        if cumulative is None:
            cumulative = np.zeros_like(rain_3h)
        cumulative = np.nansum([cumulative, rain_3h], axis=0)

    return cumulative, step_rains


# =============================================================================
#  MAP SETUP
# =============================================================================

def load_geodata():
    """Load India shapefile for boundaries."""
    if not os.path.exists(SHAPEFILE_PATH):
        log.warning(f"Shapefile not found: {SHAPEFILE_PATH} — boundaries skipped")
        return None, None
    counties = gpd.read_file(SHAPEFILE_PATH)
    states   = counties.dissolve(by="ST_NM")
    return counties, states


def build_colormap():
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    return cmap, norm


def apply_map_features(ax, lons, lats, counties, states):
    """Apply base map: extent, coastline, state/district boundaries, gridlines."""
    lon_min = float(lons.min()) 
    lon_max = float(lons.max())
    lat_min = float(lats.min())
    lat_max = float(lats.max())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="white",   zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce6ff", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5,        zorder=4)

    if counties is not None:
        counties.boundary.plot(
            ax=ax, edgecolor="#FF00FF", linewidth=0.2, zorder=4, aspect=None
        )
    if states is not None:
        states.boundary.plot(
            ax=ax, edgecolor="#FF00FF", linewidth=1.2, zorder=5, aspect=None
        )
        for state_name, row in states.iterrows():
            pt = row.geometry.representative_point()
            ax.text(
                pt.x, pt.y, state_name,
                transform=ccrs.PlateCarree(),
                fontsize=5.0, color="#880088", fontweight="bold",
                ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )

    gl = ax.gridlines(
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
        logo_ax = fig.add_axes([0.02, 0.78, 0.18, 0.18])
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
    cbar.set_label("Rainfall (mm)", fontsize=9)
    return cbar


def set_title(ax, param_label, forecast_hours, based_on_str, valid_for_str):
    line1 = f"GFS ({NWP_RES}) {param_label} FORECAST ({forecast_hours} HR)"
    line2 = f"based on {based_on_str}  valid for {valid_for_str}"
    ax.set_title(
        f"{line1}\n{line2}",
        fontsize=11, fontweight="bold",
        loc="center", pad=8, linespacing=1.6, family="monospace"
    )


def add_footer(fig):
    fig.text(
        0.5, 0.005,
        "(Background does not depict political boundary)  |  "
        "GFS Data: NOAA NCEP NOMADS",
        ha="center", fontsize=7.5, style="italic", color="#444444"
    )


def remove_mesh(cf):
    if cf is not None:
        try:
            cf.remove()
        except Exception:
            pass


# =============================================================================
#  SINGLE PLOT BUILDER
# =============================================================================

def make_plot(fig, ax, lons, lats, data, cmap, norm,
              param_label, forecast_hours,
              based_on_str, valid_for_str,
              counties, states):
    """Render one GFS rainfall map."""
    apply_map_features(ax, lons, lats, counties, states)

    cf = ax.pcolormesh(
        lons, lats, data,
        cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(), zorder=2, alpha=0.9
    )
    add_colorbar(fig, ax, cf)
    set_title(ax,
        param_label=param_label,
        forecast_hours=forecast_hours,
        based_on_str=based_on_str,
        valid_for_str=valid_for_str
    )
    add_footer(fig)
    return cf


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_pipeline(gfs_dir: str, output_dir: str, cycle: str, date: str):
    """
    Full GFS rainfall pipeline:
      1. Load all GRIB2 files and extract prate
      2. Cumulative rainfall map
      3. Per-step 3-hourly PNGs
      4. Animated MP4
    """
    if not os.path.isdir(gfs_dir):
        raise FileNotFoundError(f"GFS directory not found: {gfs_dir}")

    os.makedirs(output_dir, exist_ok=True)
    step_dir = os.path.join(output_dir, "gfs_stepwise_rainfall")
    os.makedirs(step_dir, exist_ok=True)

    log.info("=" * 60)
    log.info(f"GFS Rainfall Pipeline  —  {date} {cycle}z")
    log.info(f"  GFS dir   : {gfs_dir}")
    log.info(f"  Output dir: {output_dir}")
    log.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("[1/4] Loading GRIB2 files ...")
    records = load_all_files(gfs_dir)

    # Get lat/lon grid from first record
    prate0 = records[0]["prate"]
    lats   = prate0.latitude.values
    lons   = prate0.longitude.values

    # Base time string
    init_str = f"{cycle} UTC of {date[6:8]}-{date[4:6]}-{date[0:4]}"

    # Compute accumulated rainfall
    log.info("[1/4] Computing accumulated rainfall ...")
    cumulative, step_rains = compute_accumulated_rainfall(records)

    total_hours = records[-1]["fhr"] - records[0]["fhr"]

    # Load geodata
    counties, states = load_geodata()
    cmap, norm       = build_colormap()

    # ── 2. Cumulative rainfall map ────────────────────────────────────────────
    log.info("[2/4] Generating cumulative rainfall map ...")

    cum_masked = np.where(cumulative < 1.0, np.nan, cumulative)

    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    make_plot(
        fig, ax, lons, lats, cum_masked, cmap, norm,
        param_label="TOTAL RAINFALL (mm)",
        forecast_hours=total_hours,
        based_on_str=init_str,
        valid_for_str=records[-1]["valid_time"],
        counties=counties, states=states
    )
    add_logo(fig)

    out_cum = os.path.join(output_dir, "gfs_cumulative_rainfall.png")
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out_cum}")

    # ── 3. Per-step 3-hourly PNGs ─────────────────────────────────────────────
    log.info(f"[3/4] Generating {len(records)} stepwise 3-hourly plots ...")

    for i, rec in enumerate(records):
        rain = np.where(step_rains[i] < 0.1, np.nan, step_rains[i])

        fig, ax = plt.subplots(
            figsize=(12, 10),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        make_plot(
            fig, ax, lons, lats, rain, cmap, norm,
            param_label="3-HOURLY RAINFALL (mm)",
            forecast_hours=3,
            based_on_str=init_str,
            valid_for_str=rec["valid_time"],
            counties=counties, states=states
        )
        add_logo(fig)

        fname    = f"gfs_rainfall_f{rec['fhr']:03d}.png"
        out_path = os.path.join(step_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        log.info(f"  f{rec['fhr']:03d}: {fname}")

    # ── 4. Animation ──────────────────────────────────────────────────────────
    log.info("[4/4] Generating animation ...")

    fig2, ax2 = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    apply_map_features(ax2, lons, lats, counties, states)
    add_logo(fig2)
    add_footer(fig2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_colorbar(fig2, ax2, sm)

    _cf = [None]

    def update(frame):
        remove_mesh(_cf[0])
        rain = np.where(step_rains[frame] < 0.1, np.nan, step_rains[frame])
        cf_new = ax2.pcolormesh(
            lons, lats, rain,
            cmap=cmap, norm=norm, shading="auto",
            transform=ccrs.PlateCarree(), zorder=2, alpha=0.9
        )
        _cf[0] = cf_new
        set_title(
            ax2,
            param_label="3-HOURLY RAINFALL (mm)",
            forecast_hours=3,
            based_on_str=init_str,
            valid_for_str=records[frame]["valid_time"]
        )
        return []

    ani = animation.FuncAnimation(
        fig2, update,
        frames=len(records),
        interval=1500, blit=False
    )
    out_ani = os.path.join(output_dir, "gfs_rainfall_animation.mp4")
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    log.info(f"  Saved: {out_ani}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"DONE — Outputs saved to: {output_dir}")
    log.info(f"  • gfs_cumulative_rainfall.png")
    log.info(f"  • gfs_rainfall_animation.mp4")
    log.info(f"  • gfs_stepwise_rainfall/  ({len(records)} PNG files)")
    log.info("=" * 60)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=f"GFS GRIB2 Rainfall Plotter v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    today = datetime.now().strftime("%Y%m%d")

    parser.add_argument(
        "--gfsdir",
        default=f"/home/ras_08/WEATHER/INDIA/GFS-DATA/{today}/00z",
        help="Directory containing GFS GRIB2 files (default: today 00z)"
    )
    parser.add_argument(
        "--outdir",
        default=f"/home/ras_08/WEATHER/INDIA/Data/{today}/gfs",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--cycle",
        default="00",
        choices=["00", "06", "12", "18"],
        help="GFS cycle (default: 00)"
    )
    parser.add_argument(
        "--date",
        default=today,
        help="Date in YYYYMMDD format (default: today)"
    )

    args = parser.parse_args()

    log.info(f"Script : plot_gfs_rainfall.py v{__version__}")
    log.info(f"GFS dir: {args.gfsdir}")
    log.info(f"Output : {args.outdir}")
    log.info(f"Cycle  : {args.date} {args.cycle}z")

    run_pipeline(
        gfs_dir=args.gfsdir,
        output_dir=args.outdir,
        cycle=args.cycle,
        date=args.date
    )
