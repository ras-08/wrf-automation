#!/usr/bin/env python3
"""
plot_gfs_kerala_rainfall.py
===========================
Plot GFS forecast precipitation for Kerala directly from GRIB2 files.

Outputs:
  • gfs_kerala_cumulative_rainfall.png
  • gfs_kerala_stepwise_rainfall/gfs_kerala_rainfall_fNNN.png
  • gfs_kerala_rainfall_animation.mp4

Usage:
  python plot_gfs_kerala_rainfall.py --gfsdir /path/to/GFS-DATA/20260619/00z --outdir /path/to/output
  python plot_gfs_kerala_rainfall.py --gfsdir /path/to/dir --outdir /path/to/output --cycle 00 --date 20260619
"""

__version__ = "1.0"
__author__  = "ras_08 / NCMRWF"

import os
import glob
import argparse
import logging
import warnings
import numpy as np
import cfgrib
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FFMpegWriter
from shapely.geometry import box
from shapely.ops import unary_union
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
#  CONFIGURATION  — change only this block for a different state
# =============================================================================

STATE_NAME   = "Kerala"       # must match ST_NM column exactly
STATE_LABEL  = "Kerala"       # displayed on map
STATE_CODE   = "KL"           # used in output filenames
DIST_COL     = "DISTRICT"     # district name column in shapefile
STATE_COL    = "ST_NM"        # state name column in shapefile
NWP_RES      = "0.25 deg"     # GFS resolution label

SHAPEFILE_PATH = "/home/ras_08/WEATHER/INDIA/WORKFLOW/kerala/shapefiles/district.shp"
LOGO_PATH      = "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"

# Kerala bounding box (with small buffer)
LON_MIN = 74.5
LON_MAX = 78.0
LAT_MIN =  7.9
LAT_MAX = 12.8

# IMD rainfall color scale
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
#  GEODATA LOADER
# =============================================================================

def load_geodata():
    """
    Load shapefile, dissolve to state boundary, extract Kerala districts.
    Raises ValueError if state not found.
    """
    log.info(f"Loading shapefile: {SHAPEFILE_PATH}")
    if not os.path.exists(SHAPEFILE_PATH):
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")

    counties = gpd.read_file(SHAPEFILE_PATH)

    if counties.crs is None or counties.crs.to_epsg() != 4326:
        log.warning("CRS not EPSG:4326 — overriding to WGS84")
        counties = counties.set_crs(epsg=4326, allow_override=True)

    states       = counties.dissolve(by=STATE_COL)
    kl_districts = counties[counties[STATE_COL] == STATE_NAME].copy()
    kl_state     = states[states.index == STATE_NAME].copy()

    if kl_districts.empty:
        available = counties[STATE_COL].unique().tolist()
        raise ValueError(
            f"State '{STATE_NAME}' not found.\n"
            f"Available: {available}\n"
            f"Check STATE_NAME in CONFIGURATION block."
        )

    log.info(f"  Districts found: {len(kl_districts)}")
    return kl_districts, kl_state


# =============================================================================
#  GFS DATA LOADER
# =============================================================================

def load_prate_from_file(filepath: str) -> xr.DataArray:
    """Load precipitation rate (prate) from a single GFS GRIB2 file."""
    datasets = cfgrib.open_datasets(filepath)
    for ds in datasets:
        if "prate" in ds.data_vars:
            return ds["prate"]
    raise KeyError(f"'prate' not found in {os.path.basename(filepath)}")


def load_all_files(gfs_dir: str) -> list[dict]:
    """
    Load all GFS GRIB2 files sorted by forecast hour.
    Returns list of dicts: { fhr, filepath, prate, valid_time }
    """
    pattern = os.path.join(gfs_dir, "gfs.t??z.pgrb2.0p25.f*.grib2")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No GFS GRIB2 files found in {gfs_dir}\n"
            f"Expected: gfs.tHHz.pgrb2.0p25.fNNN.grib2"
        )

    log.info(f"Found {len(files)} GRIB2 files")

    records = []
    for f in files:
        fname = os.path.basename(f)
        fhr   = int(fname.split(".f")[-1].replace(".grib2", ""))
        try:
            prate = load_prate_from_file(f)
            vtime = str(prate.valid_time.values)[:16] if hasattr(prate, "valid_time") else f"f{fhr:03d}"
            records.append({
                "fhr": fhr, "filepath": f,
                "prate": prate, "valid_time": vtime
            })
            log.info(f"  f{fhr:03d}: loaded  valid={vtime}")
        except Exception as e:
            log.warning(f"  f{fhr:03d}: FAILED — {e}")

    if not records:
        raise RuntimeError("No prate data could be loaded.")

    return records


# =============================================================================
#  PRECIPITATION ACCUMULATION
# =============================================================================

def compute_accumulated_rainfall(records: list[dict]) -> tuple[np.ndarray, list]:
    """
    Convert prate (kg/m2/s) to 3-hourly rainfall (mm).
    rain_mm = prate * STEP_SECONDS per step.
    Returns cumulative 2D array and list of per-step 2D arrays.
    """
    step_rains = []
    cumulative = None

    for rec in records:
        prate_np = rec["prate"].values.astype(np.float32)
        rain_3h  = prate_np * STEP_SECONDS
        rain_3h  = np.where(rain_3h < 0.01, np.nan, rain_3h)
        step_rains.append(rain_3h)

        if cumulative is None:
            cumulative = np.zeros_like(rain_3h)
        cumulative = np.nansum([cumulative, rain_3h], axis=0)

    return cumulative, step_rains


# =============================================================================
#  COLORMAP
# =============================================================================

def build_colormap():
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    return cmap, norm


# =============================================================================
#  MAP FEATURES
# =============================================================================

def add_state_mask(ax, kl_state):
    """White mask outside Kerala so GFS data only shows inside the state."""
    outer    = box(60.0, -5.0, 95.0, 40.0)
    kl_union = unary_union(kl_state.geometry.values)
    outside  = outer.difference(kl_union)
    ax.add_geometries(
        [outside], crs=ccrs.PlateCarree(),
        facecolor="white", edgecolor="none", zorder=3
    )


def apply_map_features(ax, kl_districts, kl_state):
    """Apply all Kerala map decorations."""
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="white",   zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce6ff", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5,        zorder=5)

    # White mask outside Kerala
    add_state_mask(ax, kl_state)

    # District boundaries
    kl_districts.boundary.plot(
        ax=ax, edgecolor="#FF00FF", linewidth=0.5, zorder=6, aspect=None
    )

    # State boundary
    kl_state.boundary.plot(
        ax=ax, edgecolor="#FF00FF", linewidth=2.0, zorder=7, aspect=None
    )

    # District labels
    for _, row in kl_districts.iterrows():
        point     = row.geometry.representative_point()
        dist_name = row.get(DIST_COL, "")
        if dist_name:
            ax.text(
                point.x, point.y, dist_name,
                transform=ccrs.PlateCarree(),
                fontsize=5.5, color="#880088", fontweight="bold",
                ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )

    # State label
    centre = kl_state.geometry.iloc[0].representative_point()
    ax.text(
        centre.x, centre.y + 1.0,
        STATE_LABEL.upper(),
        transform=ccrs.PlateCarree(),
        fontsize=11, color="#CC0000", fontweight="bold",
        ha="center", va="center", zorder=10,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
    )

    # Gridlines
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


def add_imd_credit(fig):
    fig.text(
        0.01, 0.005,
        "IMD OPERATIONAL GLOBAL MODEL COURTESY : BTM, NCMRWF",
        ha="left", fontsize=6.5, color="#555555"
    )


def remove_mesh(cf):
    if cf is not None:
        try:
            cf.remove()
        except Exception:
            pass


# =============================================================================
#  PLOT BUILDER
# =============================================================================

def make_plot(fig, ax, lons, lats, data, cmap, norm,
              param_label, forecast_hours,
              based_on_str, valid_for_str,
              kl_districts, kl_state):
    """Render one Kerala-masked GFS rainfall map."""
    apply_map_features(ax, kl_districts, kl_state)

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
    add_imd_credit(fig)
    return cf


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_pipeline(gfs_dir: str, output_dir: str, cycle: str, date: str):
    """
    Full Kerala GFS rainfall pipeline:
      1. Load GRIB2 files + geodata
      2. Cumulative rainfall map
      3. Per-step 3-hourly PNGs
      4. Animated MP4
    """
    if not os.path.isdir(gfs_dir):
        raise FileNotFoundError(f"GFS directory not found: {gfs_dir}")

    os.makedirs(output_dir, exist_ok=True)
    step_dir = os.path.join(output_dir, f"gfs_{STATE_CODE.lower()}_stepwise_rainfall")
    os.makedirs(step_dir, exist_ok=True)

    log.info("=" * 60)
    log.info(f"GFS {STATE_LABEL} Rainfall Pipeline  —  {date} {cycle}z")
    log.info(f"  GFS dir   : {gfs_dir}")
    log.info(f"  Output dir: {output_dir}")
    log.info("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("[1/4] Loading GRIB2 files and geodata ...")
    records              = load_all_files(gfs_dir)
    kl_districts, kl_state = load_geodata()
    cmap, norm           = build_colormap()

    prate0 = records[0]["prate"]
    lats   = prate0.latitude.values
    lons   = prate0.longitude.values

    init_str    = f"{cycle} UTC of {date[6:8]}-{date[4:6]}-{date[0:4]}"
    total_hours = records[-1]["fhr"] - records[0]["fhr"]

    log.info("[1/4] Computing accumulated rainfall ...")
    cumulative, step_rains = compute_accumulated_rainfall(records)

    # ── 2. Cumulative map ─────────────────────────────────────────────────────
    log.info("[2/4] Generating cumulative rainfall map ...")
    cum_masked = np.where(cumulative < 1.0, np.nan, cumulative)

    fig, ax = plt.subplots(
        figsize=(8, 12),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    make_plot(
        fig, ax, lons, lats, cum_masked, cmap, norm,
        param_label="TOTAL RAINFALL (mm)",
        forecast_hours=total_hours,
        based_on_str=init_str,
        valid_for_str=records[-1]["valid_time"],
        kl_districts=kl_districts,
        kl_state=kl_state
    )
    add_logo(fig)

    out_cum = os.path.join(
        output_dir,
        f"gfs_{STATE_CODE.lower()}_cumulative_rainfall.png"
    )
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out_cum}")

    # ── 3. Per-step PNGs ──────────────────────────────────────────────────────
    log.info(f"[3/4] Generating {len(records)} stepwise 3-hourly plots ...")

    for i, rec in enumerate(records):
        rain = np.where(step_rains[i] < 0.1, np.nan, step_rains[i])

        fig, ax = plt.subplots(
            figsize=(8, 12),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        make_plot(
            fig, ax, lons, lats, rain, cmap, norm,
            param_label="3-HOURLY RAINFALL (mm)",
            forecast_hours=3,
            based_on_str=init_str,
            valid_for_str=rec["valid_time"],
            kl_districts=kl_districts,
            kl_state=kl_state
        )
        add_logo(fig)

        fname    = f"gfs_{STATE_CODE.lower()}_rainfall_f{rec['fhr']:03d}.png"
        out_path = os.path.join(step_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        log.info(f"  f{rec['fhr']:03d}: {fname}")

    # ── 4. Animation ──────────────────────────────────────────────────────────
    log.info("[4/4] Generating animation ...")

    fig2, ax2 = plt.subplots(
        figsize=(8, 12),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    apply_map_features(ax2, kl_districts, kl_state)
    add_logo(fig2)
    add_footer(fig2)
    add_imd_credit(fig2)

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
    out_ani = os.path.join(
        output_dir,
        f"gfs_{STATE_CODE.lower()}_rainfall_animation.mp4"
    )
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    log.info(f"  Saved: {out_ani}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"DONE — {STATE_LABEL} outputs saved to: {output_dir}")
    log.info(f"  • gfs_{STATE_CODE.lower()}_cumulative_rainfall.png")
    log.info(f"  • gfs_{STATE_CODE.lower()}_rainfall_animation.mp4")
    log.info(f"  • gfs_{STATE_CODE.lower()}_stepwise_rainfall/  ({len(records)} PNGs)")
    log.info("=" * 60)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=f"GFS {STATE_LABEL} Rainfall Plotter v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    today = datetime.now().strftime("%Y%m%d")

    parser.add_argument(
        "--gfsdir",
        default=f"/home/ras_08/WEATHER/INDIA/GFS-DATA/{today}/00z",
        help="Directory containing GFS GRIB2 files"
    )
    parser.add_argument(
        "--outdir",
        default=f"/home/ras_08/WEATHER/INDIA/Data/{today}/gfs_kerala",
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
        help="Date YYYYMMDD (default: today)"
    )

    args = parser.parse_args()

    log.info(f"Script : plot_gfs_kerala_rainfall.py v{__version__}")
    log.info(f"State  : {STATE_LABEL}  ({STATE_NAME})")
    log.info(f"GFS dir: {args.gfsdir}")
    log.info(f"Output : {args.outdir}")

    run_pipeline(
        gfs_dir=args.gfsdir,
        output_dir=args.outdir,
        cycle=args.cycle,
        date=args.date
    )
