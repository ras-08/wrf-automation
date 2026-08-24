#!/usr/bin/env python3
"""
plot_maharashtra_rainfall.py
============================
NWP WRF Rainfall plots for Maharashtra.

Outputs:
  • maharashtra_cumulative_rainfall.png
  • daily_rainfall_maharashtra/maharashtra_daily_DayN_DDMONYYYY.png
  • maharashtra_hourly_animation.mp4

Usage:
  python plot_maharashtra_rainfall.py --input wrfout_d01_YYYY-MM-DD_HH:00:00 --outdir /path/to/output
"""

__version__ = "1.0"
__state__   = "Maharashtra"
__author__  = "ras_08 / WeatherEx"

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
from datetime import datetime, timedelta
from wrf import getvar, latlon_coords, to_np
import pandas as pd
#!/usr/bin/env python3
"""
plot_maharashtra_rainfall.py
============================
NWP WRF Rainfall plots for Maharashtra.

Outputs:
  • maharashtra_cumulative_rainfall.png
  • daily_rainfall_maharashtra/maharashtra_daily_DayN_DDMONYYYY.png
  • maharashtra_hourly_animation.mp4

Usage:
  python plot_maharashtra_rainfall.py --input wrfout_d01_YYYY-MM-DD_HH:00:00 --outdir /path/to/output
"""

__version__ = "1.0"
__state__   = "Maharashtra"
__author__  = "ras_08 / WeatherEx"

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
#  CONFIGURATION  — only change this block for a new state
# =============================================================================

STATE_NAME   = "MAHARASHTRA"       # must match 'stname' column in GeoJSON exactly
STATE_LABEL  = "Maharashtra"       # displayed on the map
STATE_CODE   = "MH"                # used in output filenames
NWP_RES      = "9 Km"            # model resolution label in title
DIST_COL     = "dtname"           # district name column in GeoJSON
MODEL_NAME = "WFS (WeatherEx Forecasting System)"
DOMAIN_NAME = STATE_LABEL
SHAPEFILE_PATH = (
    "/home/ras_08/WEATHER/INDIA/WORKFLOW/MAHARASHTRA/MAHARASHTRA_DISTRICTS.geojson"
)

LOGO_PATH = "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"

# Bounding box — add ~0.5 deg buffer around the state
MH_LON_MIN = 72.5
MH_LON_MAX = 80.9
MH_LAT_MIN = 15.6
MH_LAT_MAX = 22.1

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


# =============================================================================
#  GEODATA LOADER
# =============================================================================

def load_geodata():
    """
    Load GeoJSON, dissolve to state boundary, extract state districts.
    Raises ValueError if the state is not found in the file.
    """
    log.info(f"Loading shapefile: {SHAPEFILE_PATH}")
    if not os.path.exists(SHAPEFILE_PATH):
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")

    counties = gpd.read_file(SHAPEFILE_PATH)
    log.info(f"  Columns   : {counties.columns.tolist()}")
    log.info(f"  CRS       : {counties.crs}")
    log.info(f"  Total rows: {len(counties)}")

    # Ensure WGS84
    if counties.crs is None or counties.crs.to_epsg() != 4326:
        log.warning("CRS not EPSG:4326 — overriding to WGS84")
        counties = counties.set_crs(epsg=4326, allow_override=True)

    # Dissolve to state boundary
    states = counties.dissolve(by="stname")

    # Filter to target state
    mh_districts = counties[counties["stname"] == STATE_NAME].copy()
    mh_state     = states[states.index == STATE_NAME].copy()

    # Validate — fail loudly with helpful message
    if mh_districts.empty:
        available = counties["stname"].unique().tolist()
        raise ValueError(
            f"State '{STATE_NAME}' not found in GeoJSON.\n"
            f"Available states: {available}\n"
            f"Check STATE_NAME in the CONFIGURATION block."
        )

    log.info(f"  Districts found for {STATE_NAME}: {len(mh_districts)}")
    return counties, states, mh_districts, mh_state


# =============================================================================
#  COLORMAP
# =============================================================================

def build_colormap():
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    return cmap, norm


# =============================================================================
#  TIME FORMATTERS
# =============================================================================

def fmt_time(ts: pd.Timestamp) -> str:
    """Format timestamp for plot titles: '00 UTC of 19-06-2026'"""
    return ts.strftime('%H UTC of %d-%m-%Y')


# =============================================================================
#  MAP FEATURES
# =============================================================================

def add_state_mask(ax, mh_state):
    """
    White mask outside Maharashtra so WRF data only shows inside the state.
    Uses a large bounding box minus the state polygon.
    """
    outer    = box(50.0, -10.0, 110.0, 45.0)
    mh_union = unary_union(mh_state.geometry.values)
    outside  = outer.difference(mh_union)

    ax.add_geometries(
        [outside],
        crs=ccrs.PlateCarree(),
        facecolor="white",
        edgecolor="none",
        zorder=3
    )


def apply_map_features(axis, mh_districts, mh_state):
    """
    Apply all map decorations for Maharashtra:
      - Zoom to bounding box
      - Land / ocean / coastline
      - White mask outside state
      - District boundaries
      - State boundary
      - District name labels
      - State name label
      - Gridlines
    """
    # Zoom
    axis.set_extent(
        [MH_LON_MIN, MH_LON_MAX, MH_LAT_MIN, MH_LAT_MAX],
        crs=ccrs.PlateCarree()
    )

    # Base features
    axis.add_feature(cfeature.LAND,      facecolor="white",    zorder=0)
    axis.add_feature(cfeature.OCEAN,     facecolor="#cce6ff",  zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=1.5,         zorder=5)

    # Mask outside state
    add_state_mask(axis, mh_state)

    # District boundaries (thin magenta)
    mh_districts.boundary.plot(
        ax=axis, edgecolor="#FF00FF", linewidth=0.5, zorder=6, aspect=None
    )

    # State outer boundary (thick magenta)
    mh_state.boundary.plot(
        ax=axis, edgecolor="#FF00FF", linewidth=2.0, zorder=7, aspect=None
    )

    # District name labels
    for _, row in mh_districts.iterrows():
        point     = row.geometry.representative_point()
        dist_name = row.get(DIST_COL, "")
        if dist_name:
            axis.text(
                point.x, point.y,
                dist_name,
                transform=ccrs.PlateCarree(),
                fontsize=5.0, color="#880088", fontweight="bold",
                ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )

    # State centre label
    centre = mh_state.geometry.iloc[0].representative_point()
    axis.text(
        centre.x, centre.y + 0.5,
        STATE_LABEL.upper(),
        transform=ccrs.PlateCarree(),
        fontsize=11, color="#CC0000", fontweight="bold",
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
    """Add WeatherEx logo — silently skips if file missing."""
    if os.path.exists(LOGO_PATH):
        logo    = mpimg.imread(LOGO_PATH)
        logo_ax = fig.add_axes([0.01, 0.80, 0.15, 0.15])
        logo_ax.imshow(logo)
        logo_ax.axis("off")
    else:
        log.warning(f"Logo not found at {LOGO_PATH} — skipping")


def add_colorbar(fig, ax, mappable):
    """Vertical colorbar on the right side."""
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


def set_title(ax, param_label, forecast_hours, based_on_ts, valid_start_ts=None, valid_end_ts=None):

    based_on_str = based_on_ts.strftime("%H UTC %d-%b-%Y")

    if valid_start_ts is not None and valid_end_ts is not None:
        valid_str = (
            f"{valid_start_ts.strftime('%H UTC %d-%b-%Y')} "
            f"to {valid_end_ts.strftime('%H UTC %d-%b-%Y')}"
        )
    else:
        valid_end = based_on_ts + timedelta(hours=forecast_hours)
        valid_str = (
            f"{based_on_ts.strftime('%H UTC %d-%b-%Y')} "
            f"to {valid_end.strftime('%H UTC %d-%b-%Y')}"
        )

    ax.set_title(
        f"{MODEL_NAME} ({NWP_RES}) {DOMAIN_NAME} {param_label} ({forecast_hours}-HR FCST)\n"
        f"Based on: {based_on_str}    Valid for: {valid_str}",
        fontsize=10, fontweight="bold", loc="center",
        pad=8, linespacing=1.6, family="monospace",
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
        "NWP MODEL OUTPUT | WeatherEx.Ai",
        ha="left", fontsize=6.5, color="#555555"
    )


# =============================================================================
#  PLOT BUILDER
# =============================================================================
def make_plot(fig,ax,lons,lats,
    data,cmap,norm,mh_districts,mh_state,
    param_label,forecast_hours,based_on_ts,
    valid_start_ts=None,valid_end_ts=None,):
    """
    Render one complete Maharashtra rainfall map onto fig/ax.
    Returns the pcolormesh object for colorbar reuse.
    """
    apply_map_features(ax, mh_districts, mh_state)

    #cf = ax.pcolormesh(
     #   to_np(lons), to_np(lats), data,
       # cmap=cmap, norm=norm, shading="auto",
      #  transform=ccrs.PlateCarree(), zorder=2, alpha=0.9
    #)
    

    cf = ax.contourf(
         to_np(lons),
         to_np(lats),
         data,
         levels=LEVELS,
         cmap=cmap,
         norm=norm,                    # add this
         transform=ccrs.PlateCarree(),
         extend="max",
         zorder=2                      # optional
    )

    add_colorbar(fig, ax, cf)
    set_title(ax, param_label, forecast_hours, based_on_ts, valid_start_ts, valid_end_ts)
    add_footer(fig)
    add_imd_credit(fig)
    return cf


def remove_pcolormesh(cf):
    """Safely remove a pcolormesh object from the axes."""
    if cf is not None:
        try:
            cf.remove()
        except Exception as e:
            log.debug(f"remove_pcolormesh: {e}")


# =============================================================================
#  TIME INTERVAL HELPER
# =============================================================================

def get_day_intervals(ncfile, ntimes):
    """
    Group WRF time indices into 00 UTC → 00 UTC next day buckets.
    Used for per-day accumulated rainfall plots.

    Returns list of dicts:
      { start_idx, end_idx, start_ts, end_ts }
    """
    all_ts = [
        pd.to_datetime(str(getvar(ncfile, "Times", timeidx=i).values))
        for i in range(ntimes)
    ]

    midnight_indices = [i for i, ts in enumerate(all_ts) if ts.hour == 0]
    intervals = []

    if not midnight_indices:
        # No midnight crossing — treat whole run as one interval
        intervals.append({
            "start_idx": 0,        "end_idx": ntimes - 1,
            "start_ts":  all_ts[0], "end_ts":  all_ts[-1],
        })
        return intervals

    # Partial day before first midnight
    if midnight_indices[0] != 0:
        intervals.append({
            "start_idx": 0,
            "end_idx":   midnight_indices[0],
            "start_ts":  all_ts[0],
            "end_ts":    all_ts[midnight_indices[0]],
        })

    # Full day intervals between consecutive midnights
    for k in range(len(midnight_indices) - 1):
        s, e = midnight_indices[k], midnight_indices[k + 1]
        intervals.append({
            "start_idx": s, "end_idx": e,
            "start_ts":  all_ts[s], "end_ts": all_ts[e],
        })

    # Partial day after last midnight
    last_mid = midnight_indices[-1]
    if last_mid < ntimes - 1:
        intervals.append({
            "start_idx": last_mid,       "end_idx": ntimes - 1,
            "start_ts":  all_ts[last_mid], "end_ts": all_ts[-1],
        })

    return intervals


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_pipeline(wrf_path: str, output_dir: str):
    """
    Full Maharashtra rainfall pipeline:
      1. Load WRF file + geodata
      2. Cumulative rainfall map
      3. Per-day rainfall maps
      4. Hourly animation (MP4)
    """

    # ── Input validation ──────────────────────────────────────────────────────
    if not os.path.exists(wrf_path):
        raise FileNotFoundError(f"WRF output file not found: {wrf_path}")

    os.makedirs(output_dir, exist_ok=True)
    daily_dir = os.path.join(output_dir, f"daily_rainfall_{STATE_CODE.lower()}")
    os.makedirs(daily_dir, exist_ok=True)

    log.info("=" * 60)
    log.info(f"NWP Rainfall Pipeline — {STATE_LABEL}")
    log.info(f"  WRF file  : {wrf_path}")
    log.info(f"  Output dir: {output_dir}")
    log.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("[1/4] Loading WRF file and geodata ...")
    ncfile = Dataset(wrf_path)
    counties, states, mh_districts, mh_state = load_geodata()
    cmap, norm = build_colormap()

    rain_t0      = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons   = latlon_coords(rain_t0)
    ntimes       = ncfile.dimensions["Time"].size
    start_ts     = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=0).values))
    end_ts       = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=-1).values))
    #based_on_str = fmt_time(start_ts)
    delta_hours  = int((end_ts - start_ts).total_seconds() // 3600)

    log.info(f"  WRF start : {start_ts}")
    log.info(f"  WRF end   : {end_ts}  ({delta_hours}h)")
    log.info(f"  Time steps: {ntimes}")

    # ── 2. Cumulative rainfall ────────────────────────────────────────────────
    log.info(f"[2/4] Generating {STATE_LABEL} cumulative rainfall map ...")

    rain_end   = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = np.ma.masked_less(to_np(rain_end) - to_np(rain_t0), 1.0)

    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    make_plot(
        fig,ax,lons, lats, rain_total, cmap, norm, mh_districts, mh_state,
        param_label    = "CUMULATIVE RAINFALL (mm)",
        forecast_hours = delta_hours,
        based_on_ts    = start_ts,
        valid_start_ts = start_ts,
        valid_end_ts   = end_ts,
    )
    add_logo(fig)

    out_cum = os.path.join(
        output_dir,
        f"{STATE_CODE.lower()}_cumulative_rainfall.png"
    )
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {out_cum}")

    # ── 3. Per-day plots ──────────────────────────────────────────────────────
    day_intervals = get_day_intervals(ncfile, ntimes)
    log.info(f"[3/4] Generating {len(day_intervals)} daily plot(s) ...")

    for day_num, interval in enumerate(day_intervals, start=1):
        s_idx   = interval["start_idx"]
        e_idx   = interval["end_idx"]
        s_ts    = interval["start_ts"]
        e_ts    = interval["end_ts"]
        day_hrs = int((e_ts - s_ts).total_seconds() // 3600)

        r_start = to_np(
            getvar(ncfile, "RAINC",  timeidx=s_idx) +
            getvar(ncfile, "RAINNC", timeidx=s_idx)
        )
        r_end = to_np(
            getvar(ncfile, "RAINC",  timeidx=e_idx) +
            getvar(ncfile, "RAINNC", timeidx=e_idx)
        )
        daily = np.ma.masked_less(r_end - r_start, 1.0)

        fig, ax = plt.subplots(
            figsize=(12, 8),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        make_plot(
            fig,ax,lons, lats, daily, cmap, norm, mh_districts, mh_state,
            param_label    = f"DAY {day_num} RAINFALL (mm)",
            forecast_hours = day_hrs,
            based_on_ts    = start_ts,
            valid_start_ts = s_ts,
            valid_end_ts   = e_ts,
        )
        add_logo(fig)

        fname    = f"{STATE_CODE.lower()}_daily_Day{day_num}_{s_ts.strftime('%d%b%Y')}.png"
        out_path = os.path.join(daily_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        log.info(
            f"  Day {day_num}: {s_ts.strftime('%d %b %Y %H UTC')} → "
            f"{e_ts.strftime('%d %b %Y %H UTC')}  →  {fname}"
        )

    # ── 4. Hourly animation ───────────────────────────────────────────────────
    log.info(f"[4/4] Generating {STATE_LABEL} hourly animation ...")

    fig2, ax2 = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    apply_map_features(ax2, mh_districts, mh_state)
    add_logo(fig2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_colorbar(fig2, ax2, sm)
    add_footer(fig2)
    add_imd_credit(fig2)

    _current_cf = [None]

    def update(frame):
        remove_pcolormesh(_current_cf[0])

        r_prev = to_np(
            getvar(ncfile, "RAINC",  timeidx=frame) +
            getvar(ncfile, "RAINNC", timeidx=frame)
        )
        r_now = to_np(
            getvar(ncfile, "RAINC",  timeidx=frame + 1) +
            getvar(ncfile, "RAINNC", timeidx=frame + 1)
        )
        hourly = np.ma.masked_less_equal(r_now - r_prev, 0.1)

        cf_new = ax2.pcolormesh(
            to_np(lons), to_np(lats), hourly,
            cmap=cmap, norm=norm, shading="auto",
            transform=ccrs.PlateCarree(), zorder=2, alpha=0.85
        )
        _current_cf[0] = cf_new

        t_prev = pd.to_datetime(
            str(getvar(ncfile, "Times", timeidx=frame).values)
        )
        t_now = pd.to_datetime(
            str(getvar(ncfile, "Times", timeidx=frame + 1).values)
        )
        elapsed = frame + 1

        set_title(
        ax2,
        param_label=f"HOURLY RAINFALL (mm) [+{elapsed:02d}h]",
        forecast_hours=1,
        based_on_ts=start_ts,
        valid_start_ts=t_prev,
        valid_end_ts=t_now,
        )
        return []

    ani = animation.FuncAnimation(
        fig2, update,
        frames=ntimes - 1,
        interval=1500, blit=False
    )

    out_ani = os.path.join(
        output_dir,
        f"{STATE_CODE.lower()}_hourly_animation.mp4"
    )
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    log.info(f"  Saved: {out_ani}")

    ncfile.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"DONE — All {STATE_LABEL} outputs saved to: {output_dir}")
    log.info(f"  • {STATE_CODE.lower()}_cumulative_rainfall.png")
    log.info(f"  • {STATE_CODE.lower()}_hourly_animation.mp4")
    log.info(f"  • daily_rainfall_{STATE_CODE.lower()}/  ({len(day_intervals)} files)")
    log.info("=" * 60)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=f"NWP {STATE_LABEL} Rainfall — WRF output plotter v{__version__}",
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
        help="Path to WRF output file (default: today's 00z wrfout)"
    )
    parser.add_argument(
        "--outdir",
        default=(
            f"/home/ras_08/WEATHER/INDIA/Data/"
            f"{datetime.now().strftime('%Y%m%d')}"
        ),
        help="Output directory for plots and animation"
    )

    args = parser.parse_args()

    log.info(f"Script    : plot_maharashtra_rainfall.py v{__version__}")
    log.info(f"State     : {STATE_LABEL}  ({STATE_NAME})")
    log.info(f"Input     : {args.input}")
    log.info(f"Output    : {args.outdir}")

    run_pipeline(args.input, args.outdir)

