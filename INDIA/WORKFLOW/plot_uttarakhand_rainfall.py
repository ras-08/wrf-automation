import os
import argparse
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
#  CONFIGURATION
# =============================================================================

SHAPEFILE_PATH = "/home/ras_08/WEATHER/INDIA/WORKFLOW/UTTARAKHAND/UTTARAKHAND_DISTRICTS.geojson"

# Uttarakhand bounding box (with small buffer)
UK_LON_MIN = 77.5
UK_LON_MAX = 81.0
UK_LAT_MIN = 28.7
UK_LAT_MAX = 31.5

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
#  HELPER FUNCTIONS
# =============================================================================

def load_geodata():
    """Load GeoJSON and extract Uttarakhand districts + boundary."""
    counties = gpd.read_file(SHAPEFILE_PATH)

    # Check and set CRS
    if counties.crs is None or counties.crs.to_epsg() != 4326:
        counties = counties.set_crs(epsg=4326, allow_override=True)

    # Dissolve by state name to get state boundary
    states = counties.dissolve(by="stname")

    # Uttarakhand only — column is 'stname'
    uk_districts = counties[counties["stname"] == "UTTARAKHAND"].copy()
    uk_state     = states[states.index == "UTTARAKHAND"].copy()

    return counties, states, uk_districts, uk_state


def build_colormap():
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    return cmap, norm


def fmt_bfs_time_ts(ts: pd.Timestamp) -> str:
    return ts.strftime('%H UTC of %d-%m-%Y')


def add_uk_mask(ax, uk_state):
    """
    Add a white mask OUTSIDE Uttarakhand so only UK data is visible.
    """
    outer       = box(60.0, -10.0, 100.0, 40.0)
    uk_union    = unary_union(uk_state.geometry.values)
    outside     = outer.difference(uk_union)

    ax.add_geometries(
        [outside],
        crs=ccrs.PlateCarree(),
        facecolor="white",
        edgecolor="none",
        zorder=3
    )


def apply_uk_map_features(axis, uk_districts, uk_state):
    """Map decorations zoomed to Uttarakhand with exact boundary masking."""

    axis.set_extent(
        [UK_LON_MIN, UK_LON_MAX, UK_LAT_MIN, UK_LAT_MAX],
        crs=ccrs.PlateCarree()
    )

    # Base features
    axis.add_feature(cfeature.LAND,      facecolor="white", zorder=0)
    axis.add_feature(cfeature.OCEAN,     facecolor="#cce6ff", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=5)

    # White mask outside Uttarakhand
    add_uk_mask(axis, uk_state)

    # District boundaries (thin magenta)
    uk_districts.boundary.plot(
        ax=axis, edgecolor="#FF00FF", linewidth=0.5, zorder=6, aspect=None
    )

    # State outer boundary (thick magenta)
    uk_state.boundary.plot(
        ax=axis, edgecolor="#FF00FF", linewidth=2.0, zorder=7, aspect=None
    )

    # District name labels — column is 'dtname'
    for _, row in uk_districts.iterrows():
        point     = row.geometry.representative_point()
        dist_name = row.get("dtname", "")
        if dist_name:
            axis.text(
                point.x, point.y,
                dist_name,
                transform=ccrs.PlateCarree(),
                fontsize=5.5, color="#880088", fontweight="bold",
                ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )

    # "UTTARAKHAND" label at state centre
    uk_point = uk_state.geometry.iloc[0].representative_point()
    axis.text(
        uk_point.x, uk_point.y + 0.5,
        "UTTARAKHAND",
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


def add_weatherex_logo(fig):
    logo_path = "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"
    if os.path.exists(logo_path):
        logo    = mpimg.imread(logo_path)
        logo_ax = fig.add_axes([0.01, 0.80, 0.15, 0.15])
        logo_ax.imshow(logo)
        logo_ax.axis("off")


def add_bfs_colorbar(fig, ax, mappable):
    cbar = fig.colorbar(
        mappable, ax=ax, orientation="vertical",
        pad=0.02, shrink=0.50, fraction=0.03, extend="max", ticks=LEVELS
    )
    cbar.ax.set_yticklabels(
        [str(int(l)) if l == int(l) else str(l) for l in LEVELS],
        fontsize=9
    )
    cbar.ax.tick_params(length=4)
    return cbar


def set_bfs_title(ax, param_label, forecast_hours, based_on_str, valid_for_str):
    line1 = f"NWP (9 Km) {param_label} FORECAST ({forecast_hours} HR)"
    line2 = f"based on {based_on_str} valid for {valid_for_str}"
    ax.set_title(
        f"{line1}\n{line2}",
        fontsize=12, fontweight="bold",
        loc="center", pad=8, linespacing=1.6, family="monospace"
    )


def add_bfs_footer(fig):
    fig.text(0.5, 0.005,
             "(Background does not depict political boundary)",
             ha="center", fontsize=8, style="italic", color="#444444")


def add_imd_logo_text(fig):
    fig.text(0.01, 0.005,
             "NWP MODEL OUTPUT | WeatherEx.Ai""NWP MODEL OUTPUT | WeatherEx.Ai",
             ha="left", fontsize=6.5, color="#555555")


def remove_pcolormesh(cf):
    if cf is not None:
        try:
            cf.remove()
        except Exception:
            pass


def make_uk_plot(fig, ax, lons, lats, data, cmap, norm,
                 param_label, forecast_hours,
                 based_on_str, valid_for_str,
                 uk_districts, uk_state):
    """Draw one Uttarakhand-masked rainfall plot."""
    apply_uk_map_features(ax, uk_districts, uk_state)

    cf = ax.pcolormesh(
        to_np(lons), to_np(lats), data,
        cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(), zorder=2, alpha=0.9
    )
    add_bfs_colorbar(fig, ax, cf)
    set_bfs_title(ax,
        param_label=param_label,
        forecast_hours=forecast_hours,
        based_on_str=based_on_str,
        valid_for_str=valid_for_str
    )
    add_bfs_footer(fig)
    add_imd_logo_text(fig)
    return cf


def get_day_intervals(ncfile, ntimes):
    """Group WRF time indices into 00 UTC → 00 UTC next day buckets."""
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
#  MAIN PIPELINE
# =============================================================================

def run_uttarakhand_rainfall_pipeline(wrf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    daily_dir = os.path.join(output_dir, "daily_rainfall_uttarakhand")
    os.makedirs(daily_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("[1/4] Loading WRF file and shapefiles...")
    ncfile = Dataset(wrf_path)
    counties, states, uk_districts, uk_state = load_geodata()
    cmap, norm = build_colormap()

    rain_t0      = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons   = latlon_coords(rain_t0)
    ntimes       = ncfile.dimensions["Time"].size
    start_ts     = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=0).values))
    end_ts       = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=-1).values))
    based_on_str = fmt_bfs_time_ts(start_ts)
    delta_hours  = int((end_ts - start_ts).total_seconds() // 3600)

    # -------------------------------------------------------------------------
    # 2. FULL CUMULATIVE — UTTARAKHAND
    # -------------------------------------------------------------------------
    print("[2/4] Generating Uttarakhand cumulative rainfall map...")

    rain_end   = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = np.ma.masked_less(to_np(rain_end) - to_np(rain_t0), 1.0)

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    make_uk_plot(
        fig, ax, lons, lats, rain_total, cmap, norm,
        param_label="RAINFALL (mm)",
        forecast_hours=delta_hours,
        based_on_str=based_on_str,
        valid_for_str=fmt_bfs_time_ts(end_ts),
        uk_districts=uk_districts,
        uk_state=uk_state
    )
    add_weatherex_logo(fig)

    out_cum = os.path.join(output_dir, "uttarakhand_cumulative_rainfall.png")
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_cum}")

    # -------------------------------------------------------------------------
    # 3. PER-DAY PLOTS — UTTARAKHAND
    # -------------------------------------------------------------------------
    day_intervals = get_day_intervals(ncfile, ntimes)
    print(f"[3/4] Generating {len(day_intervals)} daily Uttarakhand plot(s)...")

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
            figsize=(10, 8),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        make_uk_plot(
            fig, ax, lons, lats, daily, cmap, norm,
            param_label="RAINFALL (mm)",
            forecast_hours=day_hrs,
            based_on_str=fmt_bfs_time_ts(s_ts),
            valid_for_str=fmt_bfs_time_ts(e_ts),
            uk_districts=uk_districts,
            uk_state=uk_state
        )
        add_weatherex_logo(fig)

        fname    = f"uttarakhand_daily_Day{day_num}_{s_ts.strftime('%d%b%Y')}.png"
        out_path = os.path.join(daily_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Day {day_num}: {s_ts.strftime('%d %b %Y %H UTC')} → "
              f"{e_ts.strftime('%d %b %Y %H UTC')}  →  {fname}")

    # -------------------------------------------------------------------------
    # 4. HOURLY ANIMATION — UTTARAKHAND
    # -------------------------------------------------------------------------
    print("[4/4] Generating Uttarakhand hourly animation...")

    fig2, ax2 = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    apply_uk_map_features(ax2, uk_districts, uk_state)
    add_weatherex_logo(fig2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_bfs_colorbar(fig2, ax2, sm)
    add_bfs_footer(fig2)
    add_imd_logo_text(fig2)

    _current_cf = [None]

    def update(frame):
        remove_pcolormesh(_current_cf[0])

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
        _current_cf[0] = cf_new

        t_prev = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame).values))
        t_now  = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame+1).values))
        set_bfs_title(ax2,
            param_label="RAINFALL (mm)",
            forecast_hours=1,
            based_on_str=fmt_bfs_time_ts(t_prev),
            valid_for_str=fmt_bfs_time_ts(t_now)
        )
        return []

    ani = animation.FuncAnimation(
        fig2, update, frames=ntimes - 1,
        interval=1500, blit=False
    )
    out_ani = os.path.join(output_dir, "uttarakhand_hourly_animation.mp4")
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    print(f"    Saved: {out_ani}")

    ncfile.close()
    print("\nDone! All Uttarakhand outputs saved to:", output_dir)
    print(f"  • uttarakhand_cumulative_rainfall.png")
    print(f"  • uttarakhand_hourly_animation.mp4")
    print(f"  • daily_rainfall_uttarakhand/  ({len(day_intervals)} daily PNG files)")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NWP Uttarakhand Rainfall output"
    )

    today = datetime.now().strftime("%Y-%m-%d")

    DEFAULT_INPUT = (
        f"/home/ras_08/Models/WRF_TUTORIAL/WRFV4.5/run/"
        f"wrfout_d01_{today}_00:00:00"
    )

    DEFAULT_OUTDIR = (
        f"/home/ras_08/WEATHER/INDIA/Data/"
        f"{datetime.now().strftime('%Y%m%d')}"
    )

    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="WRF output file")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")

    args = parser.parse_args()

    run_uttarakhand_rainfall_pipeline(args.input, args.outdir)
