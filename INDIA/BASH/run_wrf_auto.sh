#!/bin/bash
# =============================================================================
#  run_wrf_auto.sh
#  Automated WRF pipeline: GFS download → WPS → WRF → Plotting
#
#  Usage:
#   ./run_wrf_auto.sh 00
#   ./run_wrf_auto.sh 12
#   ./run_wrf_auto.sh 00 --date 20260617
#   ./run_wrf_auto.sh 00 --skip-download
# =============================================================================

set -eo pipefail

# =============================================================================
# 0. ARGUMENT PARSING
# =============================================================================
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {00|06|12|18} [--date YYYYMMDD] [--skip-download]"
    exit 1
fi

CYCLE="$1"; shift

if [[ ! "$CYCLE" =~ ^(00|06|12|18)$ ]]; then
    echo "ERROR: cycle must be one of 00, 06, 12, 18 (got '$CYCLE')"
    exit 1
fi

FORCE_DATE=""
SKIP_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)          FORCE_DATE="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
HOME_DIR="/home/ras_08"
WEATHER_DIR="$HOME_DIR/WEATHER"
WPS_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WPS-4.5"
WRF_RUN_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WRFV4.5/run"
WORKFLOW_DIR="$WEATHER_DIR/INDIA/WORKFLOW"
DATA_ROOT="$WEATHER_DIR/INDIA/Data"
GFS_ROOT="$WEATHER_DIR/INDIA/GFS-DATA"
DOWNLOADER="$WORKFLOW_DIR/download_gfs_auto.py"

MPI_NP=16
N_DOMAINS=1      # d01=27km
FCST_HOURS=72    # WRF runs exactly this long
WPS_EXTRA=3      # WPS covers FCST_HOURS + WPS_EXTRA to ensure enough boundary records


# =============================================================================
# 2. LOGGING
# =============================================================================
mkdir -p "$DATA_ROOT"
PIPELINE_LOG="$DATA_ROOT/pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    local level="$1"; shift
    echo "[$(date -u +"%Y-%m-%d %H:%M:%S UTC")] [$level] $*" | tee -a "$PIPELINE_LOG"
}

die() {
    log ERROR "$*"
    log ERROR "Pipeline FAILED. Check $PIPELINE_LOG"
    exit 1
}

log INFO "========================================================"
log INFO "WRF PIPELINE STARTED"
log INFO "Log: $PIPELINE_LOG"


# =============================================================================
# 3. CONDA
# =============================================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ncl_stable

# Force MPICH
export PATH=/home/ras_08/Models/WRF_TUTORIAL/Libs/MPICH/bin:$PATH
export LD_LIBRARY_PATH=/home/ras_08/Models/WRF_TUTORIAL/Libs/MPICH/lib:$LD_LIBRARY_PATH

# Debug
which mpiexec | tee -a "$PIPELINE_LOG"

export NCARG_ROOT="$CONDA_PREFIX"
log INFO "Conda env: $CONDA_DEFAULT_ENV"


# =============================================================================
# 4. DATE / TIME SETUP
# =============================================================================
[[ -n "$FORCE_DATE" ]] && CYCLE_DATE="$FORCE_DATE" || CYCLE_DATE=$(date -u +%Y%m%d)
CYCLE_HR="$CYCLE"

# Helper — convert YYYYMMDD + HH to epoch
to_epoch() { date -u -d "${1:0:4}-${1:4:2}-${1:6:2} ${2}:00:00" +%s; }

# Helper — epoch to namelist datetime string YYYY-MM-DD_HH:00:00
epoch_to_wps() { date -u -d "@$1" +"%Y-%m-%d_%H:00:00"; }

# Helpers — extract fields from YYYY-MM-DD_HH:00:00
wps_year()  { echo "${1:0:4}"; }
wps_month() { echo "${1:5:2}"; }
wps_day()   { echo "${1:8:2}"; }
wps_hour()  { echo "${1:11:2}"; }

START_EPOCH=$(to_epoch "$CYCLE_DATE" "$CYCLE_HR")
WRF_END_EPOCH=$(( START_EPOCH + FCST_HOURS * 3600 ))
WPS_END_EPOCH=$(( START_EPOCH + (FCST_HOURS + WPS_EXTRA) * 3600 ))

# WPS end — includes +3h buffer (used in namelist.wps end_date)
WPS_END=$(epoch_to_wps "$WPS_END_EPOCH")   # e.g. 2026-06-19_03:00:00

# WRF end — exact forecast end, no buffer (used in namelist.input end_*)
WRF_END=$(epoch_to_wps "$WRF_END_EPOCH")   # e.g. 2026-06-19_00:00:00

# WRF start (same as WPS start)
START_WPS=$(epoch_to_wps "$START_EPOCH")   # e.g. 2026-06-18_00:00:00

# Integer fields for namelist.input
START_YEAR=$(wps_year  "$START_WPS")
START_MON=$(wps_month  "$START_WPS")
START_DAY=$(wps_day    "$START_WPS")
START_HOUR=$(wps_hour  "$START_WPS")

END_YEAR=$(wps_year    "$WRF_END")
END_MON=$(wps_month    "$WRF_END")
END_DAY=$(wps_day      "$WRF_END")
END_HOUR=$(wps_hour    "$WRF_END")

log INFO "Cycle      : ${CYCLE_DATE} ${CYCLE_HR}z"
log INFO "WRF start  : $START_WPS"
log INFO "WRF end    : $WRF_END  (${FCST_HOURS}h forecast)"
log INFO "WPS end    : $WPS_END  (+${WPS_EXTRA}h boundary buffer)"


# =============================================================================
# 5. GFS DOWNLOAD
# =============================================================================
log INFO "--- STEP 1/5: GFS Download ---"

GFS_FULL_PATH="$GFS_ROOT/${CYCLE_DATE}/${CYCLE_HR}z"
mkdir -p "$GFS_FULL_PATH"
log INFO "GFS path: $GFS_FULL_PATH"

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    python "$DOWNLOADER" \
        --date   "$CYCLE_DATE" \
        --cycle  "$CYCLE_HR"  \
        --outdir "$GFS_FULL_PATH" \
        || die "GFS download failed."
else
    N_GRIB=$(ls "$GFS_FULL_PATH"/*.grib2 2>/dev/null | wc -l)
    [[ "$N_GRIB" -gt 0 ]] || die "No .grib2 files in $GFS_FULL_PATH"
    log INFO "--skip-download: found $N_GRIB GRIB2 files"
fi


# =============================================================================
# 6. WPS
# =============================================================================
log INFO "--- STEP 2/5: WPS ---"
cd "$WPS_DIR" || die "Cannot cd to $WPS_DIR"

# Build comma-separated quoted date list for N domains
# e.g. '2026-06-18_00:00:00','2026-06-18_00:00:00'
date_list() {
    local dt="$1" n="$2" result="" i
    for ((i=1; i<=n; i++)); do
        [[ $i -gt 1 ]] && result+=","
        result+="'${dt}'"
    done
    echo "$result"
}

START_LIST=$(date_list "$START_WPS" "$N_DOMAINS")
END_LIST=$(date_list   "$WPS_END"   "$N_DOMAINS")

log INFO "Patching namelist.wps ..."
sed -i "s/^ *start_date.*/\ start_date = ${START_LIST},/" namelist.wps
sed -i "s/^ *end_date.*/\ end_date   = ${END_LIST},/"   namelist.wps
log INFO "  start_date = $START_LIST"
log INFO "  end_date   = $END_LIST"

# Geogrid — skip if geo_em already exists
if ls geo_em.d01.nc &>/dev/null; then
    log INFO "geo_em found — skipping geogrid.exe"
else
    log INFO "Running geogrid.exe ..."
    ./geogrid.exe > log.geogrid 2>&1 || die "geogrid.exe failed. Check $WPS_DIR/log.geogrid"
    grep -qi "Successful completion" log.geogrid || die "geogrid.exe: no success message"
    log INFO "geogrid.exe OK"
fi

log INFO "Cleaning previous WPS intermediates ..."
rm -f FILE:* GRIBFILE.* met_em.d0*.* ungrib.log metgrid.log

log INFO "Linking GRIB2 files ..."
csh ./link_grib.csh "$GFS_FULL_PATH"/gfs.t${CYCLE_HR}z.pgrb2.0p25.f*.grib2 \
    || die "link_grib.csh failed"

log INFO "Setting Vtable.GFS ..."
[[ -f "ungrib/Variable_Tables/Vtable.GFS" ]] \
    || die "Vtable.GFS not found in $WPS_DIR/ungrib/Variable_Tables/"
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable

log INFO "Running ungrib.exe ..."
./ungrib.exe > log.ungrib 2>&1 || die "ungrib.exe failed. Check $WPS_DIR/log.ungrib"
grep -qi "Successful completion" log.ungrib || die "ungrib.exe: no success message"
log INFO "ungrib.exe OK"

log INFO "Running metgrid.exe ..."
./metgrid.exe > log.metgrid 2>&1 || die "metgrid.exe failed. Check $WPS_DIR/log.metgrid"
grep -qi "Successful completion" log.metgrid || die "metgrid.exe: no success message"
log INFO "metgrid.exe OK"

for dom in $(seq 1 $N_DOMAINS); do
    dom_str=$(printf "%02d" "$dom")
    N_MET=$(ls met_em.d${dom_str}.*.nc 2>/dev/null | wc -l)
    [[ "$N_MET" -gt 0 ]] || die "No met_em files for domain d${dom_str}"
    log INFO "met_em d${dom_str}: $N_MET files"
done


# =============================================================================
# 7. WRF
# =============================================================================
log INFO "--- STEP 3/5: WRF ---"
cd "$WRF_RUN_DIR" || die "Cannot cd to $WRF_RUN_DIR"

log INFO "Cleaning previous WRF run files ..."
rm -f met_em.d0*.* wrfout_d0* wrfbdy_d0* wrfinput_d0* rsl.out.* rsl.error.*

log INFO "Linking met_em files ..."
for dom in $(seq 1 $N_DOMAINS); do
    dom_str=$(printf "%02d" "$dom")
    ln -sf "$WPS_DIR"/met_em.d${dom_str}.*.nc .
done

# Patch namelist.input
log INFO "Patching namelist.input ..."

rep() {
    local val="$1" n="$2" out="" i
    for ((i=1; i<=n; i++)); do
        [[ $i -gt 1 ]] && out+=", "
        out+="$val"
    done
    echo "$out"
}

sed -i "s/^ *start_year.*/\ start_year  = $(rep $START_YEAR  $N_DOMAINS),/" namelist.input
sed -i "s/^ *start_month.*/\ start_month = $(rep $START_MON   $N_DOMAINS),/" namelist.input
sed -i "s/^ *start_day.*/\ start_day   = $(rep $START_DAY   $N_DOMAINS),/" namelist.input
sed -i "s/^ *start_hour.*/\ start_hour  = $(rep $START_HOUR  $N_DOMAINS),/" namelist.input
sed -i "s/^ *end_year.*/\ end_year    = $(rep $END_YEAR    $N_DOMAINS),/" namelist.input
sed -i "s/^ *end_month.*/\ end_month   = $(rep $END_MON     $N_DOMAINS),/" namelist.input
sed -i "s/^ *end_day.*/\ end_day     = $(rep $END_DAY     $N_DOMAINS),/" namelist.input
sed -i "s/^ *end_hour.*/\ end_hour    = $(rep $END_HOUR    $N_DOMAINS),/" namelist.input
sed -i "s/^ *run_days.*/\ run_days    = 0,/" namelist.input
sed -i "s/^ *run_hours.*/\ run_hours   = ${FCST_HOURS},/" namelist.input
sed -i "s/^ *run_minutes.*/\ run_minutes = 0,/" namelist.input
sed -i "s/^ *run_seconds.*/\ run_seconds = 0,/" namelist.input

log INFO "  namelist.input start : ${START_YEAR}-${START_MON}-${START_DAY} ${START_HOUR}z"
log INFO "  namelist.input end   : ${END_YEAR}-${END_MON}-${END_DAY} ${END_HOUR}z"
log INFO "  run_hours            : $FCST_HOURS"

log INFO "Running real.exe (np=$MPI_NP) ..."
mpiexec -n "$MPI_NP" ./real.exe >> "$PIPELINE_LOG" 2>&1
grep -q "SUCCESS COMPLETE REAL_EM" rsl.error.0000 \
    || die "real.exe failed. Check $WRF_RUN_DIR/rsl.error.0000"
log INFO "real.exe OK"

log INFO "Verifying wrfbdy_d01 exists ..."
[[ -f wrfbdy_d01 ]] || die "wrfbdy_d01 not created by real.exe — check namelist.input end dates"
log INFO "wrfbdy_d01 OK"

log INFO "Running wrf.exe (np=$MPI_NP) — this will take a while ..."
mpiexec -n "$MPI_NP" ./wrf.exe >> "$PIPELINE_LOG" 2>&1
grep -q "SUCCESS COMPLETE WRF" rsl.error.0000 \
    || die "wrf.exe failed. Check $WRF_RUN_DIR/rsl.error.0000"
log INFO "wrf.exe OK"


# =============================================================================
# 8. VERIFY OUTPUT + BUILD WRFOUT MAP
# =============================================================================
log INFO "--- STEP 4/5: Verifying wrfout files ---"

DAILY_PLOT_DIR="$DATA_ROOT/${CYCLE_DATE}/${CYCLE_HR}z"
mkdir -p "$DAILY_PLOT_DIR"

WRFOUT_TIMESTAMP="${START_YEAR}-${START_MON}-${START_DAY}_${START_HOUR}:00:00"

declare -A WRFOUT
for dom in $(seq 1 $N_DOMAINS); do
    dom_str=$(printf "%02d" "$dom")
    WFILE="$WRF_RUN_DIR/wrfout_d${dom_str}_${WRFOUT_TIMESTAMP}"
    [[ -f "$WFILE" ]] || die "wrfout not found: $WFILE"
    WRFOUT[$dom_str]="$WFILE"
    log INFO "wrfout d${dom_str}: OK  ($WFILE)"
done


# =============================================================================
# 9. PLOTTING
# =============================================================================
log INFO "--- STEP 5/5: Plotting ---"
log INFO "Output directory: $DAILY_PLOT_DIR"

run_plot() {
    local label="$1" script="$2"; shift 2
    log INFO "  [$label] $script $*"
    python "$WORKFLOW_DIR/$script" "$@" >> "$PIPELINE_LOG" 2>&1 \
        && log INFO "  [$label] OK" \
        || log WARN "  [$label] $script returned non-zero — check log"
}

# --- d01: India-wide ---
run_plot "d01-rain"   plot_rainfall_bfs.py    --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
run_plot "d01-tmax"   plot_max_temp.py        --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
run_plot "d01-tmin"   plot_min_temp.py        --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
run_plot "d01-wind"   plot_wind.py            --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
run_plot "d01-skewt"  plot_skewT.py           --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"

# --- d01: Kerala ---
run_plot "d01-kerala" plot_kerala_rainfall.py --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"

# --- d01: Maharashtra ---
run_plot "d01-maharashtra" plot_maharashtra_rainfall.py --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"

# --- d01: Uttarakhand ---
run_plot "d01-uttarakhand" plot_uttarakhand_rainfall.py --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"

# --- d01: Northeast ---
run_plot "d01-northeast" plot_rainfall_northeast.py  --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"

# --- d01: Mumbai ---
run_plot "d01-mumbai" plot_mumbai_rainfall.py  --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
# --- d01: JK_LADAKH ---
run_plot "d01-JK_LADAKH" plot_jammu_rainfall.py  --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"

run_plot "d01-tmax_jammu"   plot_temp_jk.py        --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"

# =============================================================================
# 10. DONE
# =============================================================================
log INFO "========================================================"
log INFO "PIPELINE COMPLETE  [${CYCLE_DATE} ${CYCLE_HR}z]"
log INFO "Plots: $DAILY_PLOT_DIR"
log INFO "Log  : $PIPELINE_LOG"
log INFO "========================================================"
exit 0
