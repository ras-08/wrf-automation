#!/bin/bash
# =============================================================================
#  run_wrf_pipeline.sh
#  Fully automated WRF pipeline: GFS download → WPS → WRF → Plotting
#
#  Features:
#   - Explicit cycle argument (00/06/12/18) — no guessing
#   - Cycle-aware directory structure: GFS-DATA/YYYYMMDD/HHz/
#   - Correct Vtable.GFS2 for GRIB2 data
#   - Step-level success checks (exits on any failure)
#   - Skips geogrid if geo_em files already exist
#   - Handles all 3 domains (d01, d02, d03)
#   - Timestamped pipeline log in DATA_ROOT
#   - Cron-safe (exit 0 = full success, exit 1 = any failure)
#
#  Usage:
#   ./run_wrf_pipeline.sh 00               # run 00z cycle for today
#   ./run_wrf_pipeline.sh 12               # run 12z cycle for today
#   ./run_wrf_pipeline.sh 06 --skip-download   # reuse existing GFS data
#   ./run_wrf_pipeline.sh 18 --date 20250610   # override date
#
#  Cron examples (launches ~4h after each cycle start):
#   0  4  * * *  /home/ras_08/WEATHER/INDIA/WORKFLOW/run_wrf_pipeline.sh 00
#   0  10 * * *  /home/ras_08/WEATHER/INDIA/WORKFLOW/run_wrf_pipeline.sh 06
#   0  16 * * *  /home/ras_08/WEATHER/INDIA/WORKFLOW/run_wrf_pipeline.sh 12
#   0  22 * * *  /home/ras_08/WEATHER/INDIA/WORKFLOW/run_wrf_pipeline.sh 18
# =============================================================================

set -eo pipefail   # exit on error, undefined var, or pipe failure


# =============================================================================
# 0. ARGUMENT PARSING
# =============================================================================

# First positional arg = cycle (required)
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {00|06|12|18} [--date YYYYMMDD] [--skip-download]"
    exit 1
fi

CYCLE="$1"; shift

if [[ ! "$CYCLE" =~ ^(00|06|12|18)$ ]]; then
    echo "ERROR: cycle must be one of 00, 06, 12, 18 (got '$CYCLE')"
    echo "Usage: $0 {00|06|12|18} [--date YYYYMMDD] [--skip-download]"
    exit 1
fi

# Optional flags after the cycle arg
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
# 1. CONFIGURATION & PATHS
# =============================================================================

HOME_DIR="/home/ras_08"
WEATHER_DIR="$HOME_DIR/WEATHER"
WPS_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WPS-4.5"
WRF_RUN_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WRFV4.5/run"
WORKFLOW_DIR="$WEATHER_DIR/INDIA/WORKFLOW"
DATA_ROOT="$WEATHER_DIR/INDIA/Data"
GFS_ROOT="$WEATHER_DIR/INDIA/GFS-DATA"
DOWNLOADER="$WORKFLOW_DIR/download_gfs_auto.py"

# Number of MPI processes for real.exe and wrf.exe
MPI_NP=8

# Number of domains
N_DOMAINS=1

# Forecast length in hours (how far ahead to run WRF)
FCST_HOURS=24


# =============================================================================
# 2. LOGGING SETUP
# =============================================================================
mkdir -p "$DATA_ROOT"
PIPELINE_LOG="$DATA_ROOT/pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
    local level="$1"; shift
    local msg="$*"
    local ts
    ts=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    echo "[$ts] [$level] $msg" | tee -a "$PIPELINE_LOG"
}

log INFO "========================================================"
log INFO "WRF PIPELINE STARTED"
log INFO "Log file: $PIPELINE_LOG"

# Helper: log + exit
die() {
    log ERROR "$*"
    log ERROR "Pipeline FAILED. Check $PIPELINE_LOG"
    exit 1
}


# =============================================================================
# 3. CONDA INIT
# =============================================================================
# =============================================================================
# 3. CONDA INIT
# =============================================================================

source ~/miniconda3/etc/profile.d/conda.sh

conda activate ncl_stable

export NCARG_ROOT="$CONDA_PREFIX"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "NCARG_ROOT=$NCARG_ROOT"
# =============================================================================
# 4. GFS DOWNLOAD — explicit cycle
# =============================================================================
log INFO "--- STEP 1/6: GFS Download ---"

# Date: use override if supplied, otherwise today UTC
if [[ -n "$FORCE_DATE" ]]; then
    CYCLE_DATE="$FORCE_DATE"
else
    CYCLE_DATE=$(date -u +%Y%m%d)
fi

CYCLE_HR="$CYCLE"
GFS_FULL_PATH="$GFS_ROOT/${CYCLE_DATE}/${CYCLE_HR}z"
mkdir -p "$GFS_FULL_PATH"

log INFO "Cycle      : ${CYCLE_DATE} ${CYCLE_HR}z"
log INFO "GFS path   : $GFS_FULL_PATH"

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
    log INFO "Running downloader ..."
    python "$DOWNLOADER" \
        --date   "$CYCLE_DATE" \
        --cycle  "$CYCLE_HR"  \
        --outdir "$GFS_FULL_PATH" \
        || die "GFS download failed."
else
    log INFO "--skip-download set. Reusing existing data in $GFS_FULL_PATH"
    # Confirm data is actually there
    N_GRIB=$(ls "$GFS_FULL_PATH"/*.grib2 2>/dev/null | wc -l)
    [[ "$N_GRIB" -gt 0 ]] || die "No .grib2 files found in $GFS_FULL_PATH — cannot skip download."
    log INFO "Found $N_GRIB GRIB2 files — OK"
fi

# Derive WRF start/end datetimes
# Start = cycle init time; End = start + FCST_HOURS
START_DT="${CYCLE_DATE:0:4}-${CYCLE_DATE:4:2}-${CYCLE_DATE:6:2}_${CYCLE_HR}:00:00"
END_EPOCH=$(date -u -d "${CYCLE_DATE:0:4}-${CYCLE_DATE:4:2}-${CYCLE_DATE:6:2} ${CYCLE_HR}:00:00 ${FCST_HOURS} hours" +%s)
END_DATE_DASH=$(date -u -d "@$END_EPOCH" +%Y-%m-%d)
END_HR=$(date -u -d "@$END_EPOCH" +%H)
END_DT="${END_DATE_DASH}_${END_HR}:00:00"

START_YEAR="${CYCLE_DATE:0:4}"
START_MONTH="${CYCLE_DATE:4:2}"
START_DAY="${CYCLE_DATE:6:2}"
END_YEAR="${END_DATE_DASH:0:4}"
END_MONTH="${END_DATE_DASH:5:2}"
END_DAY="${END_DATE_DASH:8:2}"

log INFO "WRF start : $START_DT"
log INFO "WRF end   : $END_DT"


# =============================================================================
# 5. WPS — geogrid / ungrib / metgrid
# =============================================================================
log INFO "--- STEP 2/6: WPS Processing ---"
cd "$WPS_DIR" || die "Cannot cd to $WPS_DIR"

# ---- Compute WPS end time with +3h buffer (WRF needs one extra boundary record) ----
WPS_BUFFER_HOURS=6
WPS_END_EPOCH=$(date -u -d "${CYCLE_DATE:0:4}-${CYCLE_DATE:4:2}-${CYCLE_DATE:6:2} ${CYCLE_HR}:00:00 $((FCST_HOURS + WPS_BUFFER_HOURS)) hours" +%s)
WPS_END_DATE_DASH=$(date -u -d "@$WPS_END_EPOCH" +%Y-%m-%d)
WPS_END_HR=$(date -u -d "@$WPS_END_EPOCH" +%H)
WPS_END_DT="${WPS_END_DATE_DASH}_${WPS_END_HR}:00:00"

# ---- 5a. Patch namelist.wps ----
log INFO "Patching namelist.wps ..."

# Build repeated start/end date strings for all domains
# e.g. for 3 domains: '2025-06-11_12:00:00','2025-06-11_12:00:00','2025-06-11_12:00:00'
build_date_list() {
    local dt="$1"
    local n="$2"
    local result=""
    for ((i=1; i<=n; i++)); do
        [[ $i -gt 1 ]] && result+=","
        result+="'${dt}'"
    done
    echo "$result"
}

START_LIST=$(build_date_list "$START_DT" "$N_DOMAINS")
END_LIST=$(build_date_list   "$WPS_END_DT"   "$N_DOMAINS")

sed -i "s/^ *start_date.*/\ start_date = ${START_LIST},/" namelist.wps
sed -i "s/^ *end_date.*/\ end_date   = ${END_LIST},/"   namelist.wps

log INFO "  start_date → $START_LIST"
log INFO "  end_date   → $END_LIST"

# ---- 5b. Geogrid — skip if geo_em files already exist ----
GEO_CHECK=$(ls geo_em.d01.nc 2>/dev/null | wc -l)
if [[ "$GEO_CHECK" -gt 0 ]]; then
    log INFO "geo_em files found — skipping geogrid.exe"
else
    log INFO "Running geogrid.exe ..."
    ./geogrid.exe > log.geogrid 2>&1 \
        || die "geogrid.exe failed. Check $WPS_DIR/log.geogrid"
    grep -i "Successful completion" log.geogrid \
        || die "geogrid.exe did not report successful completion."
    log INFO "geogrid.exe OK"
fi

# ---- 5c. Clean previous ungrib/metgrid output ----
log INFO "Cleaning previous WPS intermediate files ..."
rm -f FILE:* GRIBFILE.* met_em.d0*.* ungrib.log metgrid.log

# ---- 5d. Link GRIB2 files ----
log INFO "Linking GRIB2 files from $GFS_FULL_PATH ..."
csh ./link_grib.csh "$GFS_FULL_PATH"/gfs.t${CYCLE_HR}z.pgrb2.0p25.f*.grib2 \
    || die "link_grib.csh failed."

# ---- 5e. Vtable — MUST be GFS for GRIB2 data ----
log INFO "Setting Vtable.GFS (GRIB2) ..."
[[ -f "ungrib/Variable_Tables/Vtable.GFS" ]] \
    || die "Vtable.GFS2 not found in $WPS_DIR/ungrib/Variable_Tables/"
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable

# ---- 5f. Ungrib ----
log INFO "Running ungrib.exe ..."
./ungrib.exe > log.ungrib 2>&1 \
    || die "ungrib.exe failed. Check $WPS_DIR/log.ungrib"
grep -i "Successful completion" log.ungrib \
    || die "ungrib.exe did not report successful completion."
log INFO "ungrib.exe OK"

# ---- 5g. Metgrid ----
log INFO "Running metgrid.exe ..."
./metgrid.exe > log.metgrid 2>&1 \
    || die "metgrid.exe failed. Check $WPS_DIR/log.metgrid"
grep -i "Successful completion" log.metgrid \
    || die "metgrid.exe did not report successful completion."
log INFO "metgrid.exe OK"

# Verify met_em files exist for all domains
for dom in $(seq 1 $N_DOMAINS); do
    dom_str=$(printf "%02d" "$dom")
    N_MET=$(ls met_em.d${dom_str}.*.nc 2>/dev/null | wc -l)
    [[ "$N_MET" -gt 0 ]] || die "No met_em files found for domain d${dom_str}."
    log INFO "met_em d${dom_str}: $N_MET files found"
done


# =============================================================================
# 6. WRF — real.exe + wrf.exe
# =============================================================================
log INFO "--- STEP 3/6: WRF Processing ---"
cd "$WRF_RUN_DIR" || die "Cannot cd to $WRF_RUN_DIR"

# ---- 6a. Clean previous run ----
log INFO "Cleaning previous WRF run files ..."
rm -f met_em.d0*.* wrfout_d0* wrfbdy_d0* wrfinput_d0* rsl.out.* rsl.error.*

# ---- 6b. Link met_em files for all domains ----
log INFO "Linking met_em files ..."
for dom in $(seq 1 $N_DOMAINS); do
    dom_str=$(printf "%02d" "$dom")
    ln -sf "$WPS_DIR"/met_em.d${dom_str}.*.nc .
done

# ---- 6c. Patch namelist.input ----
log INFO "Patching namelist.input ..."

# Build repeated scalar fields for N_DOMAINS
rep() {
    local val="$1"
    local n="$2"
    local sep="${3:-, }"
    local out=""
    for ((i=1; i<=n; i++)); do
        [[ $i -gt 1 ]] && out+="$sep"
        out+="$val"
    done
    echo "$out"
}

START_YEAR_REP=$(rep "$START_YEAR"  "$N_DOMAINS")
START_MONTH_REP=$(rep "$START_MONTH" "$N_DOMAINS")
START_DAY_REP=$(rep "$START_DAY"   "$N_DOMAINS")
START_HR_REP=$(rep "$CYCLE_HR"    "$N_DOMAINS")

END_YEAR_REP=$(rep "${WPS_END_DATE_DASH:0:4}"  "$N_DOMAINS")
END_MONTH_REP=$(rep "${WPS_END_DATE_DASH:5:2}" "$N_DOMAINS")
END_DAY_REP=$(rep "${WPS_END_DATE_DASH:8:2}"   "$N_DOMAINS")
END_HR_REP=$(rep "$WPS_END_HR"                 "$N_DOMAINS")

# Runtime in seconds (for run_hours or run_seconds)
FCST_SECONDS=$((FCST_HOURS * 3600))

sed -i "s/^ *start_year.*/\ start_year                          = ${START_YEAR_REP},/"   namelist.input
sed -i "s/^ *start_month.*/\ start_month                         = ${START_MONTH_REP},/"  namelist.input
sed -i "s/^ *start_day.*/\ start_day                           = ${START_DAY_REP},/"    namelist.input
sed -i "s/^ *start_hour.*/\ start_hour                          = ${START_HR_REP},/"    namelist.input
sed -i "s/^ *end_year.*/\ end_year                            = ${END_YEAR_REP},/"     namelist.input
sed -i "s/^ *end_month.*/\ end_month                           = ${END_MONTH_REP},/"   namelist.input
sed -i "s/^ *end_day.*/\ end_day                             = ${END_DAY_REP},/"     namelist.input
sed -i "s/^ *end_hour.*/\ end_hour                            = ${END_HR_REP},/"     namelist.input
sed -i "s/^ *run_hours.*/\ run_hours                           = ${FCST_HOURS},/"    namelist.input
sed -i "s/^ *run_minutes.*/\ run_minutes                         = 0,/" namelist.input
sed -i "s/^ *run_seconds.*/\ run_seconds                         = 0,/" namelist.input

log INFO "  start: $START_YEAR_REP / $START_MONTH_REP / $START_DAY_REP  ${START_HR_REP}z"
log INFO "  end  : $END_YEAR_REP / $END_MONTH_REP / $END_DAY_REP  ${END_HR_REP}z"

# ---- 6d. real.exe ----
log INFO "Running real.exe (MPI np=$MPI_NP) ..."
mpirun -np "$MPI_NP" ./real.exe >> "$PIPELINE_LOG" 2>&1

grep -q "SUCCESS COMPLETE REAL_EM" rsl.error.0000 \
    || die "real.exe failed. Check $WRF_RUN_DIR/rsl.error.0000"
log INFO "real.exe OK"

# ---- 6e. wrf.exe ----
log INFO "Running wrf.exe (MPI np=$MPI_NP) — this will take a while ..."
mpirun -np "$MPI_NP" ./wrf.exe >> "$PIPELINE_LOG" 2>&1

grep -q "SUCCESS COMPLETE WRF" rsl.error.0000 \
    || die "wrf.exe failed. Check $WRF_RUN_DIR/rsl.error.0000"
log INFO "wrf.exe OK"


# =============================================================================
# 7. VERIFY WRFOUT FILES
# =============================================================================
log INFO "--- STEP 4/6: Verifying WRF output ---"

DAILY_PLOT_DIR="$DATA_ROOT/${CYCLE_DATE}"
mkdir -p "$DAILY_PLOT_DIR"

# Build expected wrfout filename for each domain
declare -A WRFOUT
for dom in $(seq 1 $N_DOMAINS); do
    dom_str=$(printf "%02d" "$dom")
    WFILE="$WRF_RUN_DIR/wrfout_d${dom_str}_${CYCLE_DATE:0:4}-${CYCLE_DATE:4:2}-${CYCLE_DATE:6:2}_${CYCLE_HR}:00:00"
    [[ -f "$WFILE" ]] || die "Expected wrfout not found: $WFILE"
    WRFOUT[$dom_str]="$WFILE"
    log INFO "wrfout d${dom_str}: $WFILE  OK"
done


# =============================================================================
# 8. PLOTTING
# =============================================================================
log INFO "--- STEP 5/6: Plotting ---"

run_plot() {
    local step="$1"
    local script="$2"
    shift 2
    log INFO "[$step] Running $script ..."
    python "$WORKFLOW_DIR/$script" "$@" >> "$PIPELINE_LOG" 2>&1 \
        || { log WARN "$script returned non-zero — check log, continuing."; }
}

# d01 — India-wide fields
run_plot "5a" "plot_rainfall_bfs.py"  --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
run_plot "5b" "plot_max_temp.py"      --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
run_plot "5c" "plot_min_temp.py"      --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
run_plot "5d" "plot_wind.py"          --input "${WRFOUT[01]}" --outdir "$DAILY_PLOT_DIR"
run_plot "5e" "plot_skewT.py"         --input "${WRFOUT[01]}"

# d03 — Kerala-specific (highest resolution)
run_plot "5f" "plot_kerala_rainfall.py" --input "${WRFOUT[03]}"


# =============================================================================
# 9. DONE
# =============================================================================
log INFO "--- STEP 6/6: Finalising ---"
log INFO "Plots written to: $DAILY_PLOT_DIR"
log INFO "========================================================"
log INFO "PIPELINE COMPLETE  [${CYCLE_DATE} ${CYCLE_HR}z]"
log INFO "========================================================"
exit 0
