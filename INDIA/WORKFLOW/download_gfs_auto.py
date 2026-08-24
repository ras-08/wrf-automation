#!/usr/bin/env python3
"""
download_gfs_auto.py
--------------------
Automated GFS downloader for India WRF runs.

Features:
  - Probes NOMADS for the most recently available cycle (18/12/06/00z)
  - Retries with exponential backoff if the cycle is still uploading
  - Lock file prevents re-downloading a completed cycle
  - Structured log file written alongside the data
  - Cron-safe exit codes (0 = success, 1 = failure)

Usage:
  # Fully automatic (cron mode)
  python download_gfs_auto.py --outdir /data/gfs

  # Override cycle/date manually
  python download_gfs_auto.py --outdir /data/gfs --date 20250611 --cycle 12

  # Ignore lock file and force re-download
  python download_gfs_auto.py --outdir /data/gfs --force
"""

import os
import sys
import time
import logging
import argparse
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL      = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
CYCLES        = [18, 12, 6, 0]          # preference order: newest first
UPLOAD_DELAY  = 4.0                      # hours after cycle time before data is typically ready
RETRY_WAITS   = [5 * 60, 15 * 60, 30 * 60]  # seconds between retries (5 min, 15 min, 30 min)
LOCK_FILENAME = ".last_cycle"
LOG_FILENAME  = "gfs_download.log"

# India bounding box
TOPLAT    = 55
BOTTOMLAT = -10
LEFTLON   = 40
RIGHTLON  = 120

# Default forecast hour
DEFAULT_FORECAST_HOURS = list(range(0, 75, 3))   # 0, 3, 6, ... 72


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("gfs_downloader")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S UTC")

    # File handler — always DEBUG level
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Cycle probing
# ---------------------------------------------------------------------------

def build_probe_url(date_str: str, cycle: int) -> tuple[str, dict]:
    """Return (url, params) for a lightweight f000 probe request."""
    cycle_str = f"{cycle:02d}"
    data_dir  = f"/gfs.{date_str}/{cycle_str}/atmos"
    file_name = f"gfs.t{cycle_str}z.pgrb2.0p25.f000"
    params = {
        "dir":       data_dir,
        "file":      file_name,
        "var_TMP":   "on",       # single variable — minimal payload
        "lev_2_m_above_ground": "on",
        "subregion": "",
        "toplat":    TOPLAT,
        "leftlon":   LEFTLON,
        "rightlon":  RIGHTLON,
        "bottomlat": BOTTOMLAT,
    }
    return BASE_URL, params


def probe_cycle(date_str: str, cycle: int, logger: logging.Logger,
                timeout: int = 27) -> bool:
    """
    Returns True if the f000 file for (date_str, cycle) is available on NOMADS.
    Uses a HEAD-like GET with a single variable to keep transfer tiny.
    """
    url, params = build_probe_url(date_str, cycle)
    logger.debug(f"Probing {date_str} cycle {cycle:02d}z ...")
    try:
        r = requests.get(url, params=params, timeout=timeout, stream=True)
        r.close()
        if r.status_code == 200:
            logger.debug(f"  -> Available (HTTP 200)")
            return True
        else:
            logger.debug(f"  -> Not available (HTTP {r.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        logger.debug(f"  -> Probe error: {e}")
        return False


def find_latest_cycle(logger: logging.Logger,
                      max_retries: int = 3) -> tuple[str, str] | None:
    """
    Walk through the last 24 hours of cycles, checking availability.
    For the single most recent candidate, retry with backoff in case it's
    still being uploaded.

    Returns (date_str, cycle_str) or None if nothing is found.
    """
    now_utc = datetime.now(timezone.utc)

    # Build ordered list of (date_str, cycle_int) for past 24h
    candidates = []
    for hours_back in range(0, 28):
        t = now_utc - timedelta(hours=hours_back)
        for cyc in CYCLES:
            cycle_time = t.replace(hour=cyc, minute=0, second=0, microsecond=0)
            # Only consider cycles that *should* be uploaded by now
            expected_ready = cycle_time + timedelta(hours=UPLOAD_DELAY)
            if expected_ready <= now_utc:
                candidates.append((cycle_time.strftime("%Y%m%d"), cyc))

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    if not ordered:
        logger.error("No valid cycle candidates found (clock issue?).")
        return None

    # The very latest candidate gets retries; the rest get a single probe
    top_date, top_cyc = ordered[0]
    logger.info(f"Most recent candidate: {top_date} {top_cyc:02d}z")

    for attempt in range(max_retries + 1):
        if probe_cycle(top_date, top_cyc, logger):
            logger.info(f"Cycle confirmed available: {top_date} {top_cyc:02d}z")
            return top_date, f"{top_cyc:02d}"
        if attempt < max_retries:
            wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
            logger.warning(
                f"Cycle {top_date} {top_cyc:02d}z not ready yet. "
                f"Retry {attempt + 1}/{max_retries} in {wait // 60} min ..."
            )
            time.sleep(wait)

    logger.warning(f"Cycle {top_date} {top_cyc:02d}z not available after retries. "
                   "Falling back to older cycles.")

    # Fall back through the rest of the candidates
    for date_str, cyc in ordered[1:]:
        if probe_cycle(date_str, cyc, logger):
            logger.info(f"Falling back to: {date_str} {cyc:02d}z")
            return date_str, f"{cyc:02d}"

    logger.error("No available cycle found in the last 24 hours.")
    return None


# ---------------------------------------------------------------------------
# Lock file helpers
# ---------------------------------------------------------------------------

def read_lock(lock_path: Path) -> str | None:
    """Returns the lock string 'YYYYMMDD_HHz' or None."""
    if lock_path.exists():
        return lock_path.read_text().strip()
    return None


def write_lock(lock_path: Path, date_str: str, cycle_str: str):
    lock_path.write_text(f"{date_str}_{cycle_str}z\n")


def lock_key(date_str: str, cycle_str: str) -> str:
    return f"{date_str}_{cycle_str}z"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_gfs_india(date_str: str, cycle_str: str, save_dir: Path,
                       forecast_hours: list[int],
                       logger: logging.Logger) -> bool:
    """
    Downloads GFS GRIB2 files for the India bounding box.
    Returns True if all files downloaded successfully.
    """
    data_dir = f"/gfs.{date_str}/{cycle_str}/atmos"
    downloaded, failed = [], []

    logger.info(f"Starting download: {date_str} {cycle_str}z  →  {save_dir}")
    logger.info(f"Forecast hours: {forecast_hours[0]}–{forecast_hours[-1]}h "
                f"({len(forecast_hours)} files)")

    for fhr in forecast_hours:
        fhr_str   = f"{fhr:03d}"
        file_name = f"gfs.t{cycle_str}z.pgrb2.0p25.f{fhr_str}"
        out_file  = save_dir / f"{file_name}.grib2"

        # Skip if already present and non-empty
        if out_file.exists() and out_file.stat().st_size > 0:
            logger.debug(f"Already exists, skipping: {out_file.name}")
            downloaded.append(str(out_file))
            continue

        params = {
            "dir":       data_dir,
            "file":      file_name,
            "all_var":   "on",
            "all_lev":   "on",
            "subregion": "",
            "toplat":    TOPLAT,
            "leftlon":   LEFTLON,
            "rightlon":  RIGHTLON,
            "bottomlat": BOTTOMLAT,
        }

        max_file_retries = 5
        for attempt in range(1, max_file_retries + 1):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=180)
                if resp.status_code == 200 and len(resp.content) > 0:
                    out_file.write_bytes(resp.content)
                    size_kb = out_file.stat().st_size // 1024
                    downloaded.append(str(out_file))
                    logger.info(f"  ✓  f{fhr_str}  ({size_kb} KB)")
                    break
                else:
                    logger.warning(f"  ✗  f{fhr_str}  HTTP {resp.status_code} (attempt {attempt}/{max_file_retries})")
            except Exception as e:
                logger.warning(f"  ✗  f{fhr_str}  Error: {e} (attempt {attempt}/{max_file_retries})")

            if attempt < max_file_retries:
                time.sleep(15)
            else:
                failed.append(file_name)

    logger.info(f"Done. Downloaded: {len(downloaded)}  Failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed files: {failed}")

    return len(failed) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated GFS downloader for India WRF runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Directory where GRIB2 files will be saved."
    )
    parser.add_argument(
        "--date", default=None,
        help="Override date (YYYYMMDD). Default: auto-detect."
    )
    parser.add_argument(
        "--cycle", default=None, choices=["00", "06", "12", "18"],
        help="Override cycle. Default: auto-detect."
    )
    parser.add_argument(
        "--fhr-end", type=int, default=75,
        help="Last forecast hour to download (default: 75)."
    )
    parser.add_argument(
        "--fhr-step", type=int, default=3,
        help="Forecast hour step (default: 3)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore lock file and re-download even if already done."
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries waiting for latest cycle (default: 3)."
    )
    args = parser.parse_args()

    # --- Setup directories and logging ---
    save_dir = Path(args.outdir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path  = save_dir / LOG_FILENAME
    lock_path = save_dir / LOCK_FILENAME
    logger    = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("GFS Auto-Downloader started")
    logger.info(f"Output dir : {save_dir}")
    logger.info(f"Log file   : {log_path}")

    # --- Determine cycle ---
    if args.date and args.cycle:
        # Fully manual override
        date_str  = args.date
        cycle_str = args.cycle
        logger.info(f"Manual override: {date_str} {cycle_str}z")
    else:
        result = find_latest_cycle(logger, max_retries=args.max_retries)
        if result is None:
            logger.error("Could not find an available cycle. Exiting.")
            sys.exit(1)
        date_str, cycle_str = result
        # Allow partial override
        if args.date:
            date_str = args.date
        if args.cycle:
            cycle_str = args.cycle

    # --- Lock file check ---
    current_key = lock_key(date_str, cycle_str)
    existing    = read_lock(lock_path)

    if existing == current_key and not args.force:
        logger.info(f"Cycle {current_key} already downloaded (lock file found). "
                    "Use --force to re-download. Exiting.")
        sys.exit(0)

    # --- Download ---
    forecast_hours = list(range(0, args.fhr_end + 1, args.fhr_step))
    success = download_gfs_india(
        date_str=date_str,
        cycle_str=cycle_str,
        save_dir=save_dir,
        forecast_hours=forecast_hours,
        logger=logger,
    )

    # --- Write lock and exit ---
    if success:
        write_lock(lock_path, date_str, cycle_str)
        logger.info(f"Lock file written: {current_key}")
        logger.info("All done. Exiting with code 0.")
        sys.exit(0)
    else:
        logger.error("Some files failed to download. Exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
