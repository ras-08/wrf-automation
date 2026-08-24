import requests
import argparse
from datetime import datetime
from pathlib import Path
import time

BASE_URL = "https://data.ecmwf.int/forecasts/"

FORECAST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21, 24]


def download_ecmwf_india(date_str, output_path, cycle="00", forecast_hours=FORECAST_HOURS):
    """
    Downloads ECMWF Open Data forecast files (full global GRIB2).
    """
    save_dir = Path(output_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    downloaded, failed = [], []
    session = requests.Session()

    print(f"\n{'='*55}")
    print(f"  ECMWF Download  |  Date: {date_str}  Cycle: {cycle}z")
    print(f"  Output: {save_dir}")
    print(f"{'='*55}\n")

    for fhr in forecast_hours:
        fname    = f"{date_str}{cycle}0000-{fhr}h-oper-fc.grib2"
        url      = f"{BASE_URL}/{date_str}/{cycle}z/ifs/0p25/oper/{fname}"
        out_file = save_dir / fname

        if out_file.exists():
            print(f"  [SKIP] Already exists: {fname}")
            downloaded.append(str(out_file))
            continue

        print(f"  [GET]  {fname}")

        try:
            with session.get(url, stream=True, timeout=600) as resp:
                if resp.status_code == 200:
                    with open(out_file, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                            f.write(chunk)
                    size_mb = out_file.stat().st_size / (1024 * 1024)
                    print(f"  [OK]   {size_mb:.1f} MB → {fname}")
                    downloaded.append(str(out_file))

                elif resp.status_code == 429:
                    print(f"  [WAIT] Rate limited — waiting 60s...")
                    time.sleep(60)
                    failed.append(fname)

                else:
                    print(f"  [ERR]  HTTP {resp.status_code}: {fname}")
                    failed.append(fname)

        except Exception as e:
            print(f"  [ERR]  {e}")
            failed.append(fname)

        time.sleep(5)

    session.close()

    print(f"\n{'='*55}")
    print(f"  Downloaded : {len(downloaded)}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        print(f"  Failed files: {failed}")
    print(f"{'='*55}\n")

    return len(downloaded) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ECMWF Open Data for India WRF runs.")
    parser.add_argument("--date",  type=str,
                        default=datetime.utcnow().strftime("%Y%m%d"),
                        help="Date in YYYYMMDD format")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directory to save downloaded GRIB2 files")
    parser.add_argument("--cycle", type=str, default="00",
                        choices=["00", "12"],
                        help="Cycle: 00 or 12 (ECMWF only has 2 cycles)")
    parser.add_argument("--hours", nargs="+", type=int,
                        default=FORECAST_HOURS,
                        help="Forecast hours e.g. --hours 0 6 12 24")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        raise ValueError(f"Invalid date: '{args.date}'. Use YYYYMMDD.")

    success = download_ecmwf_india(
        date_str=args.date,
        output_path=args.outdir,
        cycle=args.cycle,
        forecast_hours=args.hours,
    )
    exit(0 if success else 1)
