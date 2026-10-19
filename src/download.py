"""Download the public Kaggle archive or explicitly import a user-supplied CSV."""
import argparse
from datetime import datetime, timezone
import io
from pathlib import Path
import shutil
import urllib.request
import zipfile
from .config import DATA_PATH, SOURCE
from .data_loader import load_data
from .utils import sha256, write_json


def acquire(destination: Path = DATA_PATH, manual: Path | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if manual:
        load_data(manual)
        if manual.resolve() != destination.resolve():
            shutil.copyfile(manual, destination)
        method = "manual CSV import; source attribution supplied by user"
    else:
        url = "https://www.kaggle.com/api/v1/datasets/download/saurabh00007/iriscsv"
        request = urllib.request.Request(url, headers={"User-Agent": "IrisIntelligence/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(10_000_001)
        if len(payload) > 10_000_000:
            raise ValueError("Unexpectedly large Kaggle archive.")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [n for n in archive.namelist() if Path(n).name.lower() == "iris.csv"]
            if len(members) != 1:
                raise ValueError("Expected exactly one Iris.csv in the Kaggle archive.")
            csv_bytes = archive.read(members[0])
        temporary = destination.with_suffix(".download.csv")
        try:
            temporary.write_bytes(csv_bytes)
            load_data(temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        method = "direct HTTPS download from official Kaggle dataset API"
    frame = load_data(destination)
    provenance = {"source": SOURCE, "method": method, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                  "sha256": sha256(destination), "filename": destination.name, "rows": len(frame)}
    write_json(destination.with_suffix(".provenance.json"), provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-csv", type=Path)
    args = parser.parse_args()
    try:
        print(acquire(manual=args.import_csv))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Dataset acquisition failed: {exc}\nDownload Iris.csv from {SOURCE}, then use --import-csv PATH. No fallback was substituted.\n")


if __name__ == "__main__":
    main()
