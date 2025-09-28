import subprocess
from pathlib import Path

COMP = "store-sales-time-series-forecasting"

def run(cmd: str):
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    run(f"kaggle competitions download -c {COMP} -p {data_dir}")
    # Unzip all zips in data/raw
    for z in data_dir.glob("*.zip"):
        run(f"unzip -o {z} -d {data_dir}")
        z.unlink()

    print("\nFiles in data/raw:")
    for p in sorted(data_dir.glob('*.csv')):
        print(" -", p.name)

if __name__ == "__main__":
    main()