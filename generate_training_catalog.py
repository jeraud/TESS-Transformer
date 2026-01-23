from pathlib import Path
import pandas as pd
from astropy.io import fits

labels = pd.read_csv("catalog.csv")  # has columns: tic, source, class

fits_dir = Path("lightcurves_raw/tessv4_test")
fits_paths = list(fits_dir.rglob("*.fits")) + list(fits_dir.rglob("*.fits.gz"))

rows = []
progress = 0
percentage = -1
print("Processing fits files...")
for p in fits_paths:
    with fits.open(p) as hdul:
        tic = hdul[0].header.get("TICID") or hdul[0].header.get("TIC_ID") or hdul[0].header.get("TARGETID")
    if tic is not None:
        rows.append({"tic": int(tic), "path": str(p)})
    progress += 1
    new_percentage = int(progress / len(fits_paths) * 100)
    if new_percentage > percentage:
        percentage = new_percentage
        print(f"Processed {percentage}% of fits files")

paths = pd.DataFrame(rows)

train = labels.merge(paths, on="tic", how="inner")
train = train.rename(columns={"class": "label"})[["label", "path"]]
train.to_csv("train_catalog.csv", index=False)