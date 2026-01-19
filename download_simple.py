
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List
from astroquery.mast import Observations

TEST_NAME = 'tessv4_test'
# Path to where the data will be downloaded (as raw FITS files)
RAW_DATA_PATH = f'/home/altair/Documents/UROP/2025_Summer/satsvac/data/lightcurves_raw/{TEST_NAME}'
# Path to the catalog of the data to download (containing the TIC IDs to download)
CATALOG_PATH = RAW_DATA_PATH + '/catalog.csv'
EXPTIME = 1800 # in seconds -- 1800 seconds = 30 minutes

def convert_to_native_byte_order(df):
    for col in df.columns:
        col_data = df[col]
        if hasattr(col_data.values, 'dtype') and hasattr(col_data.values.dtype, 'byteorder'):
            if col_data.values.dtype.byteorder == '>':
                try:    
                    df[col] = col_data.values.byteswap().newbyteorder()
                except AttributeError:
                    # Fix for NumPy 2.0: use .view(dtype.newbyteorder()) instead of .newbyteorder()
                    df[col] = col_data.values.byteswap().view(col_data.values.dtype.newbyteorder('='))
    return df

def download_qlp_data_from_catalog(
    catalog_path: Path,
    save_dir: Path,
    num_lightcurves: int = 0
):
    """
    Download QLP data for TIC IDs matched via GAIA IDs between an input catalog and a TESS reference catalog.

    Parameters
    ----------
    catalog_path : Path
        Path to the input catalog CSV.
    save_dir : Path
        Directory to save downloaded QLP data.
    num_lightcurves : int, optional
        Number of lightcurves to process (default is 0, meaning all).

    Notes
    -----
    Uses pandas merge for concise matching of GAIA IDs to TIC IDs.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load input catalog and determine GAIA ID column
    catalog_df = pd.read_csv(catalog_path)
    if num_lightcurves > 0:
        catalog_df = catalog_df.head(num_lightcurves)
    
    # Convert both the input catalog and the TESS reference catalog DataFrames to native endianness
    # to avoid ValueError: Big-endian buffer not supported on little-endian compiler.
    catalog_df = convert_to_native_byte_order(catalog_df)
    
    # Print the columns of the merged DataFrame for documentation and debugging purposes
    print(f"Columns in catalog_df: {catalog_df.columns.tolist()}")
    tic_ids = catalog_df['tic'].astype(str).unique().tolist()
    
    # Download QLP data in batches of 500 TIC IDs at a time for efficiency and to avoid overloading the server.
    # This also helps avoid issues with too many IDs in a single query.
    batch_size = 200
    for i in tqdm(range(0, len(tic_ids), batch_size), desc="Downloading QLP data", unit="batch"):
        batch_dir = save_dir / f"batch_{i//batch_size + 1}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_tic_ids = tic_ids[i:i+batch_size]
        print(f"Downloading QLP data for TIC IDs batch {i//batch_size + 1}: {batch_tic_ids[:3]}... (total {len(batch_tic_ids)})")
        try:
            download_qlp_data(batch_dir, batch_tic_ids)
        except Exception as e:
            print(f"Error downloading QLP data for batch {i//batch_size + 1}: {e}")
            continue


def download_qlp_data(save_dir: Path, ticids_list: List[str]):

    obsTable = Observations.query_criteria(
            project='TESS',
            dataproduct_type='TIMESERIES',
            provenance_name='QLP',
            t_exptime=[EXPTIME - 1, EXPTIME + 1],
            target_name=ticids_list
        )

    print(f"Found {len(obsTable)} QLP products")
    # print(obsTable)
    
    data = Observations.get_product_list(obsTable)

    download_lc = Observations.download_products(data, download_dir=save_dir)


def main():
    catalog_path = Path(CATALOG_PATH)
    save_dir = Path(RAW_DATA_PATH)
    download_qlp_data_from_catalog(catalog_path=catalog_path, save_dir=save_dir)


if __name__ == "__main__":
    main()