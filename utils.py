import pandas as pd
import os

def open_light_curve_csv(filename, dir_light_curves):
    """ Helper function to open a light curve csv file.
    """
    file_path = os.path.join(dir_light_curves, filename)
    df = pd.read_csv(file_path, header=None, sep='\s+', index_col=None, names=['time','flux','flux_error'])
    df = df.dropna(subset=['flux'])
    return df