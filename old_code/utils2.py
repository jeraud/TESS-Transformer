import os
import pandas as pd
import numpy as np
from quality import TESSQualityFlags
from astropy.units import cds
from astropy.io import fits
import lightkurve as lk

def open_light_curve_csv(filename, dir_light_curves):
    """ Helper function to open a light curve csv file.
    """
    file_path = os.path.join(dir_light_curves, filename)
    df = pd.read_csv(file_path, header=None, sep='\s+', index_col=None, names=['time','flux','flux_error'])
    df = df.dropna(subset=['flux'])
    df['flux'] /= np.max(np.abs(df['flux']))
    #(flux[i] - np.median(flux[i])) / np.std(flux[i])
    
    return df

def load_lightcurve(fname, starid=None, exclude_bad_data=True):
    """
    Load light curve from file.

    Parameters:
        fname (str): Path to file to be loaded.
        starid (int): Star identifier (TIC/KIC/EPIC number) to be added to lightcurve object.
            This is only used for file types where the number can not be determined from the
            file itself.
        exclude_bad_data (bool): Exclude data based on quality flags.

    Returns:
        :class:`lightkurve.LightCurve`: Lightcurve object.

    Raises:
        ValueError: On invalid file format.

    .. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
    """

    if fname.endswith(('.txt', '.noisy', '.sysnoise', '.clean')):
        data = np.loadtxt(fname)
        if data.shape[1] == 4:
            quality = np.asarray(data[:,3], dtype='int32')
        else:
            quality = np.zeros(data.shape[0], dtype='int32')

        lightcurve = lk.TessLightCurve(
            time=data[:,0],
            flux=data[:,1],
            flux_err=data[:,2],
            flux_unit=cds.ppm,
            quality=quality,
            time_format='jd',
            time_scale='tdb',
            targetid=starid,
            quality_bitmask=TESSQualityFlags.DEFAULT_BITMASK,
            meta={}
        )

    elif fname.endswith(('.fits.gz', '.fits')):
        with fits.open(fname, mode='readonly', memmap=True) as hdu:
            telescope = hdu[0].header.get('TELESCOP')
            if telescope == 'TESS' and hdu[0].header.get('ORIGIN') == 'MIT/QLP':
                lightcurve = lk.TessLightCurve(
                    time=hdu['LIGHTCURVE'].data['TIME'],
                    flux=hdu['LIGHTCURVE'].data['SAP_FLUX'],
                    flux_err=hdu['LIGHTCURVE'].data['KSPSAP_FLUX_ERR'],
                    quality=np.asarray(hdu['LIGHTCURVE'].data['QUALITY'], dtype='int32'),
                    cadenceno=np.asarray(hdu['LIGHTCURVE'].data['CADENCENO'], dtype='int32'),
                    time_format='btjd',
                    time_scale='tdb',
                    targetid=hdu[0].header.get('TICID'),
                    label=hdu[0].header.get('OBJECT'),
                    camera=hdu[0].header.get('CAMERA'),
                    ccd=hdu[0].header.get('CCD'),
                    sector=hdu[0].header.get('SECTOR'),
                    ra=hdu[0].header.get('RA_OBJ'),
                    dec=hdu[0].header.get('DEC_OBJ'),
                    quality_bitmask=TESSQualityFlags.DEFAULT_BITMASK,
                    meta={}
                )
                lightcurve = 1e6 * (lightcurve.normalize() - 1)
                #lightcurve.flux.unit = cds.ppm        
            elif telescope == 'TESS':
                lightcurve = lk.TessLightCurveFile(hdu,
                    quality_bitmask=TESSQualityFlags.DEFAULT_BITMASK, flux_column='pdcsap_flux')
                lightcurve = 1e6 * (lightcurve.normalize() - 1)
                #lightcurve.flux.unit = cds.ppm
            elif telescope == 'Kepler':
                lightcurve = lk.KeplerLightCurveFile(hdu,
                    quality_bitmask=lk.utils.KeplerQualityFlags.DEFAULT_BITMASK).PDCSAP_FLUX
                lightcurve = 1e6 * (lightcurve.normalize() - 1)
                #lightcurve.flux.unit = cds.ppm
            else:
                raise ValueError("Could not determine FITS lightcurve type")
    else:
        raise ValueError("Invalid file format")
    
    #TODO: alternatively, use https://docs.lightkurve.org/reference/api/lightkurve.io.qlp.read_qlp_lightcurve.html for defaults

    # Exclude bad data points based on quality flags:
    if exclude_bad_data:
        # The actual Quality Flags class used doesn't matter, as long as it derives
        # from BaseQualityFlags.
        indx = TESSQualityFlags.filter(lightcurve.quality, flags=lightcurve.quality_bitmask)
        lightcurve = lightcurve[indx]

    lightcurve = lightcurve.remove_nans()
    return lightcurve