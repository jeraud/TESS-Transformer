import numpy as np
from astropy.io import fits

def extract_qlp(curve_path):
    with fits.open(curve_path, mode="readonly") as hdulist:
        # print(hdulist[0].header)
        # assert(1==0)
        ticid = hdulist[0].header['TICID']
        # print(hdulist[0].header['TICID'])
        dont_exclude = [0,64,256,1024,2048,8192]
        # read time, sap flux, quality flag
        tess_bjds = hdulist[1].data['TIME']
        # print(hdulist[0].header)
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        qual_flags = hdulist[1].data['QUALITY']
        # remove flagged data
        where_gt0 = np.where(np.isin(qual_flags,dont_exclude))
        tess_bjds = tess_bjds[where_gt0]
        sap_fluxes = sap_fluxes[where_gt0]
        return (tess_bjds, sap_fluxes, ticid)



def extract_TGLC(curve_path):
    """
    Extracts the time, fluxes from a fits file
    curve_path: path to the TGLC fits file
    """
    with fits.open(curve_path, mode="readonly") as hdulist:
        #Flags to keep:
        dont_exclude = [0,64,256,1024,2048,8192]
        ticid = hdulist[0].header['TICID']
        # read time, sap flux, quality flag
        tess_bjds = hdulist[1].data['time']
        sap_fluxes = hdulist[1].data['aperture_flux']
        qual_flags = hdulist[1].data['TESS_flags']

        # remove flagged data
        where_no_flag = np.where(np.isin(qual_flags,dont_exclude))
        tess_bjds = tess_bjds[where_no_flag]
        sap_fluxes = sap_fluxes[where_no_flag]
        return (tess_bjds, sap_fluxes, ticid)
