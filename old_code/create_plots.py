import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse

from numpy.core.defchararray import add

from powerspectrum import powerspectrum

from utils2 import load_lightcurve

def create_plot(row, BASE_DIR):
    try:
        lc = load_lightcurve(row['local_path'], starid=row['TIC'])
        flux = lc.flux.value
        time = lc.time.value

        if lc.flux.value.shape == (0,):
            flux = np.zeros(4000)
            time = np.zeros(4000)

    except Exception as e:
        print(f"Error: {e}")
        print(row['local_path'])

        flux = np.zeros(4000)
        time = np.zeros(4000)

        pass

    lc_df = pd.DataFrame({'flux': flux, 'time': time})

    try:
        ps = powerspectrum(lc_df)
        frequency, power = ps.powerspectrum(scale='amplitude', oversampling=4)
    except (ZeroDivisionError,ValueError) as e:
        print(f"Error: {e}")
        print(row['local_path'])

        frequency = np.ones(200)
        power = np.ones(200)

        pass

    plt.figure(figsize=(19,9.5))
    ax = plt.subplot(211)

    plt.title('TIC ' + str(row['TIC']), pad=15)
    plt.plot(lc_df.time, lc_df.flux, color='dimgrey', linewidth=0.6)
    plt.xlabel('time [days]')
    plt.ylabel('flux [ppm]')

    ax = plt.subplot(212)
    plt.plot(frequency, power, color='red', linewidth=1.2)
    ax.set_xlabel(r'frequency [$\mu$Hz]')
    ax.set_xlim((0,280))
    ax.set_ylim((0,np.max(power)))

    ax2 = plt.twiny()
    ax2.set_xticks(ax.get_xticks().tolist())
    ax.set_xlim((0,280))
    ax2.set_xticklabels(add((ax2.get_xticks()/(1/86400*1e6)).astype(str), ' c/d'))

    probs = row[['DSCT', 'GDOR', 'HYBRIDS', 'OTHER']]
    ax.text(0.5,0.95,'CLASS: ' + row['max_prob'],weight='bold',bbox=dict(facecolor='white', alpha=1), transform = ax.transAxes, ha='center', va='center')
    ax.text(0.7,0.82,"Probabilities" + '\n' + probs.to_string(float_format=lambda x: '%.3f' % x),bbox=dict(facecolor='white', alpha=0.8), transform = ax.transAxes, ha='right', va='center')

    plt.tight_layout()
    plt.savefig(BASE_DIR + 'figures/' + str(row['TIC']) + '_sector' + str(row['sector']) + '.pdf')
    plt.close()

def main(args):
    BASE_DIR = '/pdo/users/jeroena/ml_classification/variable-transformer/training_set/data/targets_QLP/'

    # Load data file containing target light curves
    targets_file = BASE_DIR + 'selection_results_pulsators_QLP_em1_gaia.csv'
    df_targets = pd.read_csv(targets_file).iloc[11904:]

    for idx, row in tqdm(df_targets.iterrows(), total=df_targets.shape[0]):
        create_plot(row, BASE_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs training")
    parser.add_argument('-t', '--type', type=str, default='default', choices=['default', 'pulsators', 'l1', 'l2'], help='Training set labels')

    args = parser.parse_args()

    main(args)

'''
    sector_paths = [
    'results_pulsators_QLP_em1_sector_27.csv',
    'results_pulsators_QLP_em1_sector_28.csv',
    'results_pulsators_QLP_em1_sector_29.csv',
    'results_pulsators_QLP_em1_sector_30.csv',
    'results_pulsators_QLP_em1_sector_31.csv',
    'results_pulsators_QLP_em1_sector_32.csv',
    'results_pulsators_QLP_em1_sector_33.csv',
    'results_pulsators_QLP_em1_sector_34.csv',
    'results_pulsators_QLP_em1_sector_35.csv',
    'results_pulsators_QLP_em1_sector_36.csv',
    'results_pulsators_QLP_em1_sector_37.csv',
    'results_pulsators_QLP_em1_sector_38.csv',
    'results_pulsators_QLP_em1_sector_39.csv']
'''