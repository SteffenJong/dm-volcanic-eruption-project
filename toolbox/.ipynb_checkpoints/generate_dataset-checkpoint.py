import os, os.path
import glob
from pathlib import Path
import time
from IPython.display import clear_output
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_

def generate_aggregation(files, csv_read_path, new_csv_location, absolute=False, slice_amount=None, principal_components=None): 
    count = 1
    
    if os.path.exists(new_csv_location+".csv"):
        os.remove(new_csv_location+".csv")
    
    with tqdm(total=len([name for name in os.listdir(csv_read_path) if os.path.isfile(os.path.join(csv_read_path, name))])) as pbar:
        for file in glob.iglob(str(Path(csv_read_path,files+".csv"))):
            ts1 = time.time()

            filename = Path(file).stem

            # Read current file
            file_df = pd.read_csv(file)
            
            if(absolute):
                file_df = file_df.abs()
                
            if(slice_amount):
                file_df = file_df.iloc[::slice_amount]
            
            if(principal_components):
                pc_df = file_df.fillna(0)
                pc_df = StandardScaler().fit_transform(pc_df)

                pc_df = PCA(svd_solver="full", n_components=principal_components).fit_transform(pc_df)

                for i in range(0,principal_components):
                    file_df['pc'+str(i+1)] = pc_df[:,i]
            
            file_df = file_df.agg(['sum', 
                                   'mean', 
                                   'std', 
                                   'min', 
                                   'max', 
                                   'skew',
                                   'kurtosis',
                                   percentile(.1),
                                   percentile(.25), 
                                   percentile(.5), 
                                   percentile(.75),
                                   percentile(.9)])\
                        .unstack().to_frame().sort_index(level=1).T

            file_df.columns = file_df.columns.map('_'.join)

            # Use the file name as the segment_id
            file_df['segment_id'] = [filename]

            if(count==1): file_df.to_csv(new_csv_location+'.csv', mode='a', index=False, header=True)

            else: file_df.to_csv(new_csv_location+'.csv', mode='a', index=False, header=False)

            count = count + 1
            pbar.update(1)