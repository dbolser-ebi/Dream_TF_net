import pandas as pd
import numpy as np
a = pd.read_csv('../data/preprocess/DNASE_FEATURES/A549_ladder_600.gz', delimiter=' ',nrows=1000)
a = (a-a.mean()) / a.std()
print a.as_matrix
