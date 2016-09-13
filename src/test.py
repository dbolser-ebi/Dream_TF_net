from datareader import DataReader
from wiggleReader import *
import os
import pywt
import matplotlib.pyplot as plt
import time
import gzip
import numpy as np

a = np.loadtxt('../data/preprocess/CHIPSEQ_FEATURES/A549_CTCF_200.gz')
print np.max(a), np.mean(a), np.std(a)



