import numpy as np
import math
    
########## Common Configs ##########

configs_DDP = {
    'backend' : 'nccl',
    'world_size' : 2
}

configs_segment = {
    'window_length' : 1000, # unit: ms
    'overlapping' : 300, # unit: ms
    'sfreq' : 250, # unit: Hz
    'percentile' : 0.95
}

configs_fft = {
    'l_freq' : 2,
    'h_freq' : 90
}