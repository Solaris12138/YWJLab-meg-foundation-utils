import numpy as np
import math

from .configs_common import configs_segment

########## Dataset Configs ##########

openneuro_datasets = [
    # 'ds004107',
    # 'ds004078',
    # 'ds003922',
    # 'ds005356',
    # 'ds003633',
    # 'ds004212',
    # 'ds006012',
    # 'ds006334',
]

zic_datasets = [
    # 'ywjaulg',
]

downstream_datasets = [
    # 'ds000117',
]

reject_criteria = dict(
    grad=4000e-13,  # unit: T / m (gradiometers)
    mag=4000e-15,   # unit: T (magnetometers)
)