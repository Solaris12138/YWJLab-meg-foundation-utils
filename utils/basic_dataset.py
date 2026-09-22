import numpy as np
import os
from os import listdir

import torch
from torch.utils.data import Dataset

from .augmentaton import TimeDomainAugmentationBankNumpy, FreqDomainAugmentationBankNumpy
from .data_preprocessing import channel_normalize
from .fft import amplitude_spectrum_numpy


class BasicDataset(Dataset):

    def __init__(self, data_root, if_normalize=True, percentile=0.95, dtype=torch.float32):
        """
        Initialize dataset

        Parameters
        ----------
        data_root : str
            The folder which stores the data. Each file should be a .npy file. E.g., '1023.npy'
        if_normalize : bool
            Whether to apply channel normalization to the data. Default: True.
        percentile : float
            The percentile used to normalize each channel.
            E.g., '95-percentile of the absolute amplitude will be used to normalize each channel...'
        dtype : torch.Tensor type
            The type of tensor to which the data will be transformed. Default: torch.float32

        """
        if not isinstance(data_root, str):
            raise TypeError('The parameter data_root should be a string, but got {0}.'.format(type(data_root)))

        self.data_root = data_root
        self.if_normalize = if_normalize
        self.percentile = percentile
        self.dtype = dtype
        self.num_files = len(listdir(data_root))

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_root, '{0}.npy'.format(index)))
        if self.if_normalize:
            data = channel_normalize(data, self.percentile)
        return {'temp' : torch.as_tensor(data, dtype=self.dtype)}


class AugmentDataset(Dataset):

    def __init__(self, data_root, s_freq, if_normalize=True, percentile=0.95, dtype=torch.float32):
        """
        Initialize dataset with augmentation (only in time domain).

        Parameters
        ----------
        data_root : str
            The folder which stores the data. Each file should be a .npy file. E.g., '1023.npy'
        s_freq : int or float
            Sampling rate of signals. Unit: Hz
        if_normalize : bool
            Whether to apply channel normalization to the data. Default: True.
        percentile : float
            The percentile used to normalize each channel.
            E.g., '95-percentile of the absolute amplitude will be used to normalize each channel...'
        dtype : torch.Tensor type
            The type of tensor to which the data will be transformed. Default: torch.float32

        """
        if not isinstance(data_root, str):
            raise TypeError('The parameter data_root should be a string, but got {0}.'.format(type(data_root)))

        self.data_root = data_root
        self.s_freq = s_freq
        self.if_normalize = if_normalize
        self.percentile = percentile
        self.dtype = dtype
        self.num_files = len(listdir(data_root))

        self.time_domain_aug_func = {
            '1' : TimeDomainAugmentationBankNumpy.time_shift,
            '2' : TimeDomainAugmentationBankNumpy.direct_current_shift,
            '3' : TimeDomainAugmentationBankNumpy.smooth,
            '4' : TimeDomainAugmentationBankNumpy.amplitude_scaling,
            '5' : TimeDomainAugmentationBankNumpy.add_noise
        }

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        data_temp = np.load(os.path.join(self.data_root, '{0}.npy'.format(index)))
        data_temp_aug = self._time_domain_aug(data_temp, np.random.randint(1, 6))
        if self.if_normalize:
            data_temp = channel_normalize(data_temp, self.percentile)
            data_temp_aug = channel_normalize(data_temp_aug, self.percentile)
        
        return {
            'temp' : torch.as_tensor(data_temp, dtype=self.dtype),
            'temp_aug' : torch.as_tensor(data_temp_aug, dtype=self.dtype),
        }
    
    def _time_domain_aug(self, x, func_id):
        """
        Augmentation in time domain

        Parameters
        ----------
        x : numpy array, shape (..., n_times)
            Data to be augmented.
        func_id : int
            The index of augmentation function.
        
        Returns
        -------
        x : numpy array, shape (..., n_times)
            Augmented data.

        """
        time_domain_aug_args = {
            '1' : {'shift' : np.random.randint(1, 5)},
            '2' : {'alpha' : None},
            '3' : {'sfreq' : self.s_freq, 'window_length': np.random.randint(1, 5) * 5},
            '4' : {'scaling' : None},
            '5' : {'scaling' : 1}
        }

        return self.time_domain_aug_func[str(func_id)](x, **time_domain_aug_args[str(func_id)])


class AugmentDatasetTF(Dataset):

    def __init__(self, data_root, s_freq, l_freq, h_freq, if_normalize=True, percentile=0.95, dtype=torch.float32):
        """
        Initialize dataset with augmentation (in both time and frequency domains).

        Parameters
        ----------
        data_root : str
            The folder which stores the data. Each file should be a .npy file. E.g., '1023.npy'
        s_freq : int or float
            Sampling rate of signals. Unit: Hz
        l_freq : float or int
            The lower bound of frequencies of interest.
        h_freq : float or int
            The higher bound of frequencies of interest.
        if_normalize : bool
            Whether to apply channel normalization to the data. Default: True.
        percentile : float
            The percentile used to normalize each channel.
            E.g., '95-percentile of the absolute amplitude will be used to normalize each channel...'
        dtype : torch.Tensor type
            The type of tensor to which the data will be transformed. Default: torch.float32

        """
        if not isinstance(data_root, str):
            raise TypeError('The parameter data_root should be a string, but got {0}.'.format(type(data_root)))

        self.data_root = data_root
        self.s_freq = s_freq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.if_normalize = if_normalize
        self.percentile = percentile
        self.dtype = dtype
        self.num_files = len(listdir(data_root))

        self.time_domain_aug_func = {
            '1' : TimeDomainAugmentationBankNumpy.time_shift,
            '2' : TimeDomainAugmentationBankNumpy.direct_current_shift,
            '3' : TimeDomainAugmentationBankNumpy.smooth,
            '4' : TimeDomainAugmentationBankNumpy.amplitude_scaling,
            '5' : TimeDomainAugmentationBankNumpy.add_noise
        }

        self.freq_domain_aug_func = {
            '1' : FreqDomainAugmentationBankNumpy.remove_freqcomponent,
            '2' : FreqDomainAugmentationBankNumpy.add_freqcomponent
        }

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        data_temp = np.load(os.path.join(self.data_root, '{0}.npy'.format(index)))
        data_temp_aug = self._time_domain_aug(data_temp, np.random.randint(1, 6))
        if self.if_normalize:
            data_temp = channel_normalize(data_temp, self.percentile)
            data_temp_aug = channel_normalize(data_temp_aug, self.percentile)
        
        data_freq = amplitude_spectrum_numpy(
            data_temp,
            s_freq=self.s_freq,
            l_freq=self.l_freq,
            h_freq=self.h_freq
        )
        data_freq_aug = self._freq_domain_aug(data_freq, np.random.randint(1, 3))
        
        return {
            'temp' : torch.as_tensor(data_temp, dtype=self.dtype),
            'temp_aug' : torch.as_tensor(data_temp_aug, dtype=self.dtype),
            'freq' : torch.as_tensor(data_freq, dtype=self.dtype),
            'freq_aug' : torch.as_tensor(data_freq_aug, dtype=self.dtype)
        }
    
    def _time_domain_aug(self, x, func_id):
        """
        Augmentation in time domain

        Parameters
        ----------
        x : numpy array, shape (..., n_times)
            Data to be augmented.
        func_id : int
            The index of augmentation function.
        
        Returns
        -------
        x : numpy array, shape (..., n_times)
            Augmented data.

        """
        time_domain_aug_args = {
            '1' : {'shift' : np.random.randint(1, 5)},
            '2' : {'alpha' : None},
            '3' : {'sfreq' : self.s_freq, 'window_length': np.random.randint(1, 5) * 5},
            '4' : {'scaling' : None},
            '5' : {'scaling' : 1}
        }

        return self.time_domain_aug_func[str(func_id)](x, **time_domain_aug_args[str(func_id)])

    def _freq_domain_aug(self, x, func_id):
        """
        Augmentation in time domain

        Parameters
        ----------
        x : numpy array, shape (..., n_freqs)
            Data to be augmented.
        func_id : int
            The index of augmentation function.
        
        Returns
        -------
        x : numpy array, shape (..., n_freqs)
            Augmented data.

        """
        freq_domain_aug_args = {
            '1' : {'perturb_ratio' : 0.05},
            '2' : {'perturb_ratio' : 0.1, 'threshold' : 0.2}
        }

        return self.freq_domain_aug_func[str(func_id)](x, **freq_domain_aug_args[str(func_id)])