import torch
import numpy as np

from .augmentaton import _check_param_numpy


def _check_param_fft_freq(s_freq, l_freq, h_freq):
    if (not isinstance(s_freq, (int, float))) or (not isinstance(l_freq, (int, float))) or (not isinstance(h_freq, (int, float))):
        raise TypeError(
            'Frequency must be float or int, but got {0} for sfreq, {1} for l_freq, {2} for h_freq.'.format(type(s_freq), type(l_freq), type(h_freq))
        )
    
    if s_freq <= 0 or l_freq < 0 or h_freq < 0:
        raise ValueError('Invalid frequencies. Got {0} for s_freq, {1} for l_freq, {2} for h_freq.'.format(s_freq, l_freq, h_freq))
    
    if l_freq >= h_freq:
        raise ValueError('Lower bound should be smaller than higher bound.')


def amplitude_spectrum_numpy(x, s_freq, l_freq, h_freq):
    """
    Get amplitude spectrum for input data.

    Parameters
    ----------
    x : numpy array, shape(..., n_samples)
        The time-domain data.
    s_freq : float or int
        Sampling frequency.
    l_freq : float or int
        The lower bound of frequencies of interest.
    h_freq : float or int
        The higher bound of frequencies of interest.
    
    Returns
    -------
    amplitude_spectrum : numpy array, shape(..., n_freq_components)

    """
    _check_param_numpy(x)
    _check_param_fft_freq(s_freq, l_freq, h_freq)

    x_fft = np.fft.fft(x, axis=-1)
    amplitude_spectrum = np.abs(x_fft)
    frequencies = np.fft.fftfreq(
        n=x.shape[-1],
        d=1/s_freq
    )

    mask = (frequencies >= l_freq) & (frequencies <= h_freq)

    return amplitude_spectrum[..., mask]


def amplitude_spectrum_torch(x, s_freq, l_freq, h_freq):
    """
    Get amplitude spectrum for input data. (GPU-enabled)

    Parameters
    ----------
    x : torch.Tensor, shape(..., n_samples)
        The time-domain data.
    s_freq : float or int
        Sampling frequency.
    l_freq : float or int
        The lower bound of frequencies of interest.
    h_freq : float or int
        The higher bound of frequencies of interest.
    
    Returns
    -------
    amplitude_spectrum : torch.Tensor, shape(..., n_freq_components)

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError('Input must be a torch.Tensor.')

    _check_param_fft_freq(s_freq, l_freq, h_freq)
    
    x_fft = torch.fft.fft(x, dim=-1)
    amplitude_spectrum = torch.abs(x_fft)
    frequencies = torch.fft.fftfreq(
        n=x.shape[-1],
        d=1/s_freq
    ).to(x.device)

    mask = (frequencies >= l_freq) & (frequencies <= h_freq)

    return amplitude_spectrum[..., mask]
