import numpy as np
import mne
from mne_bids import read_raw_bids


def load_fif(fname, sfreq=None, picks='meg', allow_maxshield=False, preload=True):
    """
    Read .fif file and get the data

    Parameters
    ----------
    fname : path-like | file-like
        The raw filename to load.
    sfreq : int, float or None
        The resampling rate. If None, no resampling will be adopted.
    picks : str
        Channels to include.
    allow_maxshield : bool
        If True, allow loading of data that has been recorded with internal active compensation (MaxShield).
    preload : bool
        Preload data into memory for data manipulation and faster indexing. If True, the data will be preloaded into memory.
    
    Returns
    -------
    data : numpy array, shape (n_channels, n_times)
    
    """
    assert picks in ['meg', 'mag', 'grad'], 'Picked channcels must be magnetometers, gradiometers or both.'
    
    raw = mne.io.read_raw_fif(fname, allow_maxshield, preload=preload)
    line_freq = raw.info['line_freq']
    raw = raw.notch_filter(freqs=np.arange(line_freq, line_freq * 5 + 1, line_freq), notch_widths=1)
    if sfreq is not None:
        raw = raw.resample(sfreq=sfreq, method='fft')
    data = raw.get_data(picks=picks, units=dict(grad='fT/cm', mag='fT'))

    return data


def load_bids(bids_path, sfreq=None, picks='meg', allow_maxshield=False, preload=True):
    """
    Read MEG data in the format of BIDS

    Parameters
    ----------
    bids_path : BIDSPath
        The raw filename to load.
    sfreq : int, float or None
        The resampling rate. If None, no resampling will be adopted.
    picks : str
        Channels to include.
    allow_maxshield : bool
        If True, allow loading of data that has been recorded with internal active compensation (MaxShield).
    preload : bool
        Preload data into memory for data manipulation and faster indexing. If True, the data will be preloaded into memory.
    
    Returns
    -------
    data : numpy array, shape (n_channels, n_times)
    
    """
    assert picks in ['meg', 'mag', 'grad'], 'Picked channcels must be magnetometers, gradiometers or both.'
    
    raw = read_raw_bids(bids_path, extra_params=dict(allow_maxshield=allow_maxshield, preload=preload))
    line_freq = raw.info['line_freq']
    raw = raw.notch_filter(freqs=np.arange(line_freq, line_freq * 5 + 1, line_freq), notch_widths=1)
    if sfreq is not None:
        raw = raw.resample(sfreq=sfreq, method='fft')
    data = raw.get_data(picks=picks, units=dict(grad='fT/cm', mag='fT'))

    return data


def channel_normalize(x, percentile=0.95):
    """
    Normalization for each channel

    Parameters
    ----------
    x : numpy array, shape (..., n_times)
        The input data.
    percentile : float
        The percentile used to normalize each channel.
        E.g., '95-percentile of the absolute amplitude will be used to normalize each channel...'
    
    Returns
    -------
    x : numpy array, shape (..., n_times)
        Data after normalization.
    
    """
    if percentile > 1 or percentile < 0:
        raise ValueError('Percentile should be a number between 0 and 1.')

    div_term = np.percentile(np.abs(x), percentile, axis=-1, keepdims=True)
    
    return x / div_term


def data_splitting(x, sfreq, window_length, overlapping):
    """
    Split data into segments

    Parameters
    ----------
    x : numpy array, shape (..., n_times)
        The input data.
    sfreq : int or float
        Sampling frequency of signal.
    window_length : int or float
        The length of a segment. Unit: ms
    overlapping : int or float
        The length of overlapping part between two segments. Unit: ms

    Returns
    -------
    splited_x : numpy array, shape (..., n_windows, window_samples)

    """
    if not isinstance(window_length, (int, float)) or not isinstance(overlapping, (int, float)):
        raise TypeError('Window length and overlapping should be float or int.')
    
    if window_length <= 0 or overlapping <= 0:
        raise ValueError('Window length and overlapping should be non-negative.')

    if overlapping >= window_length:
        raise ValueError('The number of overlapping samples should be smaller than window length.')

    n_samples = x.shape[-1]
    window_samples = int(sfreq * window_length / 1000)
    overlapping_samples = int(sfreq * overlapping / 1000)
    
    stride = window_samples - overlapping_samples
    n_windows = (n_samples - overlapping_samples) // stride

    indices = np.arange(window_samples).reshape(1, -1) + \
                stride * np.arange(n_windows).reshape(-1, 1)
    
    return np.take(x, indices, axis=-1)