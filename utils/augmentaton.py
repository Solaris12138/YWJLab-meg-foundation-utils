import numpy as np
import torch
import torch.nn.functional as F


def _check_param_numpy(x):
    if not isinstance(x, np.ndarray):
        raise TypeError('Input x must be a numpy array.')
    if x.ndim < 1:
        raise ValueError('Input x must have at least one dimension.')


class TimeDomainAugmentationBankNumpy(object):
    
    @staticmethod
    def time_shift(x, shift=None):
        """
        Data shifting along with the time dimension

        Parameters
        ----------
        x : numpy array, shape (..., n_times)
            The data to be shifted.
        shift : int or None
            The number of samples that the data will be shifted. If None, one in two hundredth will be adopted.
        
        Returns
        -------
        x : numpy array, shape (..., n_times)

        """
        _check_param_numpy(x)

        if shift is not None and (not isinstance(shift, int) or shift < 0):
            raise ValueError('The number of samples should be a non-negative integer.')

        shift = shift if shift is not None else max(1, int(x.shape[-1] / 200))
        
        return np.roll(x, shift=shift, axis=-1)
    
    @staticmethod
    def direct_current_shift(x, alpha=None):
        """
        Data shifting along with the amplitude dimension

        Parameters
        ----------
        x : numpy array, shape (..., n_times)
            The data to be shifted.
        alpha : int, float, or None
            The amplitude that will be added to the data. If None, 1% of average values will be taken.
        
        Returns
        -------
        x : numpy array, shape (..., n_times)

        """
        _check_param_numpy(x)

        if alpha is not None and not isinstance(alpha, (int, float)):
            raise TypeError('Alpha must be a float, int, or None.')

        return x + (alpha if alpha is not None else np.mean(x) / 100)
    
    @staticmethod
    def smooth(x, sfreq=1000, window_length=2):
        """
        Smooth with sliding windows
    
        Parameters
        ----------
        x : numpy array, shape (..., n_times)
            The data to be smoothed.
        sfreq : int or float
            Sampling rate of signals. Unit: Hz
        window_length : int
            The length of sliding windows. Unit: ms
        
        Returns
        -------
        smoothed_x : numpy array, shape (..., n_times)
        
        """
        _check_param_numpy(x)

        if not isinstance(window_length, int) or window_length <= 0:
            raise ValueError('The length of sliding windows should be a positive integer.')

        window_samples = int(sfreq * window_length / 1000)
        window = np.ones(window_samples) / window_samples

        smoothed_x = np.zeros_like(x)
        for i in np.ndindex(x.shape[:-1]):
            smoothed_x[i] = np.convolve(x[i], window, mode='same')
        
        return smoothed_x

    @staticmethod
    def amplitude_scaling(x, scaling=None):
        """
        Data scaling

        Parameters
        ----------
        x : numpy array, shape (..., n_times)
            The data to be scaled.
        scaling : float or None
            If None, it will be randomly selected from [0.75, 1.5].

        Returns
        -------
        x : numpy array, shape (..., n_times)
        
        """
        _check_param_numpy(x)

        if scaling is not None and scaling <= 0:
            raise ValueError('Scaling factor must be a positive number.')
        
        return x * (scaling if scaling else np.random.uniform(0.75, 1.5))
    
    @staticmethod
    def add_noise(x, scaling=1):
        """
        Adding random noise to the data

        Parameters
        ----------
        x : numpy array, shape (..., n_times)
            The data.
        scaling : float or int
            The scale of the adding noise. Default: 1

        Returns
        -------
        x : numpy array, shape (..., n_times)
        
        """
        _check_param_numpy(x)

        if scaling < 0:
            raise ValueError('Scaling factor must be a non-negative number.')
        
        return x + np.random.randn(*x.shape) * scaling


class TimeDomainAugmentationBankTorch(object):
    
    @staticmethod
    def time_shift(x, shift=None):
        """
        Data shifting along with the time dimension (GPU-enabled)

        Parameters
        ----------
        x : torch.Tensor, shape (..., n_times)
            The data to be shifted.
        shift : int or None
            The number of samples that the data will be shifted. If None, one in two hundredth will be adopted.
        
        Returns
        -------
        x : torch.Tensor, shape (..., n_times)

        """
        if shift is not None and (not isinstance(shift, int) or shift < 0):
            raise ValueError('The number of samples should be a non-negative integer.')

        shift = shift if shift is not None else max(1, int(x.shape[-1] / 200))
        
        return torch.roll(x, shifts=shift, dims=-1)
    
    @staticmethod
    def direct_current_shift(x, alpha=None):
        """
        Data shifting along with the amplitude dimension (GPU-enabled)

        Parameters
        ----------
        x : torch.Tensor, shape (..., n_times)
            The data to be shifted.
        alpha : int, float or None
            The amplitude that will be added to the data. If None, 1% of average values will be taken.
        
        Returns
        -------
        x : torch.Tensor, shape (..., n_times)

        """
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a torch.Tensor.')
        
        if alpha is not None and not isinstance(alpha, (int, float)):
            raise TypeError('Alpha must be a float, int, or None.')

        alpha = alpha if alpha is not None else torch.mean(x).item() / 100
        
        return x + alpha
    
    @staticmethod
    def smooth(x, sfreq=1000, window_length=2):
        """
        Smooth with sliding windows (GPU-enabled)
    
        Parameters
        ----------
        x : torch.Tensor, shape ([batch_size], n_channel, n_signal, n_times)
            The data to be smoothed. Can be 3D or 4D
        sfreq : int or float
            Sampling rate of signals. Unit: Hz
        window_length : int
            The length of sliding windows. Unit: ms
        
        Returns
        -------
        smoothed_x : torch.Tensor, shape ([batch_size], n_channel, n_signal, n_times)
        
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a torch.Tensor.')

        if not isinstance(window_length, int) or window_length <= 0:
            raise ValueError('The length of sliding windows should be a positive integer.')
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError('Input tensor must be 3D or 4D, but got {0}D.'.format(x.dim()))
        
        if x.dtype != torch.float32:
            x = x.float()
        
        n_channel, n_signal = x.shape[1], x.shape[2]

        window_samples = max(1, int(sfreq * window_length / 1000))
        window = torch.ones(window_samples, device=x.device) / window_samples
        window = window.view(1, 1, -1)
        window = window.repeat(n_channel * n_signal, 1, 1)
        
        x = x.view(x.shape[0], n_channel * n_signal, x.shape[-1])

        smoothed_x = F.conv1d(x, window, padding='same', groups=n_channel * n_signal)
        smoothed_x = smoothed_x.reshape(x.shape[0], n_channel, n_signal, -1)

        if x.dim() == 3:
            smoothed_x = smoothed_x.squeeze(0)
        
        return smoothed_x
    
    @staticmethod
    def amplitude_scaling(x, scaling=None):
        """
        Data scaling (GPU-enabled)

        Parameters
        ----------
        x : torch.Tensor, shape (..., n_times)
            The data to be scaled.
        scaling : float or None
            If None, a random scaling factor will be randomly selected from [0.75, 1.5].

        Returns
        -------
        x : torch.Tensor, shape (..., n_times)
        
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a torch.Tensor.')

        if scaling is not None and scaling <= 0:
            raise ValueError('Scaling factor must be a positive number.')

        return x * (scaling if scaling else torch.rand(1).item() * (1.5 - 0.75) + 0.75)
    
    @staticmethod
    def add_noise(x, scaling=1):
        """
        Adding random noise to the data (GPU-enabled)

        Parameters
        ----------
        x : torch.Tensor, shape (..., n_times)
            The data.
        scaling : float or int
            The scale of the adding noise. Default: 1

        Returns
        -------
        x : torch.Tensor, shape (..., n_times)
        
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a torch.Tensor.')
        
        if scaling < 0:
            raise ValueError('Scaling factor must be a non-negative number.')

        return x + torch.randn_like(x) * scaling

    
class FreqDomainAugmentationBankNumpy(object):

    @staticmethod
    def remove_freqcomponent(x, perturb_ratio=0.1):
        """
        Remove frequency components.

        Parameters
        ----------
        x : numpy array, shape (..., n_freqs)
            The data.
        perturb_ratio : float
            The ratio of the frequency components being perturbed. Default: 0.1
        
        Returns
        -------
        x : numpy array, shape (..., n_freqs)
        
        """
        _check_param_numpy(x)

        if not isinstance(perturb_ratio, float):
            raise TypeError('Argument perturb_ratio must be a float.')

        if perturb_ratio < 0 or perturb_ratio >= 1:
            raise ValueError('The ratio of the frequency components being perturbed must be in [0, 1), but got {0}'.format(perturb_ratio))

        mask = np.random.rand(*x.shape) > perturb_ratio
        
        return x * mask
    
    @staticmethod
    def add_freqcomponent(x, perturb_ratio=0.1, threshold=0.2):
        """
        Add value to frequency components. (GPU-enabled)

        Parameters
        ----------
        x : numpy array, shape (..., n_freqs)
            The data.
        perturb_ratio : float
            The ratio of the frequency components being perturbed. Default: 0.1
        threshold : float
            The frequency component of which amplitude is smaller than threshold * maximum amplitude will be considered for perturbation.
        
        Returns
        -------
        x : numpy array, shape (..., n_freqs)
        
        """
        _check_param_numpy(x)

        if not isinstance(perturb_ratio, float):
            raise TypeError('Argument perturb_ratio must be a float.')
        
        if not isinstance(threshold, float):
            raise TypeError('Argument threshold must be a float.')

        if perturb_ratio < 0 or perturb_ratio >= 1:
            raise ValueError('The ratio of the frequency components being perturbed must be in [0, 1), but got {0}'.format(perturb_ratio))
        
        if threshold <= 0 or threshold >= 1:
            raise ValueError('The threshold must be in (0, 1), but got {0}'.format(threshold))
        
        mask = np.random.rand(*x.shape) > (1 - perturb_ratio)

        max_values = np.max(x, axis=-1, keepdims=True)
        max_amplitude_ = max_values * threshold
        am_mask = (x <= max_amplitude_)

        return x + mask * am_mask * max_amplitude_

    
class FreqDomainAugmentationBankTorch(object):

    @staticmethod
    def remove_freqcomponent(x, perturb_ratio=0.1):
        """
        Remove frequency components. (GPU-enabled)

        Parameters
        ----------
        x : torch.Tensor, shape (..., n_freqs)
            The data.
        perturb_ratio : float
            The ratio of the frequency components being perturbed. Default: 0.1
        
        Returns
        -------
        x : torch.Tensor, shape (..., n_freqs)
        
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a torch.Tensor.')

        if not isinstance(perturb_ratio, float):
            raise TypeError('Argument perturb_ratio must be a float.')

        if perturb_ratio < 0 or perturb_ratio >= 1:
            raise ValueError('The ratio of the frequency components being perturbed must be in [0, 1), but got {0}'.format(perturb_ratio))

        mask = torch.rand(x.shape, device=x.device) > perturb_ratio
        
        return x * mask
    
    @staticmethod
    def add_freqcomponent(x, perturb_ratio=0.1, threshold=0.2):
        """
        Add value to frequency components. (GPU-enabled)

        Parameters
        ----------
        x : torch.Tensor, shape (..., n_freqs)
            The data.
        perturb_ratio : float
            The ratio of the frequency components being perturbed. Default: 0.1
        threshold : float
            The frequency component of which amplitude is smaller than threshold * maximum amplitude will be considered for perturbation.
        
        Returns
        -------
        x : torch.Tensor, shape (..., n_freqs)
        
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a torch.Tensor.')

        if not isinstance(perturb_ratio, float):
            raise TypeError('Argument perturb_ratio must be a float.')
        
        if not isinstance(threshold, float):
            raise TypeError('Argument threshold must be a float.')

        if perturb_ratio < 0 or perturb_ratio >= 1:
            raise ValueError('The ratio of the frequency components being perturbed must be in [0, 1), but got {0}'.format(perturb_ratio))
        
        if threshold <= 0 or threshold >= 1:
            raise ValueError('The threshold must be in (0, 1), but got {0}'.format(threshold))
        
        mask = torch.rand(x.shape, device=x.device) > (1 - perturb_ratio)

        max_values, _ = torch.max(x, dim=-1, keepdim=True)
        max_amplitude_ = max_values.expand_as(x) * threshold
        am_mask = (x <= max_amplitude_)

        return x + mask * am_mask * max_amplitude_
