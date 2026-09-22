import os
import numpy as np

from .data_preprocessing import channel_normalize
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


def data_collector(
    data_root,
    proj,
    sub_type=None,
    is_raw=False,
    if_normalize=True,
    percentile=0.95
):
    """
    Collect data for a specified downstream task dataset.
    
    Parameters
    ----------
    data_root : str | PathLike
        The folder which stores all the data for downstream tasks.
    proj : str
        The name of dataset. E.g., 'ds001117'.
    sub_type : None | str
        Only used when 'proj' is set as 'ds005356'. Should be 'reward' or 'punishment'.
    is_raw : bool
        If True, the data of one segment would be flatten. 
        If False, the data will be transformed into the shape which is suitable for pre-trained models.
    if_normalize : bool
        Whether to apply channel normalization to the data. Default: True.
    percentile : float
        The percentile used to normalize each channel.
        E.g., '95-percentile of the absolute amplitude will be used to normalize each channel...'
    
    Returns
    -------
    x : numpy.ndarray, shape (n_samples, 306 * n_times) or (n_samples, 3, 102, n_times)
        The collected data.
    y : numpy.ndarray, shape (n_samples, )
        The label of collected data.
    
    """
    x, y= list(), list()
    if sub_type is not None and sub_type not in ['reward', 'punishment']:
        raise ValueError(f'Sub_type should be reward or punishment in ds005356, but got {sub_type}.')
    
    if proj == 'ds000117':
        proj_dir = os.path.join(data_root, proj)
        if not os.path.exists(proj_dir):
            raise RuntimeError(f'Not found dataset: {proj_dir}')
        famous_dir = os.path.join(proj_dir, 'famous')
        scrambled_dir = os.path.join(proj_dir, 'scrambled')
        unfamiliar_dir = os.path.join(proj_dir, 'unfamiliar')
        
        for file in os.listdir(famous_dir):
            data = np.load(os.path.join(famous_dir, file))
            x.append(data.flatten() if is_raw else data.reshape(102, 3, -1).transpose(1, 0, 2))
            y.append(0)
        for file in os.listdir(scrambled_dir):
            data = np.load(os.path.join(scrambled_dir, file))
            x.append(data.flatten() if is_raw else data.reshape(102, 3, -1).transpose(1, 0, 2))
            y.append(1)
        for file in os.listdir(unfamiliar_dir):
            data = np.load(os.path.join(unfamiliar_dir, file))
            x.append(data.flatten() if is_raw else data.reshape(102, 3, -1).transpose(1, 0, 2))
            y.append(2)
        
    elif proj == 'ds003392':
        proj_dir = os.path.join(data_root, proj)
        if not os.path.exists(proj_dir):
            raise RuntimeError(f'Not found dataset: {proj_dir}')
        coherent_dir = os.path.join(proj_dir, 'coherent')
        incoherent_dir = os.path.join(proj_dir, 'incoherent')
        
        for file in os.listdir(coherent_dir):
            data = np.load(os.path.join(coherent_dir, file))
            x.append(data.flatten() if is_raw else data.reshape(102, 3, -1).transpose(1, 0, 2))
            y.append(0)
        for file in os.listdir(incoherent_dir):
            data = np.load(os.path.join(incoherent_dir, file))
            x.append(data.flatten() if is_raw else data.reshape(102, 3, -1).transpose(1, 0, 2))
            y.append(1)
            
    elif proj == 'ds005356':
        if sub_type is None:
            raise ValueError('Sub_type should not be none when dataset is ds005356.')
        proj_dir = os.path.join(data_root, proj, sub_type)
        if not os.path.exists(proj_dir):
            raise RuntimeError(f'Not found dataset: {proj_dir}')
        
        for file in os.listdir(proj_dir):
            data = np.load(os.path.join(proj_dir, file))
            x.append(data.flatten() if is_raw else data.reshape(102, 3, -1).transpose(1, 0, 2))
            y.append(0 if 'MDD' in file else 1)
    else:
        raise ValueError(f'Unexpected dataset: {proj}')
    
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y)

    if if_normalize:
        x = channel_normalize(x, percentile)
    
    return x, y


def SVC_pipeline(
    train_x,
    train_y,
    test_x,
    test_y,
    random_state=None,
    **kwargs
):
    """
    Downstream tasks pipeline of SVC.

    Parameters
    ----------
    train_x : numpy array, shape (n_samples, n_features)
        Training data samples.
    train_y : numpy array, shape (n_samples, )
        The true labels of training data samples.
    test_x : numpy array, shape (n_samples, n_features)
        Testing data samples.
    test_y : numpy array, shape (n_samples, )
        The true labels of testing data samples.
    
    Returns
    -------
    score : float

    """
    svc = LinearSVC(
        penalty='l2',
        tol=0.001,
        random_state=random_state,
        multi_class='ovr',
        max_iter=1000
    )
    svc.fit(train_x, train_y)
    return svc.score(test_x, test_y)


def MLP_pipeline(
    train_x,
    train_y,
    test_x,
    test_y,
    random_state=None,
    **kwargs
):
    """
    Downstream tasks pipeline of MLP.

    Parameters
    ----------
    train_x : numpy array, shape (n_samples, n_features)
        Training data samples.
    train_y : numpy array, shape (n_samples, )
        The true labels of training data samples.
    test_x : numpy array, shape (n_samples, n_features)
        Testing data samples.
    test_y : numpy array, shape (n_samples, )
        The true labels of testing data samples.
    
    Returns
    -------
    score : float

    """
    mlp = MLPClassifier(**kwargs, random_state=random_state)
    mlp.fit(train_x, train_y)
    return mlp.score(test_x, test_y)