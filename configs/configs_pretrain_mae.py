import numpy as np

from .configs_common import configs_DDP, configs_segment, configs_fft

def _get_inshape(window_length, s_freq, l_freq, h_freq):
    shape_t = int(window_length / 1000 * s_freq)
    freqs = np.fft.fftfreq(
        n=shape_t,
        d=1/s_freq
    )
    shape_f = len(freqs[(freqs >= l_freq) & (freqs <= h_freq)])

    return shape_t, shape_f

def _get_configs(
        input_size, patch_size, patch_stride=None, 
        embedding_dim=64, n_blocks=[2, 2, 4], n_heads=4, ffn_ratio=4, init_std=0.02
    ):
    
    num_patches = (
        input_size[0],
        input_size[1] // patch_size if patch_stride is None else (input_size[1] - patch_size) // patch_stride + 1
    )
    
    models_configs = {
        'encoder': {
                'input_size' : input_size,
                'patch_size' : patch_size,
                'patch_stride' : patch_stride,
                'embedding_dim': embedding_dim,
                'n_blocks': n_blocks[0],
                'n_heads': n_heads,
                'ffn_ratio' : ffn_ratio,
                'init_std' : init_std
            },
        'predictor': {
                'num_patches' : num_patches,
                'embedding_dim': embedding_dim,
                'n_blocks': n_blocks[1],
                'n_heads': n_heads,
                'ffn_ratio' : ffn_ratio,
                'init_std' : init_std
            },
        'reconstructor': {
                'num_patches' : num_patches,
                'patch_size' : patch_size,
                'embedding_dim': embedding_dim,
                'n_blocks': n_blocks[2],
                'n_heads': n_heads,
                'ffn_ratio' : ffn_ratio,
                'init_std' : init_std
            },
    }
    return models_configs

########## Predefined Model Settings ##########
 
model_settings = {
    'tiny' : {
        'embedding_dim' : 64, 'n_blocks' : [2, 2, 4], 'n_heads' : 4, 'ffn_ratio' : 4
    },
    'base' : {
        'embedding_dim' : 256, 'n_blocks' : [4, 4, 8], 'n_heads' : 8, 'ffn_ratio' : 4
    },
    'large': {
        'embedding_dim' : 512, 'n_blocks' : [8, 8, 8], 'n_heads' : 8, 'ffn_ratio' : 4
    }
}
    
########## MAE Pretrain Configs ##########

tag = 'tiny'

ch_types = 'mag'

train_set_ratio = 0.8

configs_patch = {
    'patch_size' : 5,
    'patch_stride' : None
}

in_shape_temp, in_shape_freq = _get_inshape(
    configs_segment['window_length'],
    configs_segment['sfreq'],
    configs_fft['l_freq'],
    configs_fft['h_freq']
)

input_size = [
    102 if ch_types == 'mag' else 204,
    in_shape_temp
]

configs_pretrain = {
    'batch_size' : 16,
    'num_workers' : 16,
    'train' : {
        'epochs_num' : 100,
        'opt_params' : {
            'optimizer_cls' : 'Adam',
            'scheduler_cls' : 'ReduceLROnPlateau',
            'lr' : 1e-4 * configs_DDP['world_size'],
            'weight_decay' : 1e-3
        }
    },
    'model' : {
        'models_configs' : _get_configs(input_size, **configs_patch, **model_settings[tag]),
        'use_part_pred' : True
    }
}