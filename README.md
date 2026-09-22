# YWJLab MEG Foundation Model Utilities

**Status: unfinished — project handed over.**

This is an unfinished pre-training project for a MEG foundation model. It has been handed over to
**Claire Shao**. The repository snapshot below contains the data pipeline that was built and the
model code that was under development on top of it. Note that there is no standalone training entry
point in this snapshot: `utils/model_wrapper.py` holds the pre-training loop that the (missing)
training script was expected to drive.

All data is expected to live under a BIDS root at `./data/bids`.

## Repository layout

```
.
├── .gitignore
├── 1-download.py
├── 2-preprocessing.py
├── 3-freesurfer.py
├── 4-BEM.py
├── 5-coregistration.py
├── 6-forward_solution.py
├── 7-inverse_compute.py
├── configs/
│   ├── configs_common.py
│   ├── configs_datasets.py
│   └── configs_pretrain_mae.py
├── models/
│   ├── mae.py
│   ├── mscnn_transformer.py
│   ├── mscnn_adjacency_aware_transformer.py
│   ├── mscnn_local_global_transformer.py
│   └── components/
│       ├── __init__.py
│       ├── encoder.py
│       ├── predictor.py
│       ├── reconstructor.py
│       ├── patch_embedding.py
│       ├── transformer.py
│       ├── multi_scale_cnn.py
│       ├── adjacency_aware_transformer.py
│       ├── local_global_transformer.py
│       └── utils/
│           ├── __init__.py
│           ├── apply_mask.py
│           └── position_encoding.py
└── utils/
    ├── augmentaton.py
    ├── basic_dataset.py
    ├── data_preprocessing.py
    ├── downstream_pipeline.py
    ├── fft.py
    ├── fs_processor.py
    ├── loss_function.py
    └── model_wrapper.py
```

## Processing pipeline (numbered scripts)

The numbered scripts are meant to be run in order. Each one iterates over the dataset lists defined
in `configs/configs_datasets.py` (`openneuro_datasets`, `zic_datasets`, `downstream_datasets`).

- **`1-download.py`** — Downloads the pre-training and downstream datasets from OpenNeuro into
  `./data/bids`. Each dataset has its own `include` / `exclude` filters so that only the needed
  BIDS, `anat`, MEG and derivative files are pulled. `--n_jobs` controls download concurrency.
- **`2-preprocessing.py`** — The main MEG cleaning script, with dataset-specific branches for
  handling `emptyroom` recordings, calibration / crosstalk files and line frequency (50 vs 60 Hz).
  Per recording it runs Maxwell filtering / SSS (optional tSSS via `-t` / `--tsss`), bad-channel
  detection, a 330 Hz low-pass filter, resampling to 1000 Hz, notch filtering of line-frequency
  harmonics, muscle-artifact annotation, and ICA (`infomax`, 20 components) whose components are
  labeled with `mne-icalabel` (`megnet`) and removed together with the ECG / EOG / muscle components
  detected by MNE. Noise covariance is estimated from the empty-room recording when available,
  otherwise from the pre-stimulus rest period (falling back to an identity matrix when there is not
  enough data). Downstream datasets are processed separately and are kept at their native sampling
  rate. `--n_jobs` sets the worker count.
- **`3-freesurfer.py`** — Runs FreeSurfer reconstruction in parallel on the downloaded T1w images
  via `utils/fs_processor.ParallelFreeSurferProcessor` (16 jobs by default), with a per-dataset
  table of input glob patterns, subject regexes and output directories. `ds003633` and the local
  `ywjaulg` dataset already ship FreeSurfer reconstructions and are skipped.
- **`4-BEM.py`** — Builds a watershed BEM model and BEM solution per subject (`oct6`, `ico=5`) and
  creates the surface source space, writing `<subject>-oct6-ico5-bem.fif`,
  `<subject>-oct6-ico5-bem-sol.fif` and `<subject>-oct6-ico5-src.fif` into each subject's `bem/`
  directory. For datasets that bring their own `fsaverage` (e.g. `ds003633`, `ywjaulg`), the
  `fsaverage` source spaces at `ico4` / `ico5` / `oct6` are generated first.
- **`5-coregistration.py`** — Automatic MEG–MRI coregistration. For every `*_meg.fif` it fits the
  fiducials, refines with ICP (`nasion_weight=5`), omits head-shape points closer than 5 mm and
  runs a second ICP pass (`nasion_weight=10`), then writes `*-trans.fif` into the `trans/`
  sub-directory next to the recording.
- **`6-forward_solution.py`** — Computes the MEG forward solution from each `*-trans.fif`, the BEM
  solution and the source space (`meg=True`, `eeg=False`, `mindist=5.0`) and writes `*-fwd.fif`
  into the `fwd/` sub-directory.
- **`7-inverse_compute.py`** — Builds the minimum-norm inverse operator from the forward solution
  and the pre-computed `noise-cov.fif` (`loose=0.2`, `depth=0.8`) and writes `*-inv.fif` into the
  `inv/` sub-directory.

## Model and pre-training code

- **`models/mae.py`** — `MaskAutoEncoder`: encoder plus momentum encoder, predictor and
  reconstructor; supports masked pre-training with partial prediction (`use_part_pred`).
- **`models/components/`** — Building blocks: `encoder.py` (patch embedding + transformer encoder
  with weight init), `predictor.py` / `reconstructor.py` (visible / invisible patch handling),
  `patch_embedding.py`, `transformer.py` (attention, feed-forward, blocks),
  `multi_scale_cnn.py` (multi-kernel temporal CNN with optional residual blocks),
  `adjacency_aware_transformer.py` (attention biased by a channel adjacency matrix),
  `local_global_transformer.py` (local + global attention branches), and `utils/` with mask
  application and sinusoidal position encoding.
- **`models/mscnn_transformer.py`**, **`models/mscnn_adjacency_aware_transformer.py`**,
  **`models/mscnn_local_global_transformer.py`** — `MSCT`, `MSCAAT` and `MSCLGT`: multi-scale CNN
  front-ends combined with the plain, adjacency-aware or local-global transformer variants.
- **`configs/configs_pretrain_mae.py`** — Pre-training configuration: input shape derived from
  window length / sampling rate, patch size and stride, `tiny` / `base` / `large` model settings,
  and the optimizer / scheduler / epoch settings consumed by the pre-training loop.
- **`utils/model_wrapper.py`** — `PretrainPipeline`: DDP process-group setup, optimizer and
  scheduler construction, mask generation, pre-training and validation steps, momentum-encoder
  update and checkpoint saving.

## Utilities

- **`utils/data_preprocessing.py`** — `load_fif` / `load_bids` loaders, per-channel normalization
  and windowed segmentation (`data_splitting`).
- **`utils/basic_dataset.py`** — `BasicDataset`, `AugmentDataset` and `AugmentDatasetTF`
  (time- and frequency-domain augmentation variants).
- **`utils/augmentaton.py`** — Time-domain (time shift, DC shift, smoothing, amplitude scaling,
  noise) and frequency-domain (component removal / addition) augmentation banks, in both NumPy and
  Torch flavours.
- **`utils/fft.py`** — Amplitude spectrum computation (NumPy and Torch).
- **`utils/loss_function.py`** — `NTXentPolyLoss` contrastive loss.
- **`utils/downstream_pipeline.py`** — Downstream evaluation helpers: `data_collector` plus
  `SVC_pipeline` and `MLP_pipeline` classifiers.
- **`utils/fs_processor.py`** — `ParallelFreeSurferProcessor` used by `3-freesurfer.py`.
- **`configs/configs_common.py`** — Shared settings: DDP backend / world size, segmentation window
  (1000 ms window, 300 ms overlap, 250 Hz), FFT band (2–90 Hz) and analysis percentile.
- **`configs/configs_datasets.py`** — Dataset lists and rejection criteria. Note that in this
  snapshot the dataset lists are commented out and must be re-enabled before running the pipeline.
