import os
from pathlib import Path


class CFG:
    VERSION = "82"

    wandb = False
    debug = False
    create_eegs = False
    apex = True
    visualize = False
    save_all_models = True

    if debug:
        num_workers = 0
        parallel = False
    else:
        num_workers = os.cpu_count()
        parallel = True

    model_name = "resnet1d_gru"
    # optimizer = "Adan"
    optimizer = "AdamW"

    factor = 0.9
    eps = 1e-6
    lr = 8e-3
    min_lr = 1e-6

    batch_size = 64
    batch_koef_valid = 2
    batch_scheduler = True
    weight_decay = 1e-2
    gradient_accumulation_steps = 1
    max_grad_norm = 1e7

    fixed_kernel_size = 5
    # linear_layer_features = 424
    # kernels = [3, 5, 7, 9]
    # linear_layer_features = 448  # Full Signal = 10_000
    # linear_layer_features = 352  # Half Signal = 5_000
    linear_layer_features = 304  # 1/4, 1/5, 1/6  Signal = 2_000
    # linear_layer_features = 280  # 1/10  Signal = 1_000
    kernels = [3, 5, 7, 9, 11]
    # kernels = [5, 7, 9, 11, 13]

    seq_length = 50  # Second's
    sampling_rate = 200  # Hz
    nsamples = seq_length * sampling_rate  # Число семплов 10_000
    n_split_samples = 5
    out_samples = nsamples // n_split_samples  # 2_000
    sample_delta = nsamples - out_samples  # 8000
    sample_offset = sample_delta // 2
    multi_validation = False

    train_by_stages = False
    train_by_folds = True

    # 'GPD', 'GRDA', 'LPD', 'LRDA', 'Other', 'Seizure'
    n_stages = 2
    match n_stages:
        case 1:
            train_stages = [0]
            epochs = [100]
            test_total_eval = 2
            total_evals_old = [[(2, 3), (6, 29)]]  # Deprecated
            total_evaluators = [
                [
                    {"band": (2, 2), "excl_evals": []},
                    {"band": (6, 28), "excl_evals": []},
                ],
            ]
        case 2:
            train_stages = [0, 1]
            epochs = [50, 100]
            test_total_eval = 4
            total_evals_old = [[(1, 4), (4, 5), (5, 6)], (6, 29)]  # Deprecated
            total_evaluators = [
                [
                    {"band": (1, 3), "excl_evals": []},
                    {"band": (4, 4), "excl_evals": ["GPD"]},
                    {"band": (5, 5), "excl_evals": []},
                ],
                [
                    {"band": (6, 28), "excl_evals": []},
                ],
            ]
        case 3:
            train_stages = [0, 1, 2]
            epochs = [20, 50, 100]
            test_total_eval = 0
            total_evals_old = [(0, 3), (3, 6), (6, 29)]  # Deprecated
            total_evaluators = [
                [
                    {"band": (0, 2), "excl_evals": []},
                ],
                [
                    {"band": (3, 5), "excl_evals": []},
                ],
                [
                    {"band": (6, 28), "excl_evals": []},
                ],
            ]

    n_fold = 5
    train_folds = [0, 1, 2, 3, 4]
    # train_folds = [0]

    patience = 11
    seed = 2024

    bandpass_filter = {"low": 0.5, "high": 20, "order": 2}
    rand_filter = {"probab": 0.1, "low": 10, "high": 20, "band": 1.0, "order": 2}
    freq_channels = []  # [(8.0, 12.0)]; [(0.5, 4.5)]
    filter_order = 2

    random_divide_signal = 0.05
    random_close_zone = 0.05
    random_common_negative_signal = 0.0
    random_common_reverse_signal = 0.0
    random_negative_signal = 0.05
    random_reverse_signal = 0.05

    log_step = 100  # Шаг отображения тренировки
    log_show = False

    scheduler = "CosineAnnealingWarmRestarts"  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts','OneCycleLR']

    # CosineAnnealingLR params
    cosanneal_params = {
        "T_max": 6,
        "eta_min": 1e-5,
        "last_epoch": -1,
    }

    # ReduceLROnPlateau params
    reduce_params = {
        "mode": "min",
        "factor": 0.2,
        "patience": 4,
        "eps": 1e-6,
        "verbose": True,
    }

    # CosineAnnealingWarmRestarts params
    cosanneal_res_params = {
        "T_0": 20,
        "eta_min": 1e-6,
        "T_mult": 1,
        "last_epoch": -1,
    }

    target_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    pred_cols = [x + "_pred" for x in target_cols]

    map_features = [
        ("Fp1", "T3"),
        ("T3", "O1"),
        ("Fp1", "C3"),
        ("C3", "O1"),
        ("Fp2", "C4"),
        ("C4", "O2"),
        ("Fp2", "T4"),
        ("T4", "O2"),
        # ('Fz', 'Cz'), ('Cz', 'Pz'),
    ]

    eeg_features = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]  # 'Fz', 'Cz', 'Pz'
    # 'F3', 'P3', 'F7', 'T5', 'Fz', 'Cz', 'Pz', 'F4', 'P4', 'F8', 'T6', 'EKG']
    feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}
    simple_features = []  # 'Fz', 'Cz', 'Pz', 'EKG'

    # eeg_features = [row for row in feature_to_index]
    # eeg_feat_size = len(eeg_features)

    n_map_features = len(map_features)
    in_channels = n_map_features + n_map_features * len(freq_channels) + len(simple_features)
    target_size = len(target_cols)

    path_inp = Path("/kaggle/input")
    path_src = path_inp / "hms-harmful-brain-activity-classification/"
    file_train = path_src / "train.csv"
    path_train = path_src / "train_eegs"
    file_features_test = path_train / "100261680.parquet"
    file_eeg_specs = path_inp / "eeg-spectrogram-by-lead-id-unique/eeg_specs.npy"
    file_raw_eeg = path_inp / "brain-eeg/eegs.npy"
    # file_raw_eeg = path_inp / "brain-eegs-plus/eegs.npy"
    # file_raw_eeg = path_inp / "brain-eegs-full/eegs.npy"
