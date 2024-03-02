import os


class CFG:
    VER = 1

    # path
    BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
    # train
    SPEC_FILE_PATH = os.path.join(BASE_PATH, "train_spectrograms")
    EEG_FILE_PATH = os.path.join(BASE_PATH, "train_eegs")
    EEG_SPEC_FILE_PATH = "/kaggle/input/brain-eeg-spectrograms/EEG_Spectrograms"
    # test
    SPEC_FILE_PATH_INFER = os.path.join(BASE_PATH, "test_spectrograms")
    EEG_FILE_PATH_INFER = os.path.join(BASE_PATH, "test_eegs")
    LOAD_MODELS_FROM_TRAIN = ""
    LOAD_MODELS_FROM_INFER = "/kaggle/input/hms-efficientnet-b0"

    # model
    MIX = True  # 混合精度を入れるか
    # training params
    LR_START = 1e-4
    LR_MAX = 1e-3
    LR_RAMPUP_EPOCHS = 0
    LR_SUSTAIN_EPOCHS = 1
    LR_STEP_DECAY = 0.1
    EVERY = 1
    EPOCHS = 4
