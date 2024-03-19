import os


class CFG:
    """Features+Head Starter"""

    VER = 2
    TEST_MODE = False

    # path
    BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
    # test
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
    EPOCHS = 5
    EPOCHS2 = 3
    BATCH_SIZE = 32
