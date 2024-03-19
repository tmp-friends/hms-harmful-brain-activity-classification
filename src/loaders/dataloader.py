from typing import List, Dict
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A

TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]


class DataLoader:
    """
    PyTorchのDatasetクラスのようなイメージ
    doc: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    スペクトログラムとEEGの両方を、学習サンプル毎に(128x256x8)サイズの8チャネル画像として出力
    (最初の4チャネルはスペクトログラム、次の4チャネルはEEG)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        specs: Dict[int, np.array],
        eegs: Dict[int, np.array],
        augment: bool = False,
        mode: str = "train",
    ):
        self.data = self._build_data(df)
        self.specs = specs
        self.eegs = eegs
        self.augment = augment
        self.mode = mode

        # Override
        self.on_epoch_end()

    def _build_data(self, df):
        data = pd.concat([df] * 3, ignore_index=True)
        data.loc[: len(df), "data_type"] = "K"
        data.loc[len(df) : len(df) * 2, "data_type"] = "E"
        data.loc[len(df) * 2 :, "data_type"] = "KE"

        return data

    def __len__(self) -> int:
        """epoch毎のバッチの総数を返す

        Returns:
            int: バッチの総数
        """
        return self.data.shape[0]

    def __getitem__(self, ix):
        """1バッチを生成"""
        X, y = self._generate_data(ix)

        if self.augment:
            # X, y = self._mixup(X, y)
            # X, y = self._hbac_cutmix(X, y)
            X = self._augmentation(X)

        return X, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """各epochのあとindicesを更新する"""
        if self.mode == "train":
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def _generate_data(self, ix):
        """データを生成"""
        row = self.data.iloc[ix]

        if row["data_type"] == "KE":
            X, y = self._generate_all_specs(ix)
        elif row["data_type"] in ["K", "E"]:
            X, y = self._generate_specs(ix)

        return X, y

    def _generate_all_specs(self, ix):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[ix]
        if self.mode == "test":
            offset = 0
        else:
            offset = int(row.offset / 2)

        # spec
        spec = self.specs[row["spec_id"]]

        # spectrogram is 10mins i.e 600secs so 300 units, midpoint is 150 so 145:155 is 20secs
        imgs = [
            spec[offset : offset + 300, v * 100 : (v + 1) * 100].T for v in [0, 2, 1, 3]
        ]  # to match kaggle with eeg
        img = np.stack(imgs, axis=-1)

        # img毎に対数標準化
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        # 時間軸に沿って、256time stepsにクロッピング
        X[0_0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_k
        X[0_0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_k
        X[0_0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_k
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_k

        X[0_0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_k
        X[0_0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_K

        # EEG
        eeg = self.eegs[row["eeg_id"]]
        img = eeg

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_e
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_e
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_e

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_e

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y

    def _generate_specs(self, ix):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[ix]
        if self.mode == "test":
            offset = 0
        else:
            offset = int(row["offset"] / 2)

        if row["data_type"] == "E":
            img = self.eegs[row["eeg_id"]]
        elif row["data_type"] == "K":
            spec = self.specs[row["spec_id"]]
            imgs = [
                spec[offset : offset + 300, v * 100 : (v + 1) * 100].T for v in [0, 2, 1, 3]
            ]  # to match kaggle with eeg
            img = np.stack(imgs, axis=-1)

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0_0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0_0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0_0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0_0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 1]
        X[0_0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 1]
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 2]
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 3]
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 2]

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y

    def _mixup(self, X: np.ndarray, y: np.ndarray, alpha=2.0):
        """MixUp処理を行う

        Args:
            X (np.ndarray): 入力データ
            y (np.ndarray): ラベルデータ
            alpha (float, optional): Mixup のパラメータ. Defaults to 0.2.

        Returns:
            np.ndarray: Mixup された入力データ
            np.ndarray: Mixup されたラベルデータ
        """
        lam = np.random.beta(alpha, alpha, X.shape[0])
        index_array = np.arange(X.shape[0])
        np.random.shuffle(index_array)

        # batch_sizeに対してmixupをbroadcastで処理
        mixed_X = lam.reshape(-1, 1, 1, 1) * X + (1 - lam).reshape(-1, 1, 1, 1) * X[index_array]
        mixed_y = lam.reshape(-1, 1) * y + (1 - lam).reshape(-1, 1) * y[index_array]

        return mixed_X, mixed_y

    def _hbac_cutmix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cutmix_thr=0.5,
        margin=50,
        min_size=25,
        max_size=75,
        cut_eeg_spec=True,
    ):
        """@ref: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/479446"""
        # CutMix Data
        cutmix_data = np.copy(X)  # Use np.copy for cloning in NumPy

        for label_idx in range(y.shape[1]):
            # Indices with a confidence score greater than cutmix_thr for particular target
            indices = np.nonzero(y[:, label_idx] >= cutmix_thr)[0]

            # Skip if less than 2 samples with confidence score 1.0
            if len(indices) < 2:
                continue

            # Original Data
            data_orig = X[indices, :, :, :]

            # Shuffle
            shuffled_indices = np.random.permutation(len(indices))
            data_shuffled = data_orig[shuffled_indices, :, :, :]

            # CutMix augmentation logic
            start = (
                random.randint(0, margin)
                if random.choice([True, False])
                else random.randint(300 - max_size - margin, 300 - max_size)
            )
            size = random.randint(min_size, max_size)

            # CutMix in Specs
            for idx in range(len(indices)):
                cutmix_data[indices[idx], :, start : start + size, :] = data_shuffled[idx, :, start : start + size, :]

            # CutMix in EEG Specs
            if cut_eeg_spec:
                start = 300 + 40 + start  # Size + Padding + Start
                for idx in range(len(indices)):
                    cutmix_data[indices[idx], :, start : start + size, :] = data_shuffled[
                        idx, :, start : start + size, :
                    ]

        return cutmix_data, y

    def _augmentation(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self._transform(img_batch[i,])

        return img_batch

    def _transform(self, img):
        """
        - https://www.kaggle.com/code/medali1992/hms-efficientnetb0-train/notebook
        - https://www.kaggle.com/code/iglovikov/xymasking-aug
        """
        params1 = {
            "num_masks_x": 1,
            "mask_x_length": (0, 20),  # This line changed from fixed to a range
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        params2 = {
            "num_masks_y": 1,
            "mask_y_length": (0, 20),
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        params3 = {
            "num_masks_x": (2, 4),
            "num_masks_y": 5,
            "mask_x_length": (10, 20),
            "mask_x_length": 8,
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }

        composition = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.3),
                # A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
                # A.XYMasking(**params1, p=0.3),
                # A.XYMasking(**params2, p=0.3),
                # A.XYMasking(**params3, p=0.3),
            ]
        )

        return composition(image=img)["image"]
