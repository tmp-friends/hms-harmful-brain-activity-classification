from typing import List, Dict
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as albu


class DataLoader(tf.keras.utils.Sequence):
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
        label_columns: List[str],
        batch_size: int = 32,
        shuffle: bool = False,
        augment: bool = False,
        mode: str = "train",
    ):
        self.df = df
        self.specs = specs
        self.eegs = eegs
        self.label_columns = label_columns
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode

        # Override
        self.on_epoch_end()

    def __len__(self) -> int:
        """epoch毎のバッチの総数を返す

        Returns:
            int: バッチの総数
        """
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index) -> None:
        """1バッチを生成"""
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        X, y = self._generate_data(indices)
        X, y = self._hbac_cutmix(X, y)

        if self.augment:
            X = self._augment_batch(X)

        return X, y

    def on_epoch_end(self) -> None:
        """各epochのあとindicesを更新する"""
        self.indices = np.arange(len(self.df))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_data(self, indices):
        """バッチサイズ分のデータを生成

        Args:
            indices:
        """
        X = np.zeros((len(indices), 128, 256, 8), dtype="float32")
        y = np.zeros((len(indices), 6), dtype="float32")
        img = np.ones((128, 256), dtype="float32")

        for j, i in enumerate(indices):
            row = self.df.iloc[i]
            if self.mode == "test":
                r = 0
            else:
                # subsequenceの中点
                # 中点は(min+max)//2だが、各単位に2秒の配列があるのでさらに2で除算
                # https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467576#2605715
                r = int((row["min"] + row["max"]) // 4)

            # スペクトログラム(最初の4チャネル)
            for region in range(4):
                # spectrogram is 10mins i.e 600secs so 300 units, midpoint is 150 so 145:155 is 20secs
                img = self.specs[row.spectrogram_id][r : r + 300, region * 100 : (region + 1) * 100].T

                # img毎に対数標準化
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())

                img = (img - m) / (s + ep)
                img = np.nan_to_num(img, nan=0.0)

                # 時間軸に沿って、256time stepsにクロッピング
                """
                クロッピング:
                    画像や信号データから特定の領域を切り取る処理
                    サイズの統一や関心のある領域を強調するために使用されることが多い

                14は高さ方向のクロッピング
                    画像の上下から各14ピクセルを切り取る
                22は幅方向のクロッピング
                    画像の左右から各22ピクセルを切り取る

                いずれもHyperParameterなので、改善の余地あり
                """
                X[j, 14:-14, :, region] = img[:, 22:-22] / 2.0

            # EEG(次の4チャネル)
            # https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg
            img = self.eegs[row["eeg_id"]]
            X[j, :, :, 4:] = img

            if self.mode != "test":
                y[j,] = row[self.label_columns]

        return X, y

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

    def _augment_batch(self, img_batch):
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

        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                # albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
                albu.XYMasking(**params1, p=0.3),
                albu.XYMasking(**params2, p=0.3),
                albu.XYMasking(**params3, p=0.3),
            ]
        )

        return composition(image=img)["image"]
