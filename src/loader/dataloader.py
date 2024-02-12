from typing import List, Dict

import numpy as np
import polars as pl
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
        df: pl.DataFrame,
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
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self._generate_data(indices)

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
        # DEBUG: 8 -> 4
        X = np.zeros((len(indices), 128, 256, 4), dtype="float32")
        y = np.zeros((len(indices), len(self.label_columns)), dtype="float32")
        img = np.ones((128, 256), dtype="float32")

        for j, i in enumerate(indices):
            row = self.df.slice(i, 1)
            if self.mode == "test":
                r = 0
            else:
                min_value = row.select(pl.col("min")).to_numpy()[0, 0]
                max_value = row.select(pl.col("max")).to_numpy()[0, 0]

                r = int((min_value + max_value) // 4)

            # スペクトログラム(最初の4チャネル)
            for k in range(4):
                img = self.specs[row.get_column("spec_id").to_numpy()[0]][
                    r : r + 300, k * 100 : (k + 1) * 100
                ].T

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
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0

            # EEG(次の4チャネル)
            # TODO: EEGからスペクトログラムを作る必要あり
            # https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg
            # img = self.eegs[row.get_column("eeg_id").to_numpy()[0]]
            # X[j, :, :, 4:] = img

            if self.mode != "test":
                y[j,] = row[self.label_columns]

        return X, y

    def _augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self._random_transform(img_batch[i,])

        return img_batch

    def _random_transform(self, img):
        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                # albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
            ]
        )
