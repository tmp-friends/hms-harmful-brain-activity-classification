## EXP
### EfficientNetB0 Starter - [LB 0.43] (cdeotteさん)

https://www.kaggle.com/code/cdeotte/efficientnetb0-starter-lb-0-43

- CV: 0.6604, LB: 0.48
    - LB0.43のはずなのにスコアが悪化？？
- 原因
    - タイポしてた
        - dataloaderの_random_transformのreturnがなかった
        - データの変形(augumentation)でここまでスコアに差ができることがわかったのは収穫
    - EEG spectrogramを入れていなかった

```py
def _random_transform(self, img):
    composition = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            # albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        ]
    )

    return composition(image=img)["image"]
```

```
CV: 0.5854
LB: 0.43
```

### EPOCH 4 -> 10

```
CV: 0.589
LB: -
```

- EPOCH数などのパラメータ調整は効かなそう
- モデルもEfficientNetB0で良いように思える
  - ResNetなどもあるが、最近のCV系のモデルはどれも高性能なので、そこでの差はつきづらいと考える
- データの前処理、後処理を重点的にやるのがいいかもしれない

### Augmentation
#### XYMasking x3 only

```
CV: 0.5823
LB: 0.43
```
- 0.00x 改善

#### XYMasking x3 + HorizontalFlip

```
CV: 0.5751
LB: 0.43
```

- 改善していそう

#### XYMasking x3 + HorizontalFlip + VerticalFlip

- HorizontalFlip p=0.5
- VerticalFlip p=0.5

```
CV: 0.5947
LB: -
```

- HorizontalFlip p=0.3
- VerticalFlip p=0.3

```
CV: 0.5845
LB: -
```

- HorizontalFlip p=0.2
- VerticalFlip p=0.2

```
CV: 0.5933
LB: 0.42
```

- augmentationをしすぎると上手く学習ができなくなりそう
    - epoch数を増やせば、精度が出たりする？
- CVとLBが厳密に連動しているわけではないので、CVを優先して信じる

#### CutMix

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/479446

- HolizontalFlip(p=0.3)

```
CV: 0.5174
LB: 0.45
```

- HorizontalFlip, VerticalFlip(p=0.3)

```
CV: 0.5116
LB: 0.46
```

#### Mixup

```
CV: 0.6331
LB: 0.47
```

#### image_sizeを変える？

128x256x8 -> 512x512x3

### TODO

- Discussion読み
- 過去の類似コンペ読み
- CV系モデルの調査(論文読み)
    - CNN
        - ConvLayer
    - ResNet
