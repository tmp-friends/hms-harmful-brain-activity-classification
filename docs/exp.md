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
#### XYMasking

https://www.kaggle.com/code/iglovikov/xymasking-aug

- XYMasking x3
    - 0.00x 改善

```
CV: 0.5823
LB: 0.43
```

- XYMasking x3
- HorizontalFlip(p=0.5)
    - 改善していそう

```
CV: 0.5751
LB: 0.43
```


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

**所感**
- augmentationをしすぎると上手く学習ができなくなりそう
    - epoch数を増やせば、精度が出たりする？
- CVとLBが厳密に連動しているわけではないので、CVを優先して信じる

#### CutMix

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/479446

- cutmix
- XYMasking
- HolizontalFlip(p=0.3)

```
CV: 0.6174
LB: 0.45
```

- cutmix
- XYMasking
- HolizontalFlip(p=0.3)
- VerticalFlip(p=0.3)

```
CV: 0.6116
LB: 0.46
```

- cutmix
- XYMasking
- HolizontalFlip(p=0.5)
- VerticalFlip(p=0.5)

```
CV: 0.6225
LB: -
```

#### ShiftScaleRotate

- cutmix
- XYMasking
- HolizontalFlip(p=0.5)
- VerticalFlip(p=0.5)
- ShiftScaleRotate(p=0.3)

```
CV: 0.6249
LB:
```

#### Mixup

- cutmix
- XYMasking
- HolizontalFlip(p=0.5)
- VerticalFlip(p=0.5)
- Mixup(alpha=2.0)

```
CV: 0.6331
LB: 0.47
```

- cutmix
- XYMasking
- HolizontalFlip(p=0.5)
- VerticalFlip(p=0.5)
- Mixup(alpha=0.5)
    - 処理を最後に回す

```
CV: 0.6420
LB: -
```

- Mixup(alpha=0.5)
- cutmix
- XYMasking
- HolizontalFlip(p=0.5)
- VerticalFlip(p=0.5)

```
CV: 0.6186
LB: -
```

- Mixup(alpha=0.5)
- XYMasking
- HolizontalFlip(p=0.5)
- VerticalFlip(p=0.5)

```
CV: 0.5909
LB: -
```

#### RandomBrightnessContrast

- cutmix
- XYMasking
- HolizontalFlip(p=0.5)
- VerticalFlip(p=0.5)
- RandomBrightnessContrast(p=0.3)

```
CV: 0.6151
LB: -
```

p=1.0でbrightnessとcontrastを識別しやすいように効かせるとどうなる？

#### GaussNoise

#### 結果

- HorizontalFlip p=0.5
- XYMasking x3 p=0.3

### image_sizeを変える？

128x256x8 -> 512x512x3

### TODO

- two stage training

- ViT
- ResNet
- CNN
- スタッキング(Regression)

- CV系モデルの調査(論文読み)
    - CNN
        - ConvLayer
    - ResNet
