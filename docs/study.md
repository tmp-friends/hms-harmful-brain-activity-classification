## Study

### EfficientNet

#### 要約

- CNNにおいてモデルをスケールアップする方式であるCompound Model Scalingを提案
    - GridSearchによるスケーリング係数を決め、モデルをスケールアップする関係式を提供
- チューニングの手間を省きつつ、精度の良いモデルを作れるようになった

#### 背景

- 従来のCNNにおいて、次の３つの中から１つを選んでモデルをスケールアップさせる
  - ネットワークの層
  - ネットワークの幅(Channel数)
  - 入力画像の解像度

- ２つ以上を選んでモデルをスケールアップしようとすると、選択自由度が増える分、探索空間が広まるため、チューニングに手間がかかる問題

#### 提案手法

- Compound Model Scaling
  - 探索したスケーリング係数をもとに、Baseline Model(ResNet, MobileNetなど)をスケールアップさせることができる
  - しかし、本論文では、新たに設計したEfficientNetをBaseModelとして採用
    - NAS(=Neural Architecture Search)によって最適化されたネットワーク
        - NAS: 強化学習で探索空間から最適なモデルを自動的に見つける

#### 所感

- HyperParameterをGridSearchによってモデル側でよしなにやってくれるようにした、という理解でいる

#### 参考

https://nuka137.hatenablog.com/entry/2020/09/09/075422o

https://ai-scholar.tech/articles/image-recognition/regnet

https://qiita.com/omiita/items/83643f78baabfa210ab1

### ResNet

#### 要約

- ResidualBlockを直列に多数繋げて、残差の系列をモデル化することにより、高精度の深いCNNを学習できるようになった

#### 背景

- 当時のSoTAなCNNでも20層以上の学習が上手くいかなかった
  - 劣化(=degradation)問題
  - 勾配消失問題

#### 提案手法

- ResidualBlockの導入によるResidualLearningにより、深いCNNの学習方法を提案
    - 50層~150層でも劣化問題無しに学習可能となった
- 分散して各残差に逆伝播するので、高速かつ収束性も良くなった

- キモとしては、残差ありとなしによる分散的でアンサンブル的な学習

- ResidualBlock
    - $H(x) = F(x) + x$

#### 所感

- 今日の大規模モデルを支えるTrick
- 応用の範囲が広い
- TransformerでもResidualBlockがあったりするが、これが元ネタ？
- EfficientNetよりは前に発表されたモデル

#### 参考

https://cvml-expertguide.net/terms/dl/cnn/cnn-backbone/resnet

### KL Div

- 2つの確率分布を比較するのに使われる
- どれくらいの情報量が失われたかを計算
    - 観測データの分布$p(x)$のエントロピーから推定モデル$q(x)$のエントロピーを引く

#### 参考

https://qiita.com/shuva/items/81ad2a337175c035988f

### Data Augmentation(データ拡張)

- 手元にあるデータに何らかの処理をして、データのバリエーションを増やすこと
  - 学習データが増えるので、より汎用的なモデルを構築できる
  - 推論時にもバリエーションに富んだ推論を行える
  - アンサンブルしてもいい（Augmentationなしとありで）

- よく使われるライブラリ
    - torchvision
    - albumentations

#### Albumentations
公式Doc: https://albumentations.ai/

albuでの処理例が載っている
https://qiita.com/kurilab/items/b69e1be8d0224ae139ad

#### Cutout, Mixup

- 参考実装
  - https://www.kaggle.com/code/kaushal2896/data-augmentation-tutorial-basic-cutout-mixup
  - https://www.kaggle.com/code/prajeshsanghvi/mixup-cutmix-albumentation

#### 参考

https://qiita.com/Takayoshi_Makabe/items/79c8a5ba692aa94043f7

