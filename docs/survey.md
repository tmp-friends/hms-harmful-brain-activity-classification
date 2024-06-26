## Survey

### HMS - Harmful Brain Activity Classification
#### Understanding Competition Data and EfficientNetB2 Starter - LB 0.43 🎉

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010


- Q. 106,800行のtrain.csvに対し、以下の件数しかない
  - unique `eeg_id`: 17089
  - unique `spectrogram_id`: 11138
  - unique `partient_id`: 1950
- A.
  - train.csvの各行は、一人の特定の患者からの`window of time`
  - 対応するEEGとspectrogramはparquetファイルで見つけられる

- EEGのtime window: 50s
- spectrogramのtime window: 600s
- 中央のイベント(10s間)を予測する必要あり

- KaggleDataset作った
  - https://www.kaggle.com/datasets/cdeotte/brain-spectrograms
  - 11138のスペクトログラムファイルを1つにまとめた
  - 一気にRAMに読み込んでおくことで、逐一ファイルを読み込むより高速になる

**所感**

- 図もついているしわかりやすい
  - フーリエ変換については、復習する
- 中央のイベントを予測するの？10sなのはなぜ？？

**備考**

EEGとスペクトログラムの関係性 by ChatGPT

```
EEG（脳波信号）からスペクトログラムへの変換は、信号の時間領域表現から周波数領域表現への変換プロセスです。この変換により、信号の周波数成分が時間とともにどのように変化するかを視覚的に表現できます。以下は、この変換プロセスの一般的なステップです。

### 1. 時間領域のEEG信号
EEG信号は、脳の電気活動を時間とともに記録したものです。これらの信号は、脳波の形で表され、通常はミリ秒単位で電圧の変化を示します。

### 2. 短時間フーリエ変換（STFT）
スペクトログラムを生成するためには、まずEEG信号に短時間フーリエ変換（STFT）を適用します。STFTは、信号を短いセグメントに分割し、各セグメントにフーリエ変換を適用して周波数成分を抽出します。これにより、各セグメントの周波数成分とそのセグメントが発生した時間が得られます。

### 3. パワースペクトルの計算
各セグメントのフーリエ変換から、パワースペクトルを計算します。パワースペクトルは、各周波数成分の強度（パワー）を示し、信号のエネルギーがどの周波数に集中しているかを示します。

### 4. スペクトログラムの生成
STFTから得られたパワースペクトルを時間軸に沿って並べることで、スペクトログラムが生成されます。スペクトログラムは2次元の画像で、横軸が時間、縦軸が周波数、色の濃淡（または色相）がパワースペクトルの強度を示します。

### 5. データの視覚化
最終的に、スペクトログラムを用いて、EEG信号の時間的および周波数的特性を視覚的に分析できます。たとえば、睡眠段階、瞑想の状態、または特定の認知タスクを実行している間に脳活動がどのように変化するかを調べることができます。

このプロセスは、Pythonの`matplotlib`や`scipy`ライブラリを使用して実装できます。特に`scipy.signal.spectrogram`関数は、EEGデータから直接スペクトログラムを生成するのに便利です。
```

時間と周波数の関係性 by ChatGPT

```
時間と周波数の関係性は信号処理や音響学、電磁波の研究など多くの分野で基本的な概念です。これら二つの概念は、信号や波形を解析する際に互いに補完的な情報を提供します。以下に、時間と周波数の関係性について基本的な説明をします。

### 時間ドメイン
時間ドメインにおける信号は、時間の経過に伴う振幅（強度や電圧など）の変化を表します。例えば、音声信号や心電図（ECG）は時間ドメインで表される典型的な例です。これらの信号では、時間が進むにつれて振幅がどのように変化するかを観察できます。

### 周波数ドメイン
周波数ドメインにおける信号は、信号が含む周波数成分とその強度（またはエネルギー）を表します。信号を周波数ドメインに変換することで、その信号がどの周波数を含んでいるか、各周波数成分の強度はどの程度かを知ることができます。この変換はフーリエ変換によって行われます。

### 時間と周波数のトレードオフ
時間ドメインと周波数ドメインの間には、トレードオフの関係があります。信号をより詳細に時間的に解析するほど、周波数に関する情報は曖昧になります。逆に、信号を周波数成分に分解して詳細に周波数情報を得るほど、それがいつ発生したのかについての時間的な情報は失われます。これは不確定性原理として知られており、信号処理において重要な考慮事項です。

### 時間-周波数解析
時間と周波数の情報を同時に得るために、短時間フーリエ変換（STFT）やウェーブレット変換などの時間-周波数解析手法が用いられます。これらの手法では、信号を時間的に局所化された小さなセグメントに分割し、各セグメントに対してフーリエ変換を適用することで、時間と周波数の情報を同時に表現します。

### まとめ
時間と周波数の関係性は、信号をどのように解析するかに基づいて異なる視点を提供します。時間ドメインでは信号の時間的変化を、周波数ドメインでは信号の周波数成分とその強度を捉えることができます。時間-周波数解析手法を用いることで、これら二つのドメインの情報を組み合わせて信号をより深く理解することが可能になります。
```

- フーリエ変換
  - パラメータ
    - time window
      - 信号をどれだけの長さで区切るか
      - windowが長いほど周波数解像度は高くなるが、時間解像度は低くなる
      - 今回は600s
    - overwrap
      - 連続する時間窓がどれだけ重なるか
      - overwrapを多くすると、時間解像度は高くなるが、計算量は増える
        - 今回はなさそう(by 別のDiscussionより)
    - window func
      - 各time windowに適用する関数で、信号の端における不連続を減少させるために使用

#### There are ~2640 unique EEG ids in the hidden test data!

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/471287

- subしてかかった時間を調べることでprivateのunique eeg_idを調べた
- 1つのidあたり5sのsleepを入れていた
- eeg_idは2640ほどあると考えられる

#### How To Make Spectrogram from EEG

https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg

- EEGのraw dataからspectrogramを生成する方法

#### Magic Formula to Convert EEG to Spectrograms!

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/469760

#### LB probing results in HMS-HBAC

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/471890

LB probingの結果

- test data内にeeg_idの重複はない
- test data内にspectrogram_idの重複はない

#### EfficientNetB0 Starter - [LB 0.43]

- baseline

#### Grad Cam - What is important in Spectrograms?

https://www.kaggle.com/code/cdeotte/grad-cam-what-is-important-in-spectrograms

-

#### UPDATED - CatBoost Starter Notebook and Kaggle Dataset - LB 0.60

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467576

- notebook: https://www.kaggle.com/code/cdeotte/catboost-starter-lb-0-60?scriptVersionId=159895287
- スペクトログラムデータのみのCatBoostでCV0.74, LB0.60
- uniqueな`eeg_id`に対応するスペクトログラムの中央10分から特徴量を生成
  - 10分間のスペクトログラムは300回(2秒ごとに測定しているため)
  - 脳の4象限から100の周波数について測定 -> 400の時系列の時間平均
- LBの向上させ方
  - スペクトログラムからさらに特徴を作れる
  - 各eeg_idに対して、10（または任意の）サンプルを作成し、各サンプルが異なる10分間のspectrogram windowを使うことも可能
    - ランダムクロップデータの増強のようなイメージ
    - これにより10倍のデータが作成される
  - eeg parquetsから特徴量を生成することもできる

- Comments
  - GroupKFoldとStratifiedGroupKFoldの使い分け
    - StraitifiedGroupKFoldを使うとき
      - 1つのターゲットクラスが非常に稀で、trainとvalidにそのクラスを必ず含めたい
      - testがtrainと同じ割合のターゲットクラスを持っている
    - 今回は、testがtrainと同じ割合を持つかはわからないので、GroupKFoldを用いている
    - GroupKFoldのほうが未知のtest比率に対して、わずかにうまく汎用化できる

**所感**
- スペクトログラムからの特徴量の作り方や、GroupKFoldとStratifiedGroupKFoldの使い分け方などを学べた

#### 🧠📈 Beginner's EDA 📈🧠

https://www.kaggle.com/code/clehmann10/beginner-s-eda

- train.csvなどの統計データが詳しく載っている
- あまり読み込めてないので、後で読む

#### EDA Train.csv

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467021

- testはtrainに近い
  - 6つのターゲットクラスに対して、重複を許さないほうがLBが良い
- testはpatient_id毎に重複をしないEEG配列が繰り返される

- Comments
  - trainでは重複するeeg_idを持っているのに対し、testは重複しない
  - trainでは重複するpatient_idを持っているのに対し、testは重複しない

#### Kullback Leibler Divergence Applications, Limitations and KL Divergence on Kaggle.

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/466731

-

#### Adjutant resources to refer

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/466721

#### Previous Competitions Top Solutions (May help in this competition too)

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467979

- 過去の類似コンペの紹介
  - BirdCLEF 2023
  - University of Liverpool - Ion Switching

#### Papers & Model Architectures

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468771

- CV系のDeepモデルの論文
  - 1D CNN
  - CNN+RNN
  - CNN+Bi-LSTM
  - Bi-LSTM+ATT


#### Resources for getting started with Heart Data

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/467129

- ECG(心電図)の主な特徴量
  - QRS complexes
  - R peaks
  - R-R intervals

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2030387%2F1fdc5942d17ce7b15801884d74241d66%2FScreenshot%202024-01-11%20093911.png?generation=1704962385485913&alt=media" width="400px" />

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2030387%2F8de171d5fde4afbadc1691ababdde3d8%2Fecg-morphology.png?generation=1704962638836359&alt=media" width="600px" />

**所感**
- 心電図の話なので、あまり関係なさそうではある


#### Resources for getting started with Brain Data - EEG

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/466768

- Youtube(Time and Frequency domain of EEG by Mike X Cohen)の解説がわかりやすかった
  - Frequency domainにすることで1sに何回山が出るかをあらわせる
  - Time domainではnoiseがのっていてもFrequency domainにすることで省けるようになる

**所感**
- 脳波に関するドメイン知識が皆無なので助かる
  - 大学で脳波のことをあつかっている研究室があった気がする


#### XYMasking aug

https://www.kaggle.com/code/iglovikov/xymasking-aug

- albuのXYMaskingの使い方

#### TTA not helping?

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/480775

- TTA=TestTimeAugmentation
  - LBが0.02ほど改善したらしい

- LB4位の人曰く、
  - data augmentationにおいては、mixupのみ有効らしい
  - それ以外は、negativeな作用
  - time-frequency maps(spec)とrawEEGを組み合わせて、1つのモデルの複数の入力としているそう

- 入力の種類を増やせば、モデルの性能が上がりそう

#### Need for New Insights

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/480674

- baseline: 0.43
- mixup: 0.41
- vit: 0.4
- XYMasking: 0.39
- 2stage training: 0.35
- Ensemble: 0.34

**所感**
- ViT使う
  - ensembleにも使えそう
- LB0.2xのレートはself-attention layerを組み込んでいるのでは、という意見もあった

#### EffNetB0 model trained twice, once for each of two training populations - [LB 0.39]

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/477135

- 各label_idの総得票数に関して、training dataが異なる2つのソースから集約された可能性がある
  - 各グループ内のユニークなクラスの不均衡もある
- 1票しか入っていないdatasetのシナリオを考えると、モデルが自信過剰にクラスを予測し、残りの確率を0にする可能性がある
  - KL-divergenseも悪化
- two-stage training
  - First stage: 1-7票のデータ
    - ピーク分布からの学習の可能性を利用
  - Second stage: 10-28票のデータ
    - KL-divergenseの問題を緩和
- アプローチの改善
  - cdeotteさんの提案する50sのsampleを中心にEEGスペクトログラムを生成する方法から離れ、かわりに`eeg_label_offset_seconds`に基づいて、各label_id毎に異なるspectrogramを生成する方法を選択
  - ニュアンスの異なるバリエーションを持つ、データセットを得られた

**所感**

> cdeotteさんの提案する50sのsampleを中心にEEGスペクトログラムを生成する方法

この手法より有効な手はありそうな気はする

#### Hard samples are more important - [LB 0.37]

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/477461:

- Motivation: 分布にピークがあるsampleはモデルが誇張された信頼度で予測することになり、パフォーマンスが低下する
- Method: two-stage training

しかし、得票数と得票分布の間に相関関係はない

- First stage: training with all data
- Second stage: training with samples (KL Loss < 5.5)

**所感**

- この分布はtestデータでも有効なのか

#### Proper Augmentations is a Key!

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/479776

- data augmentationで効いたものや効かなかったことが書かれている
- 効いたこと
  - XYMasking
  - VerticalFlip
- 効かなかったこと
  - HorizontalFlip
  - Brightness contrast
  - rotation
  - cropping

- Commentにて
  - mixupで0.02ほど改善した方もいた
  - XYMaskingのparameterのさじ加減が難しい
    - 複数の小さなマスクをかけたほうがいい感じになる

#### How are your experience with Augmentation ?

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/469953

- mixupが効いている
  - alpha > 1だと効かなかった人いる
- brightness/contrastが効いた人もいる

**所感**

- mixupをalpha=0.5で試す
- brightness/contrastを試す

#### HMS EfficientNetB0 Train

https://www.kaggle.com/code/medali1992/hms-efficientnetb0-train

- 公開notebookで（アンサンブルでなく）LB0.36
- 差分を確認

```
先日、@nischaydnkが作成したこのノートブックを改良したディスカッションを共有した。私が使ったトリックは、特にオプティマイザを変更したこと（私はAdanを使った）、そして@seanbeardenがここで述べているように2段階トレーニングを使い、ダウンサンプリングを使わなかったことで、かなりうまくいった。しかし、2段階トレーニングにはデータ漏洩という問題があります。klはミスラベルをしてもモデルにほとんどペナルティを与えないからだ。そのため、以前の実験では2つのデータセットに対して2回のgroupkfold CVを行ったが、投票数の少ないサンプルはどちらのデータセットにも存在した。

そこでこのノートでは、1つのCVスキームを使い、各ステージでデータをフィルタリングし、両方の母集団を含むデータで検証することで、データの漏れを防いでいる。このアプローチが正しいかどうか、コメントで教えてください。
```

#### Patient Variation - EDA

https://www.kaggle.com/code/pcjimmmy/patient-variation-eda

- patient_idに基づいたEDA

**所感**
- データの作り方を試行するときに使えるかも

---

### atmaCup

#### DataAugmentationどれ使おう？ってときに役立つページなど

https://www.guruguru.science/competitions/17/discussions/9382fbc5-73b6-47b9-8571-77ed3fd8763b/
https://www.guruguru.science/competitions/17/discussions/ab9515a4-3099-4405-9bba-dcf1529191ca/

- CNNの学習において、（特にデータが少量のときに）DataAugmentationは不可欠
  - 過学習緩和
- DataAugmentationの種類
  - 画像に変動を与えるもの -> 認識器が受け取る画像で起こりうる変動に対してロバストにしたい
    - HorizontalFlip
    - VerticalFlip
    - ShiftScaleRotate
    - RandomResizeCrop
  - 情報を欠落させるもの(random erasing) -> 過学習緩和
    - CutOut
    - RandomErasing
  - 情報を混ぜるもの(mixup系) -> 過学習緩和
    - MixUp

**所感**
- 情報を欠落させると、波形のPeakなどの情報がなくなり精度が悪くなる気がする
- mixupにおいて、以下の記述が気になった

> 概念的には二つの画像をランダムな重みで混ぜ合わせると共にそのラベルとして同じ重みで平均したものを使うのですが(※実際はそれぞれのラベルで Loss を計算して重み付き平均する場合が多い)、 今回のタスクだと 0 (16世紀以前) と 2(18世紀) の画像を混ぜたからといってラベルが 1 (17世紀) になるとはとても思えません。

---

### Perfinder

#### 【kaggle】うっかり金メダルを獲得してしまったPetfinderコンペの取り組みまとめ & 5th Place

https://cpptake.com/archives/923
https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/300928

- Overview
  - 1st : Make a lot of single models. (our final submittion has 14 single models)
  - 2nd : Calculate similarity score to detect the same images, previous competiton's train+test images (about 70000 images) and this competition's images.
  - Final : On overthreshold data, We can use previous competitions meta data. So, we made models for each adoptionspeed as post proccessing.

- data augmentation

```
'train' : A.Compose([
    A.OneOf([
        A.RandomResizedCrop(CFG.IMG_SIZE, CFG.IMG_SIZE, p=0.3, scale=(0.85, 0.95)),
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, p=0.4),
        A.Compose([
            A.Resize(int(CFG.IMG_SIZE * 1.5), int(CFG.IMG_SIZE * 1.5), p=1.0),
            A.CenterCrop(p=1.0, height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
        ], p=0.3),
    ], p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
], p=1.0),
'valid' : A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0,),
], p=1.0),
```

- stacking
  - BayesianRidge

#### 6th Place - Multitask Learning

https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/301015

- chris deotteさん

---

### BirdCLEF 2023

####

- FocalLoss

---

###

---

### CMI - DetectSleep
