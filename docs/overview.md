# Overview

EGG(脳波)から発作やその他の種類の有害な脳活動を検出・分類する

- 検出対象
  - SZ: seizure(発作)
  - LPD: lateralized periodic discharges(側方化周期性放電)
  - GPD: generalized periodic discharges(汎発性周期性放電)
  - LRDA: lateralized rhythmic delta activity(側方化リズミックデルタ活動)
  - GRDA: generalized rhythmic delta activity(汎発性リズミックデルタ活動)
  - other

- アノテーション
  - EGGセグメントのアノテーションは、専門家グループによって行われた
    - 手作業のため、セグメントによって専門家間で一致したり、そうでなかったりする
  - パターン
    - 専門家間で高いレベルで一致するセグメント -> "idealized"パターン
    - 1/2の専門家が"other"をつけ、1/2の専門家が5つのラベルのいずれかをつけたセグメント -> "proto pattern"
    - 専門家間で5つのラベルのうち、ほぼ二分に分かれるセグメント -> "edge cases"

- submission
  - egg_idに対し、6つの検出対象の確率

## Domain info

https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey

- EEG: Electroencephalography(脳波信号)
  - 頭皮に電極を設置し、脳細胞の活動から生じる微小な電荷を検出する
  - 電極が捉えた信号は増幅されて記録

- 観測されるEEGのパターン
  - Wave Patterns: 異なる種類の波形によって特徴づけられる
    - e.g. alpha, beta, delta, theta
      - alpha -> リラックスしている状態
      - beta -> 集中している状態
    - それぞれの種類の波形が異なる脳の状態に対応する
  - Amplitude and Frequency: 波形の特徴を表す
    - amplitudes: 振幅・波の高さ
    - frequency: 振動数・波の速さ
  - Artifacts: 脳以外の波形
    - 筋肉の動き、目の瞬き、電気的干渉によって現れる可能性がある
  - Abnormal Patterns: 異常な脳活動を示すパターン
    - てんかんのような神経疾患の場合

- spectrograms
  - EEGとの関係性
    - EEGの周波数スペクトルを経時的に分析・可視化するという点で密接な関係がある
  - ユースケース
    - Frequency Analysis(周波数分析)
      - EEGは異なる周波数を持つ脳波で構成されている
      - spectogramはこれらの周波数を視覚的に表示し、EEG記録中の時間経過に伴う変化を示せる
    - Identifying Patterns(パターンの特定)
      - 生のEEGでは用意に識別できないようなパターンを識別するのに役立つ
      - e.g. 異なる睡眠段階における脳活動の変化を検出、特定の神経疾患に関連する振動活動を特定
    - Temporal and Frequency Resolution(時間分解能と周波数分解能)
      - 脳活動のダイナミックな変化を捉えられる
    - DataVisualization
      - 複雑なEEGデータをより直感的な形にできる
      - EEGのtime-domainデータをよりわかりやすいfrequency-domainデータに変換できる

- LPD/GPD/LRDA/GRDA
  - LPD: Lateralized Periodic Discharges(側方化周期性放電) - 一般的に急性または亜急性の脳機能障害と関連しており、多くの場合、脳の構造的病変や急性脳損傷と関連しています。LPDは神経学の分野で特に注目されており、発作や精神状態の変化などの急性神経症状を持つ患者でしばしば調査されます。
    - Lateralization(側方化)： LPDは脳の片側(半球)に多く発生する
    - Periodicity(周期性)： LPDは周期性があり、繰り返しパターンを示す。この周期性は、脳波でLPDを識別する際の重要な特徴である。
    - Waveform Characteristics(波形の特徴)： LPDは通常、背景脳波活動から明確に区別できる鋭い波形または複合波からなる。これらの鋭い波の後には通常、徐波成分が続く。
    - Genearl Characteristics(一般的特徴)： 脳波データでは、LPDは規則的で輪郭のはっきりした波形として現れ、背景の脳活動の中で際立っており、一定の間隔で繰り返される。LPDは通常片側性で、左半球か右半球のどちらかに影響を及ぼす。

  - GPD: Generalized Periodic Discharges(汎発性周期性放電) - びまん性または全身性の脳機能障害と関連することが多い脳波パターンである。これらの放電は、中毒性代謝性脳症から重度のびまん性脳損傷に至るまで、さまざまな神経学的状態と関連している可能性がある。GPDは、臨床神経生理学および神経学において、特に意識変容や昏睡状態の患者の診断および管理という観点から、重要な関心を集めている。
    - Generalized Distribution(汎化分布)： GPDは、脳の片側に限局するのではなく、両半球にわたって分布しているのが特徴である。
    - Periodicity(周期性)： LPDと同様に、GPDは繰り返しパターンを示す。その周期性は脳波で識別するために重要であり、他の全般性脳波異常と区別される。
    - Waveform Characteristics(波形の特徴)： GPDは通常、反復する鋭い輪郭の波形からなり、その形状や持続時間はさまざまである。他の脳波パターンと比較して、より同期していることが多い。
    - Clinical Implications(臨床的意義)： GPDは、重度のびまん性脳損傷、低酸素性虚血性脳症、ある種の薬物中毒や代謝異常など、さまざまな臨床場面でみられる。その存在は、重篤な脳機能障害が根底にあることを示すことがある。

  - LRDA: Lateralized Rhythmic Delta Activity(側方化リズミカルデルタ活動) - リズミカルな徐波活動を特徴とする脳波パターンで、通常デルタ周波数域にあり、片側の半球に限局している。
    - Lateralization(側方化)： LRDAの主な特徴は、脳の主に片側の半球に影響を及ぼすという側在性であり、これは神経学的病変や機能障害を特定する上で極めて重要である。
    - Rhythmic Delta Waves(リズミカルなデルタ波)： LPDのシャープな波形とは異なり、LRDAは、主にデルタ周波数領域（1～4Hz）の、より滑らかでリズミカルな波形によって定義される。
    - Clinical Context(臨床的背景): LRDAは、脳卒中、腫瘍、炎症などによる局所的な脳病変を有する患者でしばしば観察される。また、焦点性発作に伴ってみられることもある。
    - Interpretation(解釈): 脳波測定におけるLRDAの存在は、脳の病変の場所や性質に関する貴重な情報を提供し、診断や治療計画の助けとなる。

  - GRDA: Generalized Rhythmic Delta Activity(汎発性リズミックデルタ活動) - 脳の両半球に一様に分布するリズミックデルタ活動を特徴とする脳波パターン
    - Generalized Distribution(一般化された分布)： GRDAがLRDAと異なる点は、側方性ではなく、両半球を含み、多くの場合左右対称であることである。
    - Rhythmic Delta Waves(リズミカルデルタ波)： このパターンは、連続的または準連続的なリズミックデルタ波によって定義される。GPDと比較すると、活動はより緩慢でリズミカルである。
    - Associated Conditions(関連疾患)： GRDAは、さまざまな病因の脳症（中毒性代謝障害など）や、場合によっては特定の睡眠段階やびまん性脳障害など、さまざまな臨床症状でみられる。
    - Diagnostic Significance(診断上の意義)： GRDAの存在は、全体的な脳機能障害を示すことがあり、その原因を特定するためにさらなる調査が必要となることがある。GRDAは、意識レベルの変化またはびまん性神経障害を有する患者の評価において特に重要である。

- Hostの上げている論文: https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf


## Data

- train.csv
  - eeg_id: 脳波ID (unique)
  - eeg_sub_id: labelが適用される特定の50sのsubsample ID
  - eeg_label_offset_seconds: 連結EEGの開始からこのsubsampleまでの時間
  - spectrogram_id: スペクトログラムID (unique)
  - spectrogram_sub_id: labelが適用される特定の10minのsubsampleID
  - spectrogram_label_offset_seconds: 連結spectrogramの開始からこのsubsampleまでの時間
  - label_id: ラベルID
  - patient_id: 患者ID
  - expert_consensus: 専門家の合意 - 便宜上記載とあるので、あまり重要視しなくて良さげ
  - (seizure|lpd|gpd|lrda|grda|other)_vote: 投票数

- test.csv
  - eeg_id
  - spectrogram_id
  - patient_id

- train_eegs/
  - EEGデータ
  - train.csvのメタデータと紐づけて使う
  - 列名はEEGの電極位置の名前
    - EKG列のみは、心臓からのデータを記録する心電図リードのもの
  - 200サンプル/s

- train_spectrograms/
  - スペクトログラムデータ
  - train.csvのメタデータと紐づけて使う
  - 列名はHz単位の周波数とEEG電極の記録領域
    - LL=左外側、RL=右外側、LP=左矢状面、RP=右矢状面
