<b>タイタニック号沈没の際の生存者を予想する</b>

Kaggleコンペ<br>
https://www.kaggle.com/competitions/titanic<br>
で、タイタニック号沈没の際の生存者を予想します。<br>

<b>データ解析について</b><br>
今回は決定木モデルを使用したベースラインの作成について記述しています。<br>
今後の課題として、LightGBMモデルを用いたパラメータ設定ができるようにしたいと思っています。<br>
また、検証方法はホールドアウト検証としましたが、クロスバリデーションも取り入れていきます。<br>

<b>環境</b>
【言語】Python<br>
【OS】Windows<br>
Google Colaboratory上でPython実行。<br>
GitHubのローカルリポジトリは、Visual Studio Codeで編集。<br>

<b>背景</b><br>
1912年4月15日、豪華客船タイタニック号が氷山とぶつかって沈没しました。<br>
乗客約2,000名のうち、2/3に及ぶ乗客が死亡した沈没事故です。<br>
すでに生死がわかっている学習用データを用いて、生死の不明な乗客リストの生死を予想します。<br>

乗客はそれぞれ以下のような属性を持っています。（一部欠損あり）<br>
PassengerId 乗客のID(ユニーク)<br>
Survived ⽣存フラグ（0=死亡、1=⽣存）<br>
Pclass チケットのクラス（1が最も良いクラス）<br>
Name 乗客の名前<br>
Sex 性別（male=男性、female＝⼥性）<br>
Age 乗客の年齢<br>
SibSp 同乗している兄弟/配偶者の数<br>
Parch 同乗している親/⼦供の数<br>
Ticket チケット番号<br>
Fare 料⾦<br>
Cabin 客室番号<br>
Embarked タイタニック号に乗った港<br>

<b>Titanicデータの構成</b><br>
gener_submission.csv : 乗客のIDと生死のリスト<br>
test.csv : 生死の不明な乗客リスト<br>
train.csv : 生死のわかっている乗客リスト<br>

<CODE>
#0.Titanicデータのマウント
from google.colab import drive
drive.mount('/content/drive')
<br>
#1.データ理解・可視化
#1.1 ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
<br>
#Numoy:行列計算や数値を扱う用のモジュール
#Pandas:表形式のデータを扱う用のモジュール
#Matplotlib:グラフ描画用モジュール
#Seaborn:Matplotlibよりもきれいなグラフを描画するためのモジュール
<br>
#1.2 データの読み込み
# PATHの設定
dir_path = '/content/drive/MyDrive/datascience-for-beginner/titanic/'
# 学習データの読み込み
train_df = pd.read_csv(dir_path + 'train.csv')
# テストデータの読み込み
test_df = pd.read_csv(dir_path + 'test.csv')
<br>
#1.3 欠損値の確認
#データ内の⽋損値を確認する
train_df.isnull().sum()
<br>
# 結果
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
<br>
# 学習用データtrain_dfでは、Age,Cabin,Embakredに欠損が認められる。
# 同様にテスト用データtest_dfでも、データ内の⽋損値を確認する
test_df.isnull().sum()
<br>
# 結果
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
<br>
# テスト用データtest_dfでは、Age,Cabinの他、Fareでも欠損が認められる。Embarkedでの欠損は無し。
<br>
#1.4 データの理解
# 今回はKaggleコンペなので、学習用データとテストデータが分かれているが、実際の現場ではデータに区別はないため、
# 2つのデータを結合する。

# 学習データとテストデータを連結する
df = pd.concat([train_df, test_df], ignore_index=True)
# 結合がうまくできているか確認する。
# 連結したデータの⼤きさを確認する
df.shape
# 結果
(1309, 12)
<br>
# 学習用データが891行、テスト用データが418行だったので結合に成功していると言える。(891+418=1309)
# テスト用データにはSurvivdedのカラムがないので、欠損しているはずである。本当に欠けているか確認する。
# 最後の5⾏を確認
df.tail()
# 結果
<img src="https://drive.google.com/file/d/1V6lXqNo4Xp1qKVg338wgU0tCIEVaqfFs/view?usp=sharing">
<br>






</CODE>

<b>課題</b><br>
今回は、IDと生存フラグを除く10属性のうち、2属性で評価しましたが、残り8属性についても追加して確認したいです。<br>
評価モデルは、簡易的な決定木モデルとしましたが、LightGBMモデルを用いて、パラメータ設定ができるようにしたいと思っています。<br>
また、検証方法はホールドアウト検証としましたが、クロスバリデーションも取り入れてきたいと考えています。<br>
