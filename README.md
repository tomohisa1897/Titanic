<b>タイタニック号沈没の際の生存者を予想する</b>

Kaggleコンペ
https://www.kaggle.com/competitions/titanic
で、タイタニック号沈没の際の生存者を予想します。

データ解析について
今回は決定木モデルを使用したベースラインの作成について記述しています。
今後の課題として、LightGBMモデルを用いたパラメータ設定ができるようにしたいと思っています。
また、検証方法はホールドアウト検証としましたが、クロスバリデーションも取り入れていきます。

背景<br>
1912年4月15日、豪華客船タイタニック号が氷山とぶつかって沈没しました。
乗客約2,000名のうち、2/3に及ぶ乗客が死亡した沈没事故です。
すでに生死がわかっている学習用データを用いて、生死の不明な乗客リストの生死を予想します。

乗客はそれぞれ以下のような属性を持っています。（一部欠損あり）
PassengerId 乗客のID(ユニーク)
Survived ⽣存フラグ（0=死亡、1=⽣存）
Pclass チケットのクラス（1が最も良いクラス）
Name 乗客の名前
Sex 性別（male=男性、female＝⼥性）
Age 乗客の年齢
SibSp 同乗している兄弟/配偶者の数
Parch 同乗している親/⼦供の数
Ticket チケット番号
Fare 料⾦
Cabin 客室番号
Embarked タイタニック号に乗った港



<CODE>
imp.sort_values("imp", ascending=False, ignore_index=True)
</CODE>
imp.sort_values("imp", ascending=False, ignore_index=True)
