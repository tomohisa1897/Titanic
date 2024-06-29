<b>タイタニック号沈没の際の生存者を予想する</b>

Kaggleコンペ<br>
https://www.kaggle.com/competitions/titanic<br>
で、タイタニック号沈没の際の生存者を予想します。<br>

<b>データ解析について</b><br>
今回は決定木モデルを使用したベースラインの作成について記述しています。<br>
今後の課題として、LightGBMモデルを用いたパラメータ設定ができるようにしたいと思っています。<br>
また、検証方法はホールドアウト検証としましたが、クロスバリデーションも取り入れていきます。<br>

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



<CODE>
imp.sort_values("imp", ascending=False, ignore_index=True)
imp.sort_values("imp", ascending=False, ignore_index=True)
imp.sort_values("imp", ascending=False, ignore_index=True)
</CODE>
imp.sort_values("imp", ascending=False, ignore_index=True)
imp.sort_values("imp", ascending=False, ignore_index=True)
imp.sort_values("imp", ascending=False, ignore_index=True)

<b>課題</b><br>
今回は、IDと生存フラグを除く10属性のうち、2属性で評価しましたが、残り8属性についても追加して確認したいです。<br>
評価モデルは、LightGBMモデルを用いて、パラメータ設定ができるようにしたいと思っています。<br>
また、検証方法はホールドアウト検証としましたが、クロスバリデーションも取り入れてきたいと考えています。<br>
