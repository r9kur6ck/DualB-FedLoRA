# DualB-FedLoRA / SPSA-FedLoRA 実験リポジトリ

このリポジトリは、LoRAを用いた連合学習において、サーバ側のB行列を強化学習（SPSA）または勾配ベースで更新する手法を実装し、その貢献度（Shapley値）を検証するための実験コード群です。

各スクリプトは、実験の進展に伴うアーキテクチャの変遷を示しています。

---

## 1. `main_loo.py` (Leave-One-Out 検証)

* **略称:** `LOO` = **L**eave-**O**ne-**O**ut
* [cite_start]**手法:** `Dual-B FedLoRA` [cite: 6]
    * [cite_start]クライアントは `B_local` と `A_i` の両方をローカル学習します [cite: 12-13]。
    * [cite_start]サーバは `B_server` をSPSA（強化学習）で更新します [cite: 21]。
* [cite_start]**検証方法:** **Leave-One-Out (LOO)法** [cite: 121-127]
    * [cite_start]クライアントを1人ずつ除外して(N+1)回の**再学習**を行い、実際の貢献度 $\Delta_i$ を測定します [cite: 124-125]。
    * [cite_start]計算したShapley値 $\phi_i$ と $\Delta_i$ の相関を比較します [cite: 127]。
* **ステータス:** **アーカイブ済み (非推奨)**
    * (N+1)回の再学習コストが現実的ではないため、この検証アプローチは破棄されました。

---

## 2. `main_gdp.py` (Gradient-based Proxy 検証)

* [cite_start]**略称:** `GDP` = **G**radient-**b**ased **P**roxy (※PDF原文[cite: 231])
* [cite_start]**手法:** `Dual-B FedLoRA` [cite: 6]
    * 学習アーキテクチャは `main_loo.py` と同一です。
* [cite_start]**検証方法:** **Gradient-based Proxy (GDP/GBP)法** [cite: 231]
    * [cite_start]LOOの再学習コストを回避するため、軽量な代理指標（Proxy） $C_i$ （勾配の内積）を導入しました [cite: 238, 261]。
    * `B_server` を使う $\phi_i$ と、`B_local` を使う $C_i$ の相関を比較します。
* **ステータス:** **失敗（学習破綻）**
    * `Global Test Accuracy` が**10%（ランダム）**から上昇しませんでした。
    * これは `Dual-B` アーキテクチャの根本的な**不整合**（$A_i$ の学習相手 `B_local` と、評価相手 `B_server` が異なる）が原因であると特定されました。

---

## 3. `main_ospsa.py` (Only-SPSA-FedLoRA)

* **略称:** `SPSA` = **S**imultaneous **P**erturbation **S**tochastic **A**pproximation
* **手法:** `SPSA-FedLoRA`（**解決策1**の実装）
    * [cite_start]`main_gdp.py` の学習失敗（不整合）を解決するため、**`B_local` を完全に廃止**しました [cite: 173, 179-180]。
    * クライアントは、$A_i$ を学習する時点からサーバの `B_server`（の摂動 $B_+$ または $B_-$）と協調して学習します。
* **検証方法:** Gradient-based Proxy (GDP/GBP)法
* **ステータス:** **学習成功（ただし不安定）**
    * `Global Test Accuracy` が**10%の壁を突破**し、**28%**まで上昇しました。
    * しかし、SPSA特有の「高分散（ノイズの多さ）」により、学習が激しく振動し、28%付近で頭打ちになりました。

---

## 4. `main_sgd.py` (FedSGD for B-matrix)

* **略称:** `SGD` = **S**tochastic **G**radient **D**escent
* **手法:** `FedSGD-LoRA`（**解決策2** / PDFの「選択肢1」）
    * `main_ospsa.py` のアーキテクチャ（`B_local` 廃止）を維持しつつ、学習が不安定な**SPSAを廃止**しました。
    * [cite_start]サーバは報酬 $R_i$ ではなく、クライアントが計算した**正確な勾配 $\nabla_B \mathcal{L}$** を収集します [cite: 160-164, 327]。
    * サーバは収集した勾配を平均化（FedSGD）し、$B^{server}$ を安定的に更新します。
* **検証方法:** Gradient-based Proxy (GDP/GBP)法
* **ステータス:** **現在の推奨手法**
    * [cite_start]SPSAの「高分散（ノイズ）」問題を、PDFが提示する「低分散（正確な勾配）」な方法で解決するアプローチです。