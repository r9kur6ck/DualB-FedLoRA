# FedLoRAサーバ更新とShapley値妥当性検証

このリポジトリは、LoRAを用いた連合学習（Federated LoRA）において、サーバ側のB行列を更新する様々な手法を実装し、その貢献度（Shapley値）を検証するための実験コード群です。

## フォルダ構成

各フォルダは、異なるベースモデル（CNN, ResNet, ViT）での実験を示しています。

* `simpleCNN/`
    * 実験の初期段階で使用した小規模なCNNモデルです。
    * 私たちが行った試行錯誤の全履歴（`Dual-B` の失敗、`SPSA` の不安定さ、`FedSGD` での成功）がすべて含まれています。
* `resnet/`
    * `simpleCNN/` での実験を経て確立された、安定したアーキテクチャ（`Fixed-B` と `FedSGD`）を、ResNet-50（BiT）モデルに適用した実装です。
* `vit/`
    * `resnet/` と同様に、安定版のアーキテクチャをVision Transformer (ViT) モデルに適用した実装です。

---

## 各実験ファイル（手法）の説明

`simpleCNN/` フォルダ内にある多数の `main_*.py` ファイルは、以下の順序で開発された手法の変遷を示しています。

### 1. `main_loo.py` (アーカイブ済み)

* **略称:** `LOO` = **L**eave-**O**ne-**O**ut（1人抜き法）
* **手法:** `Dual-B FedLoRA`（PDFの最初の設計案）
* **検証:** クライアントを1人ずつ除外して(N+1)回の**再学習**を行い、実際の貢献度 $\Delta_i$ を測定する。
* **結果:** 検証コスト（再学習）が非現実的なため、このアプローチは破棄されました。

### 2. `main_gdp.py` (失敗)

* **略称:** `GDP` = **G**radient-**b**ased **P**roxy
* **手法:** `Dual-B FedLoRA`（`main_loo.py` と同じ）
* **検証:** LOOのコストを回避するため、軽量な代理指標 $C_i$ （`A`の勾配の内積）を導入。
* **結果:** **学習破綻**。`Global Test Accuracy` が**10%（ランダム）**から上昇せず。
* **原因:** `Dual-B` アーキテクチャの根本的な**不整合**（クライアントの `B_local` とサーバの `B_server` が協調しない）により、SPSAがノイズしか学習できず失敗しました。

### 3. `main_ospsa.py` (不安定)

* **略称:** `SPSA` = **S**imultaneous **P**erturbation **S**tochastic **A**pproximation
* **手法:** `SPSA-FedLoRA`（解決策1：`B_local` を廃止）
* **検証:** GDP法
* **結果:** 学習の不整合が解消され、精度が**10%の壁を突破**（〜28%）。
* **課題:** SPSA特有の「高分散（ノイズ）」により、学習が激しく**振動**し、安定しませんでした。

### 4. `main_wspsa.py` (不安定・高コスト)

* **略称:** `W-SPSA` = **W**eighted-**SPSA**（Shapley加重SPSA）
* **手法:** `main_ospsa.py` の改良版。SPSAの更新をShapley値で加重平均する。
* **検証:** GDP法
* **結果:** SPSAの振動を抑えきれず、さらに「`B`の更新」と「Shapley値の計算」が**循環参照**（鶏と卵の問題）となり、計算コストが爆発的に増大しました。

### 5. `main_sgd.py` (✅ 推奨手法)

* **略称:** `SGD` = **S**tochastic **G**radient **D**escent
* **手法:** `FedSGD-LoRA`（PDFの「選択肢1」）
* **アーキテクチャ:**
    * `main_ospsa.py` のアーキテクチャ（`B_local` 廃止）を維持しつつ、不安定な**SPSAを廃止**。
    * クライアントが `A_i` を学習し、同時に **`B_server` の正確な勾配 `g_B`** を計算してサーバに送信。
    * サーバが `g_B` を平均化（FedSGD）し、`B_server` を安定的に更新する。
* **結果:**
    * **学習の成功:** `Global Test Accuracy` が**安定して収束**し、`32.78%` の精度を達成。
    * **検証の成功:** Shapley値 $\phi_i$ とProxyスコア $C_i$ の相関が **`rho = 0.9`** となり、妥当性検証にも成功しました。

### 6. `main_fixed.py` (デバッグ用)

* **略称:** `Fixed-B` = `B`行列固定
* **手法:** `main_sgd.py` から `B_server` の更新ロジックを**削除**したもの。
* **目的:** `B_server` が動かない「静的な土俵」で `A_i` だけを学習させ、Shapley値とProxyスコアの相関（`rho`）がどうなるかを観測するためのデバッグ用スクリプトです。

---

## 🏃 実行方法 (推奨手法)

`main_sgd.py`（または `resnet/main_sgd.py`, `vit/main_sgd.py`）を実行するのが最も安定した結果を得られます。

1.  `config.yml` を編集し、学習率（`server_lr`, `client_lr`）やモデルタイプ（`model_type: 'ResNet'`など）を設定します。

    ```yaml
    # config.yml
    
    experiment_name: "FedSGD_ResNet_Test"
    num_clients: 5
    num_rounds: 20
    
    model_type: 'ResNet' # 'SimpleCNN', 'ResNet', 'ViT' から選択
    image_size: 128      # ResNet/ViT の場合は 128 または 224
    
    rank: 4
    lora_target_modules: ['mlp', 'head'] # model_type: 'ViT' の場合のみ有効
    
    server_update_strategy: 'FedSGD' # この設定は 'FedSGD' のまま
    
    server_lr: 0.1
    client_lr: 0.01
    
    eval_interval: 1
    ```

2.  Conda環境を有効化し、目的のモデルのスクリプトを実行します。

    ```bash
    conda activate FedSA-LoRA-DP-py310
    
    # ResNetで実行する場合
    python resnet/main_sgd.py
    
    # ViTで実行する場合
    # python vit/main_sgd.py
    ```

3.  `experiment_fedsgd_resnet.log` のようなログファイルが生成されます。