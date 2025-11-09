# FedLoRAサーバ更新とShapley値妥当性検証

このリポジトリは、LoRAを用いた連合学習において、サーバ側のB行列を更新する様々な手法を実装し、その貢献度（Shapley値）を検証するための実験コード群です。

## フォルダ構成と実行方法

このプロジェクトは、使用する「ベースモデル」によってフォルダが分かれています。

* `simpleCNN/`: 軽量なCNNでの試行錯誤の履歴。
* `resnet/`: ResNet-50 をベースモデルとして使用するスクリプト群。
* `vit/`: Vision Transformer (ViT) をベースモデルとして使用するスクリプト群。

### 実行手順

1.  ルートディレクトリにある `config.yml` を編集し、学習パラメータ（`client_lr`, `server_lr` など）や、`image_size` を設定します。
2.  Conda環境を有効化します。
    ```bash
    conda activate FedSA-LoRA-DP-py310
    ```
3.  実行したいモデルのフォルダに入り、実行したい手法のスクリプトを実行します。

    **（例：ResNetで、FedSGD手法を実行する場合）**
    ```bash
    python resnet/main_sgd.py
    ```
    （スクリプトは自動的に `../config.yml` を読み込みます）

4.  ログはルートディレクトリの `logs/` フォルダ内に、`config.yml` の `experiment_name` に基づいた名前で保存されます。

---

## 各実験ファイル（手法）の説明

各フォルダ内にある `main_*.py` ファイルは、サーバ側の `B` 行列をどう扱うか、という戦略（アルゴリズム）の違いを示しています。

### 1. `main_sgd.py` (✅ 推奨手法)

* **略称:** `SGD` = **S**tochastic **G**radient **D**escent
* **手法:** `FedSGD-LoRA`（PDFの「選択肢1」）
* **アーキテクチャ:**
    * **クライアント:** `A_i` をローカル学習（`client_lr` を使用）し、同時に `B_server` の勾配 `g_B` を計算してサーバに送信します。
    * **サーバ:** 全クライアントから `g_B` を収集・平均化し、`server_lr` を使って `B_server` を（FedSGDで）更新します。
* **ステータス:** **学習成功**。
    * `simpleCNN/` での実験では、この手法のみが安定して収束し（最大32.78%）、Shapley値の妥当性検証（`rho = 0.9`）にも成功しました。

### 2. `main_fixed.py` (デバッグ用)

* **略称:** `Fixed-B` = `B`行列固定
* **手法:** `main_sgd.py` から `B_server` の更新ロジックを**削除**したもの。
* **目的:** `B_server` が動かない「静的な土俵」で `A_i` だけを学習させ、`Global Test Accuracy` や妥当性検証（`rho`）がどう変化するかを観測するためのデバッグ用スクリプトです。

---

### 3. `simpleCNN/` 内の旧実験（不安定または失敗）

`simpleCNN/` フォルダには、`main_sgd.py` にたどり着くまでの試行錯誤の履歴が残されています。

* **`main_ospsa.py` (SPSA版):**
    * `B_server` を強化学習（SPSA）で更新する手法。
    * `Global Test Accuracy` が10%の壁を突破（〜28%）しましたが、SPSA特有の「高分散（ノイズ）」により学習が激しく**振動**し、安定しませんでした。
* **`main_wspsa.py` (Shapley加重SPSA版):**
    * SPSAの更新をShapley値で加重平均する改良案。
    * SPSAの振動を抑えきれず、さらに「`B`の更新」と「Shapley値の計算」が**循環参照**（鶏と卵の問題）となり、計算コストが爆発的に増大しました。
* **`main_gdp.py` (Dual-B版):**
    * PDFの最初の設計案（`B_local` と `B_server` の2種類を持つ）。
    * `Global Test Accuracy` が**10%（ランダム）**から上昇せず、**学習が破綻**しました。アーキテクチャの根本的な不整合が原因と特定されました。
* **`main_loo.py` (LOO検証版):**
    * `Dual-B` 設計の妥当性を、(N+1)回の再学習で検証しようとした初期スクリプト。検証コストが非現実的なため破棄されました。