# CLAUDE.md

このファイルは、このリポジトリでのコード作業において Claude Code (claude.ai/code) にガイダンスを提供します。

## プロジェクト概要

OmniGlue は、基盤モデルの知識を活用した汎用的な画像特徴マッチング手法です（CVPR'24）。異なる画像ドメインに対する汎化性能を重視して設計されており、SuperPoint（キーポイント検出）とDINOv2（視覚基盤モデル）を組み合わせています。

## 開発環境とパッケージ管理

### uv の使用
このプロジェクトは `uv` パッケージマネージャーを推奨します：

```bash
# プロジェクトの依存関係をインストール
uv sync

# スクリプト実行
uv run gradio_demo.py
```

### 必要なモデルファイル
プロジェクトを動作させるには以下のモデルファイルが `./models/` ディレクトリに必要です：
- SuperPoint: `sp_v6/` 
- DINOv2: `dinov2_vitb14_pretrain.pth`
- OmniGlue: `og_export/`

## コードアーキテクチャ

### 主要なコンポーネント

1. **OmniGlue クラス** (`src/omniglue/omniglue_extract.py`)
   - メインのマッチング機能を提供
   - `FindMatches(image0, image1)` メソッドで2つの画像間のマッチングを実行
   - SuperPointとDINOv2の特徴量を統合してマッチングを行う

2. **特徴抽出器**
   - `superpoint_extract.py`: SuperPointによるキーポイント検出と記述子抽出
   - `dino_extract.py`: DINOv2による密な特徴マップ抽出

3. **ユーティリティ** (`src/omniglue/utils.py`)
   - `visualize_matches()`: マッチング結果の可視化
   - `soft_assignment_to_match_matrix()`: ソフトアサインメントからバイナリマッチ行列への変換
   - `lookup_descriptor_bilinear()`: 双線形補間による記述子の取得

### データフロー
1. 2つの画像を入力
2. SuperPointでキーポイントと記述子を抽出
3. DINOv2で密な特徴マップを抽出
4. OmniGlueモデルで特徴マッチングを実行
5. 信頼度に基づいてマッチングを閾値処理
6. マッチング結果（キーポイント座標と信頼度）を返す

## 開発時の重要なポイント

### マッチング処理の流れ
- `OmniGlue.FindMatches()` は主要なエントリポイント
- SuperPointとDINOの特徴量を組み合わせてマッチングを実行
- 結果は `(match_kp0s, match_kp1s, match_confidences)` として返される

### 可視化機能
- `utils.visualize_matches()` でマッチング結果を可視化可能
- パラメータで線の太さ、キーポイントの表示などをカスタマイズ可能

### テンソル処理
- TensorFlowを使用してモデル推論を実行
- NumPy配列とTensorFlowテンソル間の変換が頻繁に発生

## デモとテスト

### 既存のデモ
```bash
uv run gradio_demo.py
```
- WebブラウザでGUIを使用してマッチング実行
- http://localhost:7860 でアクセス
- 2つの画像をアップロードしてリアルタイムでマッチング確認が可能

### 推奨される開発パターン
- 新機能開発前にgradio_demo.pyで動作を確認
- Gradioデモでインタラクティブにパラメータ調整が可能

## 依存関係

主要な依存関係：
- tensorflow: モデル推論
- numpy: 数値計算
- opencv-python: 画像処理と可視化
- matplotlib: グラフ作成
- Pillow: 画像読み込み
- torch: PyTorchベースの特徴抽出

## 注意事項

- GPUメモリ使用量が大きいため、大きな画像では注意が必要
- モデルファイルのダウンロードには時間がかかる場合がある
- 信頼度閾値（デフォルト：0.02）の調整でマッチング品質が変わる