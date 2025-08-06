"""Gradio demo for OmniGlue image matching."""

import os
import time
import gradio as gr
import numpy as np
import omniglue
from omniglue import utils


class OmniGlueDemo:
    def __init__(self):
        self.og = None
        self.load_model()

    def load_model(self):
        """OmniGlueモデルを読み込み"""
        print("OmniGlue (SuperPoint & DINOv2) を読み込み中...")
        start_time = time.time()
        
        try:
            self.og = omniglue.OmniGlue(
                og_export="./models/og_export",
                sp_export="./models/sp_v6",
                dino_export="./models/dinov2_vitb14_pretrain.pth",
            )
            elapsed_time = time.time() - start_time
            print(f"モデル読み込み完了 ({elapsed_time:.2f}秒)")
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            raise

    def match_images(self, image1_input, image2_input, confidence_threshold=0.02):
        """2つの画像をマッチング"""
        if image1_input is None or image2_input is None:
            return None, "画像を2つアップロードしてください"
        
        if self.og is None:
            return None, "モデルが読み込まれていません"

        try:
            # PIL画像をnumpy配列に変換
            image1 = np.array(image1_input.convert("RGB"))
            image2 = np.array(image2_input.convert("RGB"))
            
            print("マッチングを実行中...")
            start_time = time.time()
            
            # マッチング実行
            match_kp0, match_kp1, match_confidences = self.og.FindMatches(image1, image2)
            num_matches = match_kp0.shape[0]
            
            print(f"初期マッチング数: {num_matches}")
            
            # 信頼度によるフィルタリング
            keep_idx = []
            for i in range(match_kp0.shape[0]):
                if match_confidences[i] > confidence_threshold:
                    keep_idx.append(i)
            
            num_filtered_matches = len(keep_idx)
            if num_filtered_matches == 0:
                return None, f"信頼度 {confidence_threshold} 以上のマッチングが見つかりませんでした"
            
            match_kp0 = match_kp0[keep_idx]
            match_kp1 = match_kp1[keep_idx]
            match_confidences = match_confidences[keep_idx]
            
            # 可視化
            viz = utils.visualize_matches(
                image1,
                image2,
                match_kp0,
                match_kp1,
                np.eye(num_filtered_matches),
                show_keypoints=True,
                highlight_unmatched=True,
                title=f"{num_filtered_matches} matches (threshold: {confidence_threshold})",
                line_width=2,
            )
            
            elapsed_time = time.time() - start_time
            result_message = (
                f"マッチング完了!\n"
                f"実行時間: {elapsed_time:.2f}秒\n"
                f"全マッチング数: {num_matches}\n"
                f"フィルタ後マッチング数: {num_filtered_matches}\n"
                f"信頼度閾値: {confidence_threshold}"
            )
            
            return viz, result_message
            
        except Exception as e:
            error_message = f"マッチング処理でエラーが発生しました: {str(e)}"
            print(error_message)
            return None, error_message


def create_demo():
    """Gradioデモインターフェースを作成"""
    demo_instance = OmniGlueDemo()
    
    with gr.Blocks(title="OmniGlue Image Matching Demo") as demo:
        gr.Markdown("""
        # OmniGlue: 汎用画像特徴マッチングデモ
        
        2つの画像をアップロードして、OmniGlueによる特徴点マッチングを試すことができます。
        OmniGlueは、SuperPointとDINOv2を組み合わせた汎用的な画像マッチング手法です。
        
        **使い方:**
        1. 左右に画像をアップロードしてください
        2. 信頼度閾値を調整できます（低い値でより多くのマッチング、高い値でより確実なマッチング）
        3. "マッチング実行"ボタンをクリックしてください
        """)
        
        with gr.Row():
            with gr.Column():
                image1_input = gr.Image(
                    type="pil",
                    label="画像1",
                    height=300
                )
            with gr.Column():
                image2_input = gr.Image(
                    type="pil", 
                    label="画像2",
                    height=300
                )
        
        with gr.Row():
            confidence_slider = gr.Slider(
                minimum=0.001,
                maximum=0.1,
                value=0.02,
                step=0.001,
                label="信頼度閾値",
                info="マッチングの信頼度閾値（低い値でより多くのマッチングを表示）"
            )
        
        match_button = gr.Button("マッチング実行", variant="primary", size="lg")
        
        with gr.Row():
            output_image = gr.Image(
                type="numpy",
                label="マッチング結果", 
                height=400
            )
        
        result_text = gr.Textbox(
            label="実行結果",
            lines=5,
            max_lines=10
        )
        
        # デモ用のサンプル画像
        gr.Markdown("### サンプル画像")
        with gr.Row():
            gr.Examples(
                examples=[
                    ["./res/demo1.jpg", "./res/demo2.jpg", 0.02]
                ],
                inputs=[image1_input, image2_input, confidence_slider],
                label="サンプル画像を試す"
            )
        
        # イベントハンドラー
        match_button.click(
            fn=demo_instance.match_images,
            inputs=[image1_input, image2_input, confidence_slider],
            outputs=[output_image, result_text]
        )
        
        # 技術仕様
        gr.Markdown("""
        ### 技術仕様
        - **特徴点検出**: SuperPoint
        - **視覚基盤モデル**: DINOv2 (ViT-B/14)
        - **マッチング手法**: OmniGlue (CVPR'24)
        - **主要な特徴**: クロスドメイン汎化性能、位置情報と外観情報の分離
        """)
    
    return demo


def main():
    """メイン関数"""
    # モデルファイルの存在確認
    required_models = [
        "./models/og_export",
        "./models/sp_v6", 
        "./models/dinov2_vitb14_pretrain.pth"
    ]
    
    missing_models = []
    for model_path in required_models:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print("必要なモデルファイルが見つかりません:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nREADME.mdの指示に従ってモデルファイルをダウンロードしてください。")
        return
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()