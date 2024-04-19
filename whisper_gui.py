# import whisper

# model = whisper.load_model("base")
# result = model.transcribe("1-higedan-2-kanji.wav")
# print(result["text"])

import gradio as gr
import whisper

# Whisperモデルをロードする関数。大がかりな作業を避けるために、一度ロード後は維持されます。
model = whisper.load_model("base")

def transcribe_audio(file_obj):
    """マイクから受け取った音声ファイルを文字に変換します"""
    result = model.transcribe(file_obj.name)
    return result["text"]

with gr.Blocks() as demo:
    # Gradioインターフェースを定義
    gr.Markdown("### Whisperによる音声認識")
    with gr.Row():
        # ユーザーが音声録音を提供できるグラフィカル・インターフェース要素を作成
        audio_input = gr.Audio(sources="microphone", type="filepath", label="音声を録音")
    with gr.Row():
        # 録音された音声を文字に変換するボタン
        submit_btn = gr.Button("変換")
    # 変換結果を表示するテキストボックス
    transcription_output = gr.Textbox(label="認識結果")

    # ボタンが押されたときの動作を設定（音声ファイルを渡して、結果をテキストボックスに表示）
    submit_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=transcription_output,
    )

# アプリケーションを起動
demo.launch()