import os

import numpy as np

import gradio as gr
import whisper

model = whisper.load_model("base")

def transcribe_audio():
    result = model.transcribe("./flagged/audio/")
    return result["text"]


input_audio = gr.Audio(
    sources=["microphone"],
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=input_audio,
    outputs="audio"
)

if __name__ == "__main__":
    demo.launch()