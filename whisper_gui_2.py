import os

import numpy as np

import gradio as gr
import whisper

model = whisper.load_model("large")

# def transcribe_audio():
#     result = model.transcribe("./flagged/audio/")
#     return result["text"]

def transcribe_audio(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return result.text

input_audio = gr.Audio(
    sources=["microphone"],
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
    type="filepath"
)
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=[input_audio],
    outputs="textbox"
)

if __name__ == "__main__":
    demo.launch()