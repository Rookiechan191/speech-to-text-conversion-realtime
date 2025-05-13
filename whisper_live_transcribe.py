import sounddevice as sd
import numpy as np
import torch
from transformers import pipeline
import gradio as gr
import queue
import threading

# Initialize Whisper ASR pipeline
MODEL_NAME = "openai/whisper-small"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=5,
    device=0 if torch.cuda.is_available() else "cpu",
)

# Audio settings
SAMPLING_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5  # seconds
CHUNK_SIZE = SAMPLING_RATE * CHUNK_DURATION

# Queue to store audio data
audio_queue = queue.Queue()

# Audio callback to put audio data into the queue
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_chunk = indata.copy().flatten()
    audio_queue.put(audio_chunk)

# Thread to continuously read from the queue and transcribe
def transcribe_audio():
    global latest_transcription
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        audio_data = audio_data.astype(np.float32)
        transcription = pipe(audio_data)["text"]
        print("Transcription:", transcription)
        latest_transcription = transcription

# Variable to hold the latest transcription
latest_transcription = ""

# Start audio stream and transcription thread
def start_transcription():
    threading.Thread(target=transcribe_audio, daemon=True).start()
    with sd.InputStream(callback=audio_callback, channels=CHANNELS,
                        samplerate=SAMPLING_RATE, blocksize=CHUNK_SIZE):
        while True:
            sd.sleep(1000)

# Gradio interface function
def get_latest_transcription():
    return latest_transcription

# Start audio streaming in background
threading.Thread(target=start_transcription, daemon=True).start()

# Gradio UI to display latest transcription
gr.Interface(fn=get_latest_transcription,
             inputs=[],
             outputs="text",
             live=True,
             title="Real-time Whisper Transcription",
             description="Speak into your microphone and see the transcription in real time").launch()

