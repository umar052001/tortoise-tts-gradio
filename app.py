import tempfile
import gradio as gr
import numpy as np
from typing import List

import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice

# Constants
TORTOISE_SR_IN = 22050
TORTOISE_SR_OUT = 24000

# Function to chunk audio into smaller segments
def chunk_audio(t: torch.Tensor, sample_rate: int, chunk_duration_sec: int) -> List[torch.Tensor]:
    duration = t.shape[1] / sample_rate
    num_chunks = 1 + int(duration / chunk_duration_sec)
    chunks = [t[:, (sample_rate * chunk_duration_sec * i):(sample_rate * chunk_duration_sec * (i + 1))] for i in range(num_chunks)]
    # Remove 0-width chunks
    chunks = [chunk for chunk in chunks if chunk.shape[1] > 0]
    return chunks

# Function to generate text-to-speech
def tts_main(voice_samples: List[torch.Tensor], text: str, model_preset: str) -> str:
    tts = TextToSpeech()
    gen = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=None,
        preset=model_preset
    )
    torchaudio.save("generated.wav", gen.squeeze(0).cpu(), TORTOISE_SR_OUT)
    return "generated.wav"

# Function to generate text-to-speech from a preset voice
def tts_from_preset(voice: str, text, model_preset):
    voice_samples, _ = load_voice(voice)
    return tts_main(voice_samples, text, model_preset)

# Function to generate text-to-speech from uploaded audio files
def tts_from_files(files: List[tempfile._TemporaryFileWrapper], do_chunk, text, model_preset):
    voice_samples = [load_audio(f.name, TORTOISE_SR_IN) for f in files]
    if do_chunk:
        voice_samples = [chunk for t in voice_samples for chunk in chunk_audio(t, TORTOISE_SR_IN, 10)]
    return tts_main(voice_samples, text, model_preset)

# Function to generate text-to-speech from recorded audio
def tts_from_recording(recording: Tuple[int, np.ndarray], do_chunk, text, model_preset):
    sample_rate, audio = recording
    # Normalize the audio
    norm_fix = 1
    if audio.dtype == np.int32:
        norm_fix = 2 ** 31
    elif audio.dtype == np.int16:
        norm_fix = 2 ** 15
    audio = torch.FloatTensor(audio.T) / norm_fix
    if len(audio.shape) > 1:
        # Convert to mono
        audio = torch.mean(audio, axis=0).unsqueeze(0)
    audio = torchaudio.transforms.Resample(sample_rate, TORTOISE_SR_IN)(audio)
    if do_chunk:
        voice_samples = chunk_audio(audio, TORTOISE_SR_IN, 10)
    else:
        voice_samples = [audio]
    return tts_main(voice_samples, text, model_preset)

# Function to generate text-to-speech from a URL (YouTube audio)
def tts_from_url(audio_url, start_time, end_time, do_chunk, text, model_preset):
    os.system(f"yt-dlp -x --audio-format mp3 --force-overwrites {audio_url} -o audio.mp3")
    audio = load_audio("audio.mp3", TORTOISE_SR_IN)
    audio = audio[:, start_time * TORTOISE_SR_IN:end_time * TORTOISE_SR_IN]
    if do_chunk:
        voice_samples = chunk_audio(audio, TORTOISE_SR_IN, 10)
    else:
        voice_samples = [audio]
    return tts_main(voice_samples, text, model_preset)

# Gradio interface setup
with gr.Blocks() as demo:
   gr.Markdown(README)

    preset = gr.Dropdown(PRESETS, label="Model preset", value=DEFAULT_PRESET)
    text   = gr.Textbox(label="Text to speak", value=DEFAULT_TEXT)
    do_chunk_label = "Split audio into chunks? (for audio much longer than 10 seconds.)"
    do_chunk_default = True

    with gr.Tab("Choose preset voice"):
      inp1      = gr.Dropdown(VOICES, value=DEFAULT_VOICE, label="Preset voice")
      btn1      = gr.Button("Generate")

    with gr.Tab("Upload audio"):
      inp2      = gr.File(file_count="multiple")
      do_chunk2 = gr.Checkbox(label=do_chunk_label, value=do_chunk_default)
      btn2      = gr.Button("Generate")
    
    with gr.Tab("Record audio"):
      inp3      = gr.Audio(source="microphone")
      do_chunk3 = gr.Checkbox(label=do_chunk_label, value=do_chunk_default)
      btn3      = gr.Button("Generate")

#    with gr.Tab("From YouTube"):
#      inp4       = gr.Textbox(label="URL")
#      do_chunk4  = gr.Checkbox(label=do_chunk_label, value=do_chunk_default)
#      start_time = gr.Number(label="Start time (seconds)", precision=0)
#      end_time   = gr.Number(label="End time (seconds)", precision=0)
#      btn4       = gr.Button("Generate")

    audio_out = gr.Audio()

    btn1.click(
      tts_from_preset,
      [inp1, text, preset],
      [audio_out],
    )
    btn2.click(
      tts_from_files,
      [inp2, do_chunk2, text, preset],
      [audio_out],
    )
    btn3.click(
      tts_from_recording,
      [inp3, do_chunk3, text, preset],
      [audio_out],
    )
#    btn4.click(
#      tts_from_url,
#      [inp4, start_time, end_time, do_chunk4, text, preset],
#      [audio_out],
#    )

    
demo.launch()
