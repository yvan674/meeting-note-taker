"""Meeting Note Taker Main Script.

Takes notes during meetings by running a Whisper instance during the meeting
"""
import logging
import threading
import queue
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import mlx.core as mx
import numpy as np
import sounddevice as sd
import soundfile as sf
import lighting_whisper_mlx as lwm
from lighting_whisper_mlx.transcribe import ModelHolder, transcribe_audio
from pydub import AudioSegment
from huggingface_hub import hf_hub_download

logging.basicConfig(
            level="INFO",
            format="[%(levelname)s] %(asctime)s %(filename)s:%(lineno)4d "
                   "— %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS=1
BUFFER_SIZE = 1024

audio_queue = queue.Queue()
whisper_queue = queue.Queue()
recording = True

MODEL_NAME = lwm.lightning.models["base"]["8bit"]
SAVE_DIR = (Path.home() / "dev/mlx_models").resolve()
SAVE_DIR.mkdir(exist_ok=True)
MODEL_DIR = SAVE_DIR / "whisper-base-mlx-8bit"


def load_model():

    files = ("weights.npz", "config.json")
    for file in files:
        hf_hub_download(repo_id=MODEL_NAME,
                        filename=file,
                        local_dir=MODEL_DIR)
    # Preload the model
    ModelHolder.get_model(str(MODEL_DIR), dtype=mx.float16)


logger.info("Loading Lightning Whisper MLX...")
load_model()


def parse_args():
    """Parses CLI arguments."""
    p = ArgumentParser(description="Meeting Note Taker")


    return p.parse_args()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())
    whisper_queue.put(indata.copy())


def transcriber():
    """Collects audio from the queue and processes it for transcription."""
    buffer_length_seconds = 30
    buffer_size = SAMPLE_RATE * buffer_length_seconds
    buffer = mx.zeros(buffer_size, dtype=mx.float16)
    last_pos = 0
    collected_frames = 0

    logging.info("Starting transcriber buffer process...")

    while recording or not whisper_queue.empty():
        try:
            data = whisper_queue.get(timeout=0.1)

            if last_pos + len(data) > buffer_size:
                buffer = mx.roll(buffer, -len(data))
                last_pos = buffer_size - len(data)

            buffer[last_pos:last_pos + len(data)] = mx.array(data.flatten(),
                                                             dtype=mx.float16)
            last_pos += len(data)
            collected_frames += len(data)

            if collected_frames >= SAMPLE_RATE:
                # This block would typically trigger a transcription
                # collected_frames = 0
                result = transcribe_audio(buffer,
                                          path_or_hf_repo=str(MODEL_DIR),
                                          batch_size=12)
                print(result)

        except queue.Empty:
            # No data in the queue, continue the loop and check again
            continue


def file_writer(file_path: Path):
    """Collects audio from the queue and writes 1-second chunks at a time."""
    buffer = []
    collected_frames = 0

    with sf.SoundFile(file_path, mode='w', samplerate=SAMPLE_RATE,
                      channels=CHANNELS, subtype='PCM_16') as file:
        while recording or not audio_queue.empty():
            try:
                data = audio_queue.get(timeout=0.1)
                buffer.append(data)
                collected_frames += len(data)
                # Once we've collected ~1 second of audio, write it
                if collected_frames >= SAMPLE_RATE:
                    file.write(np.concatenate(buffer))
                    buffer.clear()
                    collected_frames = 0
                audio_queue.task_done()
            except queue.Empty:
                continue

        # Flush leftover frames at the end
        if buffer:
            file.write(np.concatenate(buffer))


def convert_wav_to_mp3(wav_fp: Path, mp3_fp: Path):
    """Convert WAV file to MP3 using pydub."""
    audio = AudioSegment.from_wav(wav_fp)
    audio.export(mp3_fp, format="mp3")


def main():
    global recording
    logger.info("Starting recording. Press Ctrl+C to stop.")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file_path = Path(tmpdir) / "meeting_audio.wav"
        writer_thread = threading.Thread(target=file_writer, args=(audio_file_path,))
        writer_thread.start()

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                callback=audio_callback, blocksize=BUFFER_SIZE):
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            logger.info("Stopping recording...")
        finally:
            recording = False
            audio_queue.join()
            writer_thread.join()

            mp3_file_path = Path.cwd() / "meeting_audio.mp3"
            convert_wav_to_mp3(audio_file_path, mp3_file_path)
            logger.info(f"Audio saved to {mp3_file_path}")


if __name__ == '__main__':
    args = parse_args()
    main()
