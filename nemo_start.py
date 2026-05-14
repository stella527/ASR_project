import os
import torch
import torchaudio
import pandas as pd
import librosa

import general
from general import shift_audio
import nemo.collections.asr as nemo_asr


asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

os.makedirs("Nemo_all_shifts", exist_ok=True)
output_folder = "Nemo_all_shifts"

shifted_dir = os.path.join(output_folder, "Shifted_Audio")
os.makedirs(shifted_dir, exist_ok=True)


# Call the audio
audio_folder = general.nemo_only


# def process_audio(audio):

#     # Run ASR model
#     output = asr_model.transcribe(audio, timestamps=True)

#     word_timestamps = output[0].timestamp['word']

#     word_data = []
#     for word in word_timestamps:

#         word_data.append({"Word": word['word'],
#                         "Start": word['start'],
#                         "End": word['end']})

#     df_word = pd.DataFrame(word_data).round(decimals = 3)

#     return df_word

#Get transcription
def process_audio(shifted_audio, sr, ms):

    # Create file path for the shifted version
    base_name = os.path.join(shifted_dir, f"audio_shift_{ms}ms.wav")

    # Save the shifted version
    torchaudio.save(base_name, shifted_audio, sr)

    # #Now inputing new audio
    # shifted_waveform, sample_rate = librosa.load(base_name, sr=None)

    # # Shift audio
    # #shifted_waveform = shift_audio(, shift_ms, sr)

    # # Ensure shape [samples] for librosa
    # if shifted_waveform.ndim() > 1:
    #     shifted_waveform = shifted_waveform.squeeze(0)

    # Convert torch.Tensor to numpy for librosa
    speech_np = shifted_audio.squeeze().cpu().numpy()

    # Resample to 16kHz if needed
    target_sr = 16000
    if sr != target_sr:
        speech_np = librosa.resample(speech_np, orig_sr=sr, target_sr=target_sr)

     # Run ASR model
    output = asr_model.transcribe(speech_np, timestamps=True)

    # by default, timestamps are enabled for char, word and segment level
    word_timestamps = output[0].timestamp['word'] # word level timestamps for first sample

    word_data = []
    for word in word_timestamps:
        #print(f"{word['start']}s - {word['end']}s : {word['word']}")

        word_data.append({"word": word['word'],
                          "start": word['start'],
                          "end": word['end']})

    df_word = pd.DataFrame(word_data).round(decimals = 3)

    return df_word