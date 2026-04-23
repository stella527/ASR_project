import wav2vec_start 
import os
import pandas as pd

import torchaudio

import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC,  Wav2Vec2CTCTokenizer

from wav2vec_start  import shift_audio
from wav2vec_start  import process_shift
from wav2vec_start  import decoding_to_timings

audio_folder = wav2vec_start.audio_folder
shift_ms = wav2vec_start.shift_range
shifted_dfs = {}


output_folder = wav2vec_start.output_folder


for wav in os.listdir(audio_folder):

    if wav.endswith(".wav"):

        print(wav)
        #Get the name without the extension
        base = os.path.splitext(wav)[0]
        #Split each word in the name
        speaker_type = base.split("_")[0]

        #Get path of each audio file
        audio_path = os.path.join(audio_folder, wav)
        print(audio_path)
        
        og_waveform, sr = torchaudio.load(audio_path)
        #og_waveform, sr = sf.read(audio_path)
        print("Trying to load:", audio_path)

        speaker_dir = os.path.join(output_folder, speaker_type)
        os.makedirs(speaker_dir, exist_ok=True)
        
        for ms in shift_ms:
            shifted_audio = shift_audio(og_waveform, ms, sr)
        
            #print(shifted_audio.detach().cpu().numpy().shape)
        
        
            shifted_df = process_shift(
                shifted_audio,
                ms,
                sr,
                output_folder,
                wav2vec_start.processor,
                wav2vec_start.model,
                decoding_to_timings)

            shifted_df.to_csv(
            os.path.join(speaker_dir, f"{speaker_type}_{ms}ms.csv"),
            index=False)
            
            # store in dict with a unique key
            shifted_dfs[f"Shift_{speaker_type}_{ms}ms"] = shifted_df
        
            print(f"Finished {speaker_type}_{ms}ms")

dfs = []

