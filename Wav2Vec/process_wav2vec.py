import start 
import os
import pandas as pd

import torchaudio

import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC,  Wav2Vec2CTCTokenizer

from start import shift_audio
from start import process_shift
from start import decoding_to_timings

audio_folder = start.audio_folder
shift_ms = start.shift_range
shifted_dfs = {}


output_folder = start.output_folder


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
                start.processor,
                start.model,
                decoding_to_timings)

            shifted_df.to_csv(
            os.path.join(speaker_dir, f"{speaker_type}_{ms}ms.csv"),
            index=False)
            
            # store in dict with a unique key
            shifted_dfs[f"Shift_{speaker_type}_{ms}ms"] = shifted_df
        
            print(f"Finished {speaker_type}_{ms}ms")

dfs = []

for file in os.listdir(output_folder):
    if file.endswith(".csv") and file != "Wav2vec_OG.csv" and not file.startswith("combined"):
        #Get path of each file
        file_path = os.path.join(output_folder, file)
       
        # Read CSV
        df = pd.read_csv(file_path)

        #Get the name without the extension
        base = os.path.splitext(file)[0]

        #Split each word in the name
        parts = base.split("_")
        #print(parts)

        # The timing value is in index 2
        timing = parts[2]   
    
         # Rename columns to timing_columnname
        df.columns = [f"{timing}_{col}" for col in df.columns]

        # Store
        dfs.append(df)

#All the shifted csv appending
combined_df = pd.concat(dfs, axis=1)

# Read the non-shifted
df_wav2vec = pd.read_csv(f"{output_folder}/Wav2Vec_OG.csv")

# Reset index
df_wav2vec.reset_index(drop=True, inplace=True)

#combine them into one
full_df = pd.concat([combined_df, df_wav2vec], axis = 1)

full_df.to_csv(os.path.join(output_folder, "combined.csv"), index=False)

print("Done:", full_df.shape)