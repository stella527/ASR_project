import os
import pandas as pd
import torch
import soundfile as sf

#Local functions
import general
import nemo_start

from nemo_start import process_audio

from general import shift_audio
from general import combine_transcription
from general import align_words
from general import split_shifted_dfs
from general import final_timestamps




# Call the audio
audio_folder = general.nemo_only

shift_ms = general.shift_range

output_folder = nemo_start.output_folder


#######################
# Timing map for shifting
# Include in each model
#########################

word_columns = []
timing_map = {}
for i in shift_ms:
    word_columns.append((f"{i}ms_word"))
    timing_map[f"{i}ms_word"] = [f"{i}ms_start", f"{i}ms_end"]


reference_col = '0ms_Word'  # Original timing
window_size = 5
############
#End


shifted_dfs = {}

for wav in os.listdir(audio_folder):

    if wav.endswith(".wav"):
        #Get path of each audio file
        audio_path = os.path.join(audio_folder, wav)

        #Get the name without the extension
        base = os.path.splitext(wav)[0]

        #Split each word in the name
        speaker_type = base.split("_")[0]

        #Load Audio
        sr = 16000        #sampling rate
        waveform, sr = sf.read(audio_path)
        og_waveform = torch.tensor(waveform).T  # match torchaudio shape

        speaker_dir = os.path.join(output_folder, speaker_type)
        os.makedirs(speaker_dir, exist_ok=True)

        shifted_dfs = {}
        #Now, do the shifts
        for ms in shift_ms:

            output_file = os.path.join(speaker_dir, f"{speaker_type}_{ms}ms.csv" )
            
             # Skip if file already exists
            if os.path.exists(output_file):
                print(f"Skipping {speaker_type}_{ms} (already exists)")
    
                shifted_dfs[f"Shift_{speaker_type}_{ms}ms"] = pd.read_csv(output_file)
                continue
             
            
            shifted_audio = shift_audio(og_waveform, output_folder, ms, sr)
            shifted_df = process_audio(shifted_audio, sr, ms)     #FOR NEMO

            shifted_df.to_csv(
            os.path.join(speaker_dir, f"{speaker_type}_{ms}ms.csv"),
            index=False)

            # store in dict with a unique key
            shifted_dfs[f"Shift_{speaker_type}_{ms}ms"] = shifted_df
            print(f"Finished {speaker_type}_{ms}ms")



combined_df = combine_transcription(shifted_dfs)

print(combined_df.head())

sorted_df = align_words(combined_df, word_columns, timing_map, reference_col="0ms_word", window=3)

sorted_df.to_csv(f"{output_folder}/Nemo_combined_sorted.csv", index=False)

#Split into negative and positive
neg_pos_df = split_shifted_dfs(sorted_df)

final_timestamps = final_timestamps(neg_pos_df, output_folder)
