import os
import pandas as pd
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC,  Wav2Vec2CTCTokenizer

#Local functions
import general
import wav2vec_start

from general  import shift_audio
from general import combine_transcription
from general import align_words
from general import split_shifted_dfs
from general import final_timestamps

from wav2vec_start  import process_shift
from wav2vec_start  import decoding_to_timings


audio_folder = general.audio_folder
shift_ms = general.shift_range

output_folder = wav2vec_start.output_folder


#######################
# Timing map for shifting
# Include in each model
#########################

word_columns = []
timing_map = {}
for i in shift_ms:
    word_columns.append((f"{i}ms_word"))
    timing_map[f"{i}ms_word"] = [f"{i}ms_start", f"{i}ms_end"]


reference_col = '0ms_word'  # Original timing
window_size = 5
############
#End


shifted_dfs = {}

for wav in os.listdir(audio_folder):

    if wav.endswith(".wav"):

        #Get the name without the extension
        base = os.path.splitext(wav)[0]

        #Split each word in the name
        speaker_type = base.split("_")[0]

        #Get path of each audio file
        audio_path = os.path.join(audio_folder, wav)
        
        og_waveform, sr = torchaudio.load(audio_path)
        
        print("Loading:", audio_path)

        speaker_dir = os.path.join(output_folder, speaker_type)
        os.makedirs(speaker_dir, exist_ok=True)
        
        for ms in shift_ms:
            
             output_file = os.path.join(speaker_dir, f"{speaker_type}_{ms}ms.csv")

            # Skip if file already exists
             if os.path.exists(output_file):
                print(f"Skipping {speaker_type}_{ms} (already exists)")
        
        
                shifted_dfs[f"Shift_{speaker_type}_{ms}ms"] = pd.read_csv(output_file)
                continue

             shifted_audio = shift_audio(og_waveform, output_folder, ms, sr)
        
        
             shifted_df = process_shift(
                shifted_audio,
                ms,
                sr,
                wav2vec_start.processor,
                wav2vec_start.model,
                decoding_to_timings)

             shifted_df.to_csv(os.path.join(speaker_dir, f"{speaker_type}_{ms}ms.csv"), index=False)
            
            # store in dict with a unique key
             shifted_dfs[f"Shift_{speaker_type}_{ms}ms"] = shifted_df
        
             print(f"Finished {speaker_type}_{ms}ms")


combined_df = combine_transcription(shifted_dfs)

sorted_df = align_words(combined_df, word_columns, timing_map, reference_col="0ms_word", window=3)

sorted_df.to_csv(f"{output_folder}/Wav2vec_combined_sorted.csv", index=False)

#Split into negative and positive
neg_pos_df = split_shifted_dfs(sorted_df)

final_timestamps = final_timestamps(neg_pos_df, output_folder)