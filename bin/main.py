import os
import pandas as pd
import soundfile as sf
import sys

#import local files and functions here

sys.path.append("../") 
from lib import audio_processing, __init__

from lib.__init__ import asr_models

from lib.audio_processing import load_and_parse_audio
from lib.audio_processing import shift_audio
from lib.audio_processing import run_asr
from lib.audio_processing import combine_transcription
from lib.audio_processing import align_words
from lib.audio_processing import split_shifted_dfs
from lib.audio_processing import final_timestamps


#Create folder to store output
script_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.dirname(script_dir)                  
output_folder = os.path.join(parent_dir, "All Output Folder")
os.makedirs(output_folder, exist_ok=True)

audio_folder = audio_processing.audio_dir
#audio_folder = audio_processing.nemo_only

shift_ms = audio_processing.shift_range


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

        base = os.path.splitext(wav)[0]
        speaker_type = base.split("_")[0]

        audio_path, speaker_type, og_waveform, sr = load_and_parse_audio(audio_folder, wav)

        speaker_dir = os.path.join(output_folder, speaker_type)
        os.makedirs(speaker_dir, exist_ok=True)

        # Precompute shifted audio once per ms (reused across all models)
        shifted_audios = {}
        for ms in shift_ms:
            shifted_audios[ms] = shift_audio(og_waveform, speaker_dir, ms, sr)

        # ----------- Loop per model, collecting ALL ms shifts for that model -----------#
        for asr in asr_models:
            model_name = asr.__name__.split('.')[-1]
            model_name = model_name.split("_")[0]
            ind_models = os.path.join(speaker_dir, model_name)
            os.makedirs(ind_models, exist_ok=True)

            shifted_dfs = {}  # reset per model

            for ms in shift_ms:

                shifted_audio = shifted_audios[ms]

                output_file = os.path.join(ind_models, f"{speaker_type}_{model_name}_{ms}ms.csv")

                if os.path.exists(output_file):
                    print(f"Skipping {speaker_type}_{model_name}_{ms}ms (already exists)")
                    df_words = pd.read_csv(output_file)
                else:
                    df_words = run_asr(asr, shifted_audio, ms, sr)
                    df_words.to_csv(output_file, index=False)
                    print(f"Finished {speaker_type}_{model_name}_{ms}ms")

                shifted_dfs[f"Shift_{speaker_type}_{model_name}_{ms}ms"] = df_words
                

            # Now combine ALL ms shifts for this model in one go
            combined_df = combine_transcription(shifted_dfs)
            

            sorted_df = align_words(combined_df, word_columns, timing_map, reference_col="0ms_word", window=3)

            neg_pos_df = split_shifted_dfs(sorted_df)

            final_timestamps_df = final_timestamps(neg_pos_df, speaker_dir, speaker_type)

            final_timestamps_df.to_csv(
                os.path.join(ind_models, f"{speaker_type}_{model_name}_final_timestamps.csv"),
                index=False
            )