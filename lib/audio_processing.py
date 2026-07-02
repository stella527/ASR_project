import torch
import pandas as pd
import re
import os
import soundfile as sf
import sys

from . import Qwen_init, nemo_init, wav2vec_init
from .wav2vec_init import decoding_to_timings
from .nemo_init import process_nemo
from .Qwen_init import process_qwen

sys.path.append("../") 
import user_config
from user_config import audio_folder, shift_start, shift_end, shift_step



# Call audio
audio_dir = audio_folder

# Read Times
start = shift_start
end = shift_end
step = shift_step
shift_range = list(range(start, end + 1, step))



def load_and_parse_audio(audio_folder, wav):

    audio_path = os.path.join(audio_folder, wav)

    # name without extension
    base = os.path.splitext(wav)[0]

    # speaker type
    speaker_type = base.split("_")[0]

    # load audio
    waveform, sr = sf.read(audio_path)
    og_waveform = torch.tensor(waveform)

    return audio_path, speaker_type, og_waveform, sr

def shift_audio(og_waveform, output_dir, shift_ms, sr):
    
     # Create file path for the shifted version
    #base_name = os.path.join(output_dir, f"audio_shift_{shift_ms}ms.wav")
   
    shift_samples = int(sr * (shift_ms / 1000.0))

    if og_waveform.dim() == 1:
        og_waveform = og_waveform.unsqueeze(0)
        
    if shift_samples > 0:
        shifted = torch.cat([torch.zeros(1, shift_samples), og_waveform[:, :-shift_samples]], dim=1)
    elif shift_samples < 0:
        shifted = torch.cat([og_waveform[:, -shift_samples:], torch.zeros(1, -shift_samples)], dim=1)
    else:
        shifted = og_waveform

     # Save the shifted version
    #torchaudio.save(base_name, shifted, sr)

    return shifted

def run_asr(model, audio, ms, sr):

    if model is nemo_init:
        df_word = nemo_init.process_nemo(audio)

    if model is wav2vec_init:
        df_word = wav2vec_init.process_shift(audio, ms, sr, wav2vec_init.processor, wav2vec_init.model, decoding_to_timings)

    if model is Qwen_init:
        df_word = Qwen_init.process_qwen(audio)

    return df_word


def align_words(df,word_columns, timing_map, reference_col="0ms_word", window=3):
    
    aligned_rows = []

    # Lowercase version for matching
    df_words = df[word_columns].apply(lambda col: col.map(lambda x: str(x).lower() if isinstance(x, str) else ''))
    df_ref = df[reference_col].apply(lambda x: str(x).lower() if isinstance(x, str) else '')

    for i, ref_word in df_ref.items():
        if not ref_word:
            continue

        match_info = {'word': df.at[i, reference_col], 
                      'start': df.at[i, '0ms_start'], 
                        'end': df.at[i, '0ms_end']}
        all_found = True

        
        for col in word_columns:

            # Check same row first
            if df_words.at[i, col] == ref_word:
                idx_match = i
            else:
                # search window
                window_start = max(0, i - window)
                window_end = min(len(df), i + window + 1)
                window_rows = df_words[col].iloc[window_start:window_end]
                found_rows = window_rows[window_rows == ref_word]
                if len(found_rows) == 0:
                    all_found = False
                    break
                idx_match = found_rows.index[0]

            # add timing info for this column
            if col in timing_map:
                for timing_col in timing_map[col]:
                    match_info[timing_col] = df.at[idx_match, timing_col]
            else: 
                print("Check timing map and cross reference the columns in df")

        if all_found:
            aligned_rows.append(match_info)
    aligned_df = pd.DataFrame(aligned_rows).reset_index(drop=True)
    
    return aligned_df


def find_shift(df):
    new_df = []
    for idx, row in df.iterrows():
    
        #First, check if all words for each time shift is present
        first_value = row.iloc[1]
        
        count = 0
    
        for col_name, col_value in row.items():
            if col_name.endswith("start"):
                if col_value == first_value:
                    count += 1
                else:
                    num = int(re.search(r"-?\d+", col_name).group())
    
                    #Subtract the offset from the shifted time
                    new_time = col_value - (num/1000)
                    
                    new_df.append({
                    "Word": row.iloc[0],
                    "Start": round(new_time, 3)
                    })
                    
                    break
        else:
             new_df.append({
                    "Word": row.iloc[0],
                    "Start": round(col_value, 3)
                    })
    
    return pd.DataFrame(new_df)

def split_shifted_dfs(combined_df):
    num_cols = len(combined_df.columns)

    idx_halfway = round((num_cols - 3) / 2) + 3
    no_shift = combined_df[["word", "start", "end"]]

    neg_split = combined_df.iloc[:,3:idx_halfway]

    #Negative Shifts
    neg_split = neg_split.iloc[:, ::-1]
    neg_split = pd.concat([no_shift, neg_split], axis = 1)
    neg_df = find_shift(neg_split)

    #Positive Shifts
    pos_split = combined_df.iloc[:, idx_halfway:]
    pos_split = pd.concat([no_shift, pos_split],axis = 1)
    pos_df = find_shift(pos_split)

    neg_df = neg_df.rename(columns={'Start': 'Neg_Start'})
    pos_df = pos_df.rename(columns={'Start': 'Pos_Start'})

    neg_pos = pd.concat([no_shift, neg_df, pos_df], axis = 1)

    return neg_pos


def final_timestamps(neg_pos, output_folder, speaker_type):
    final_df = []

    for index, row in neg_pos.iterrows():
        if abs(row['Pos_Start'] - row['start']) > abs(row['Neg_Start'] - row['start']) :

            final_df.append({"Word": row.iloc[0],
                            "Start": row['Neg_Start']})
        else:
            final_df.append({"Word": row.iloc[0],
                            "Start": row['Pos_Start']})
        
    final_df = pd.DataFrame(final_df)


    return final_df


def combine_transcription(shifted_dfs):

    dfs = []

    for filename, df in shifted_dfs.items():

        # Split each word in the name
        parts = filename.split("_")


        # The timing value is in index 2
        timing = parts[3]   
        

        # Rename columns to timing_columnname
        df.columns = [f"{timing}_{col}" for col in df.columns]

        # Store
        dfs.append(df)

    # All the shifted csv appending
    combined_df = pd.concat(dfs, axis=1)

    return combined_df  