#Not using anymore
import pandas as pd
import os
import wav2vec_start
import process_wav2vec

from wav2vec_start import align_words

output_folder = process_wav2vec.output_folder
shift_range = wav2vec_start.shift_range

#transcribed words and times
shifted_dfs = process_wav2vec.shifted_dfs

# def combine_transcription(shifted_dfs):
#     dfs = []

#     for filename, df in shifted_dfs.items():

#         #Split each word in the name
#         parts = filename.split("_")


#         # The timing value is in index 2
#         timing = parts[2]   

#             # Rename columns to timing_columnname
#         df.columns = [f"{timing}_{col}" for col in df.columns]

#         # Store
#         dfs.append(df)


#     #All the shifted csv appending
#     combined_df = pd.concat(dfs, axis=1)

#     return combined_df  



# word_columns = []
# timing_map = {}
# for i in shift_range:
#     word_columns.append((f"{i}ms_word"))
#     timing_map[f"{i}ms_word"] = [f"{i}ms_start", f"{i}ms_end"]


# reference_col = '0ms_word'  # Original timing
# window_size = 5

# sorted_df = align_words(combined_df, word_columns, timing_map, reference_col="0ms_word", window=3)

# sorted_df.to_csv(f"{output_folder}/Wav2vec_combined_sorted.csv", index=False)

