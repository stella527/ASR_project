import pandas as pd
import os
import wav2vec_start
import process_wav2vec

from wav2vec_start import align_words

output_folder = process_wav2vec.speaker_dir

dfs = []

for file in os.listdir(output_folder):
    
    if file.endswith(".csv") and not file.startswith("combined"):
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


combined_df.to_csv(os.path.join(output_folder, "combined.csv"), index=False)

print("Done:", combined_df.shape)
combined_dir = pd.read_csv(f"{output_folder}/combined.csv")


word_columns = []
timing_map = {}
for i in list(range(-20,21)):
    if i != 0:
        word_columns.append((f"{i}ms_word"))
        timing_map[f"{i}ms_word"] = [f"{i}ms_start", f"{i}ms_end"]
    else:
        word_columns.append("word")
        timing_map["word"] = ["start", "end"]

reference_col = 'word'  # Original timing
window_size = 5


sorted_df = align_words(combined_dir, word_columns, timing_map, reference_col="word", window=3)

sorted_df.to_csv(f"{output_folder}/combined_sorted.csv", index=False)