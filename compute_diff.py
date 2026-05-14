#Not using anymore
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

import wav2vec_start 
import process_alignment

from wav2vec_start  import find_shift


output_folder = process_alignment.output_folder

# get file
combined_df = process_alignment.sorted_df

# def split_find_shifts(combined_df):
#   num_cols = len(combined_df.columns)

#   idx_halfway = round((num_cols - 3) / 2) + 3
#   no_shift = combined_df[["word", "start", "end"]]

#   neg_split = combined_df.iloc[:,3:idx_halfway]

#   #Negative Shifts
#   neg_split = neg_split.iloc[:, ::-1]
#   neg_split = pd.concat([no_shift, neg_split], axis = 1)
#   neg_df = find_shift(neg_split)


#   #Positive Shifts
#   pos_split = combined_df.iloc[:, idx_halfway:]
#   pos_split = pd.concat([no_shift, pos_split],axis = 1)
#   pos_df = find_shift(pos_split)

#   neg_df = neg_df.rename(columns={'Start': 'Neg_Start'})
#   pos_df = pos_df.rename(columns={'Start': 'Pos_Start'})

#   neg_pos = pd.concat([no_shift, neg_df, pos_df], axis = 1)

#   return neg_pos

# def final_timestamps (neg_pos, output_folder):
#   final_df = []

#   for index, row in neg_pos.iterrows():
#     if abs(row['Pos_Start'] - row['start']) > abs(row['Neg_Start'] - row['start']) :

#       final_df.append({"Word": row.iloc[0],
#                       "Start": row['Neg_Start']})
#     else:
#       final_df.append({"Word": row.iloc[0],
#                       "Start": row['Pos_Start']})
      
#   final_df = pd.DataFrame(final_df)

#   final_df.to_csv(f"{output_folder}/final_timestamps.csv", index=False)

#   return final_df
