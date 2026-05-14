import os
import torch
import pandas as pd

import nemo.collections.asr as nemo_asr


asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")


config = {}

with open("config.txt", "r") as f:
    for line in f:
        # Skip empty lines or comments
        if "=" in line:
            name, value = line.split("=", 1)
            config[name.strip()] = value.strip()

# Call them
audio_folder = config["for_nemo"]


# read times
start = int(config["shift_start"])
end = int(config["shift_end"])
step = int(config["shift_step"])

shift_range = list(range(start, end + 1, step))

os.makedirs("Nemo_Output", exist_ok=True)
output_folder = "Nemo_Output"

 # Run ASR model
output = asr_model.transcribe(audio_folder, timestamps=True)

word_timestamps = output[0].timestamp['word']

word_data = []
for word in word_timestamps:
    #print(f"{word['start']}s - {word['end']}s : {word['word']}")

    word_data.append({"Word": word['word'],
                      "Start": word['start'],
                      "End": word['end']})

df_word = pd.DataFrame(word_data).round(decimals = 3)