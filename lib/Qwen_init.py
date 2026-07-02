import torch
import torchaudio
import pandas as pd
import os

from qwen_asr import Qwen3ASRModel


model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        # attn_implementation="flash_attention_2",
    ),
)

def process_qwen(shifted_audio):

    # Save it to a temporary file
    temp_filename = "temp_shifted_audio.wav"
    torchaudio.save(temp_filename, shifted_audio.cpu(), sample_rate=16000)

    results = model.transcribe(audio=temp_filename, language=["English"], return_time_stamps=True,)

    # Remove temp after transcription
    os.remove("temp_shifted_audio.wav")

    word_data = []
    for r in results:
        
        for word in r.time_stamps:

            word_data.append({
                            "word": word.text,
                            "start": word.start_time,
                            "end": word.end_time

                    })
        
    df_word = pd.DataFrame(word_data).round(decimals = 3)


    return df_word


# for filename in os.listdir(audio_folder):
#     if filename.endswith(".wav"):
#         wav_path = os.path.join(audio_folder, filename)

#         word_df = process_audio(wav_path)

#         print("File:", filename)
#         print(word_df.tail(10))



