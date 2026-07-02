import pandas as pd

import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

#Get transcription
def process_nemo(shifted_audio):

    # Convert torch.Tensor to numpy for librosa
    speech_np = shifted_audio.squeeze().cpu().numpy()

     # Run ASR model
    output = asr_model.transcribe(speech_np, timestamps=True)

    # by default, timestamps are enabled for char, word and segment level
    word_timestamps = output[0].timestamp['word'] # word level timestamps for first sample

    word_data = []
    for word in word_timestamps:
        word_data.append({"word": word['word'],
                          "start": word['start'],
                          "end": word['end']})

    df_word = pd.DataFrame(word_data).round(decimals = 3)

    return df_word