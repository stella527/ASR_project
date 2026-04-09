from itertools import groupby
import torch
import transformers 
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC,  Wav2Vec2CTCTokenizer#, Wav2Vec2PhonemeCTCTokenizer
import librosa
import argparse
import eng_to_ipa as ipa
import pandas as pd
import numpy as np
from phonemizer import phonemize
import torchaudio
import os
import re
import json



""""""

arpabet2ipa = {
    'AA':'ɑ',
    'AA0':'a',
    'AA1':'ɐ',
    'AE':'æ',
    'AH':'ʌ',
    'AH0':'ə',
    'AO':'ɔ',
    'AW':'aʊ',
    'AY':'aɪ',
    'EH':'ɛ',
    'ER':'ɝ',
    'ER0':'ɚ',
    'ER1': 'ər',
    'EY':'eɪ',
    'IH':'ɪ',
    'IH0':'ɨ',
    'IY':'i',
    'O':'o',
    'OW':'oʊ',
    'OY':'ɔɪ',
    'OY0':'oɪ',
    'UH':'ʊ',
    'UW':'u',
    'B':'b',
    'CH':'tʃ',
    'CH0':'ʧ',
    'D':'d',
    'DH':'ð',
    'EL':'l̩ ',
    'EM':'m̩',
    'EN':'n̩',
    'F':'f',
    'G':'g',
    'G0':'ɡ',
    'HH':'h',
    'JH':'dʒ',
    'JH0':'ʤ',
    'K':'k',
    'L':'l',
    'M':'m',
    'N':'n',
    'NG':'ŋ',
    'P':'p',
    'Q':'ʔ',
    'R':'ɹ',
    'RR':'r',
    'S':'s',
    'SH':'ʃ',
    'T':'t',
    'T0':'ɾ',
    'TH':'θ',
    'V':'v',
    'W':'w',
    'WH':'ʍ',
    'Y':'j',
    'Z':'z',
    'ZH':'ʒ'
}

ipa2arpabet = dict([(value, key) for key, value in arpabet2ipa.items()])
arpabet = list(arpabet2ipa.keys())

# corrections were determined out of band in a separate script; phonemes with
# less than 20 instances were set to 0 bias due to lack of data.
correction = {
    'N/A': 0.0, # phoneme not recognized, default to word onset time (0 bias)
    'a': 0.0,
    'aɪ': 0.0,
    'aʊ': 0.0,
    'b': 0.0273137254902025,
    'd': 0.010840996877361064,
    'dʒ': 0.0,
    'eɪ': 0.0,
    'f': 0.073921307102367,
    'h': 0.032686324714017445,
    'i': 0.08094752107244707,
    'j': 0.0,
    'k': 0.007710410189417871,
    'l': 0.06733196046128143,
    'm': 0.06112231113256161,
    'n': 0.05480637254902376,
    'o': 0.0,
    'oʊ': 0.0,
    'oː': 0.0,
    'p': 0.00361545986718248,
    's': 0.09089372726133593,
    't': 0.011572554518172629,
    'tʃ': 0.0,
    'u': 0.0,
    'v': 0.0,
    'w': 0.04421147058823749,
    'æ': 0.04221674054039681,
    'ð': 0.012261764496436456,
    'ŋ': 0.0,
    'ɐ': 0.029734671333123686,
    'ɑ': 0.0,
    'ɔ': 0.0,
    'ə': 0.040999408453004094,
    'ɚ': 0.0,
    'ɛ': 0.0,
    'ɡ': 0.04172284737497023,
    'ɪ': 0.03661764705882575,
    'ɹ': 0.056741440234150176,
    'ɾ': 0.0,
    'ʃ': 0.0,
    'ʊ': 0.0,
    'ʌ': 0.03150195552649926,
    'θ': 0.0
}

argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--file", type=str, help="wav file to process")
argParser.add_argument("-d", "--dest", type=str, help="path to save predictor")
args = argParser.parse_args()

################################################################################
# load model and audio and run audio through model
################################################################################

# load pretrained models
model_name = 'facebook/wav2vec2-large-960h-lv60-self'
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)
phoneme_model_name = 'facebook/wav2vec2-xlsr-53-espeak-cv-ft'
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(phoneme_model_name)
phoneme_model = Wav2Vec2ForCTC.from_pretrained(phoneme_model_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(phoneme_model_name)
phoneme_processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer = tokenizer)


# output_folder = "Wav2Vec_Shifted"
# audio_folder = "Female_Single_Speakers"


config = {}

with open("config.txt", "r") as f:
    for line in f:
        # Skip empty lines or comments
        if "=" in line:
            name, value = line.split("=", 1)
            config[name.strip()] = value.strip()

# Call them
audio_folder = config["audio_folder"]


# read times
start = int(config["shift_start"])
end = int(config["shift_end"])
step = int(config["shift_step"])

shift_range = list(range(start, end + 1, step))

os.makedirs("Wav2Vec2_all_shifts", exist_ok=True)
output_folder = "Wav2Vec2_all_shifts"



#Functions

def decoding_to_timings(transcription, predicted_ids, input_values, processor, 
                        sr):
    symbols = [s for s in transcription.split(' ') if len(s) > 0]
    predicted_ids = predicted_ids[0].tolist()
    duration_sec = input_values.shape[1] / sr


    ids_s_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in \
                  enumerate(predicted_ids)]

    # now split the ids into groups of ids where each group represents a symbol
    if isinstance(processor.tokenizer, transformers.models.wav2vec2_phoneme.\
                  tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizer): 
        split_ids_s_time = [list(group) for k, group
                        in groupby(ids_s_time, lambda x: x[1])
                        if k != phoneme_processor.tokenizer.pad_token_id]
    elif isinstance(processor.tokenizer, transformers.models.wav2vec2.\
                    tokenization_wav2vec2.Wav2Vec2CTCTokenizer):
        ids_s_time = [i for i in ids_s_time \
                      if i[1] != processor.tokenizer.pad_token_id]
        split_ids_s_time = [list(group) for k, group in groupby(ids_s_time, \
                lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
                if not k]
    else:
        raise Exception(f"tokenizer {type(processor.tokenizer)} not supported")

    # make sure that there are the same number of id-groups as words. 
    # otherwise something is wrong
    assert len(split_ids_s_time) == len(symbols)  

    symbol_start_times = []
    symbol_end_times = []
    for cur_ids_s_time, cur_symbol in zip(split_ids_s_time, symbols):
        _times = [_time for _time, _id in cur_ids_s_time]
        symbol_start_times.append(min(_times))
        symbol_end_times.append(max(_times))


    word_data = []

    for w, s, e in zip(symbols, symbol_start_times, symbol_end_times):
        word_data.append({
            "word": w,
            "start": s,
            "end": e
        })
    
    df = pd.DataFrame(word_data)

    return df

def shift_audio(og_waveform, shift_ms, sr):
    
    shift_samples = int(sr * (shift_ms / 1000.0))

    if og_waveform.dim() == 1:
        og_waveform = og_waveform.unsqueeze(0)
        
    if shift_samples > 0:
        shifted = torch.cat([torch.zeros(1, shift_samples), og_waveform[:, :-shift_samples]], dim=1)
    elif shift_samples < 0:
        shifted = torch.cat([og_waveform[:, -shift_samples:], torch.zeros(1, -shift_samples)], dim=1)
    else:
        shifted = og_waveform
    return shifted


def process_shift( og_waveform, ms, sr, output_folder, processor, model, decoding_to_timings):


    #assert (not np.all(shifted_audio.detach().cpu().numpy() == og_waveform.detach().cpu().numpy().flatten()))
    shifted_audio = shift_audio(og_waveform, ms, sr)
    
     # Create file path for the shifted version
    base_name = f"{output_folder}/audio_shift_{ms}ms.wav"

    # Save the shifted version
    torchaudio.save(base_name, shifted_audio, sr)

    #Now inputing new audio
    speech, sample_rate = librosa.load(base_name)

    
    # pretrained models need 16kHz, final timings are in clock times, so resampling
    # outputs is not necessary
    target_sample_rate = 16000
    speech = librosa.resample(speech, orig_sr=sample_rate, 
                              target_sr=target_sample_rate)
    
    # run speech through processors
    a = processor(speech, sampling_rate=target_sample_rate, return_tensors="pt")
    b = phoneme_processor(speech, sampling_rate=target_sample_rate, 
                          return_tensors="pt")
    
    input_values = a.input_values
    p_input_values = b.input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
        p_logits = phoneme_model(p_input_values).logits
    
    # decode model outputs
    predicted_ids = torch.argmax(logits, dim=-1)
    p_predicted_ids = torch.argmax(p_logits, dim=-1)
    
    transcription = processor.decode(predicted_ids[0]).lower()
    phonemes = phoneme_processor.decode(p_predicted_ids[0]).lower()
    #phonemes = phoneme_processor.batch_decode(p_predicted_ids)

    df  = decoding_to_timings(transcription, \
        predicted_ids, input_values, processor, target_sample_rate)
    
    round_df = df.round(decimals = 3)

    round_df.rename(columns={"Start": f"Start_{ms}", "End": f"End_{ms}"})

    return round_df


