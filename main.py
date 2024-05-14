import torch
import torchaudio
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

# Configuraciones de los modelos
encoder_model_fpath = Path("Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt")
synthesizer_model_fpath = Path("Real-Time-Voice-Cloning/synthesizer/saved_models/pretrained/pretrained.pt")
vocoder_model_fpath = Path("Real-Time-Voice-Cloning/vocoder/saved_models/pretrained/pretrained.pt")

# Cargar el modelo del sintetizador y el vocoder
synthesizer = Synthesizer(synthesizer_model_fpath)
vocoder.load_model(vocoder_model_fpath)
encoder.load_model(encoder_model_fpath)

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != synthesizer.sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=synthesizer.sample_rate)(waveform)
    return waveform.numpy().flatten()

def clone_voice(text, audio_path):
    # Preprocesar el archivo de voz clonada
    original_wav = preprocess_audio(audio_path)
    embedding = encoder.embed_utterance(original_wav)

    # Generar el espectrograma mel
    specs = synthesizer.synthesize_spectrograms([text], [embedding])
    spec = specs[0]

    # Generar el archivo de voz clonada
    generated_wav = vocoder.infer_waveform(spec)

    # Post-procesamiento
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Guardar el archivo de voz clonada
    output_path = "cloned_voice_output.wav"
    wavfile.write(output_path, synthesizer.sample_rate, generated_wav.astype(np.float32))
    print(f"Archivo de voz clonada guardado en: {output_path}")

if __name__ == "__main__":
    texto = "Hola, esta es una prueba de clonación de voz."
    audio_mp3_path = "path_to_your_mp3_file.mp3"
    clone_voice(texto, audio_mp3_path)
