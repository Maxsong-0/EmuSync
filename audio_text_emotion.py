import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import librosa
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from datetime import datetime

def create_timestamp():
    """Generate a timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(directory):
    """Ensure the directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_audio(audio_path, target_sr=16000):
    """
    Load audio file and convert to required format
    - 16kHz mono WAV
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None
    
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio, target_sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {str(e)}")
        return None

def process_audio(audio_path):
    """
    Process audio for emotion analysis
    - Split audio into 1-second chunks
    - Use FSER model to analyze top-2 emotional probabilities
    """
    audio_data = load_audio(audio_path)
    if audio_data is None:
        return None
    
    audio, sr = audio_data
    
    model_name = "superb/wav2vec2-large-superb-er"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    
    id2label = model.config.id2label
    
    chunk_size = sr  # 1 second of audio at the given sample rate
    results = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            values, indices = torch.topk(probs, 2, dim=-1)
            
            second_idx = i // chunk_size
            
            results.append({
                'timestamp': second_idx,
                'emotion1': id2label[indices[0][0].item()],
                'score1': values[0][0].item(),
                'emotion2': id2label[indices[0][1].item()],
                'score2': values[0][1].item()
            })
        
        if i % (10 * chunk_size) == 0:
            progress = (i / len(audio)) * 100
            print(f"Progress: {progress:.2f}%")
    
    return pd.DataFrame(results)

def main():
    ensure_directory('separated_audio')
    ensure_directory('audio_text_emotion')
    
    audio_files = [f for f in os.listdir('separated_audio') if f.endswith('_audio.mp3')]
    
    if not audio_files:
        print("No audio files found in separated_audio directory")
        return
    
    for audio_file in audio_files:
        audio_path = os.path.join('separated_audio', audio_file)
        print(f"Processing audio: {audio_path}")
        
        base_filename = os.path.splitext(audio_file)[0]
        
        results_df = process_audio(audio_path)
        
        if results_df is not None:
            timestamp = create_timestamp()
            output_path = os.path.join('audio_text_emotion', f'session_{timestamp}.csv')
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
