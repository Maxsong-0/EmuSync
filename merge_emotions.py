import os
import pandas as pd
import glob
from datetime import datetime

def ensure_directory(directory):
    """Ensure the directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_latest_file(directory, pattern):
    """Get the latest file matching the pattern in the directory"""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def merge_emotions():
    """
    Multimodal Fusion
    - Align data using second-wise timestamps
    - Apply weight: video (70%), audio (30%)
    - Merge scores and sum matching emotions
    - Output top-2 emotions per second
    """
    ensure_directory('video_emotion')
    ensure_directory('audio_text_emotion')
    ensure_directory('merge_emotions')
    
    video_file = get_latest_file('video_emotion', 'emotion_analysis_*.csv')
    audio_file = get_latest_file('audio_text_emotion', 'session_*.csv')
    
    if not video_file or not audio_file:
        print("Error: Could not find required emotion analysis files")
        return
    
    print(f"Merging video emotions from: {video_file}")
    print(f"Merging audio emotions from: {audio_file}")
    
    video_df = pd.read_csv(video_file)
    audio_df = pd.read_csv(audio_file)
    
    if 'timestamp' not in video_df.columns or 'timestamp' not in audio_df.columns:
        print("Error: Required timestamp column missing in input files")
        return
    
    merged_df = pd.merge(video_df, audio_df, on='timestamp', suffixes=('_video', '_audio'))
    
    results = []
    
    for _, row in merged_df.iterrows():
        video_emotions = {
            row['emotion1_video']: row['score1_video'],
            row['emotion2_video']: row['score2_video'] if pd.notna(row['emotion2_video']) else 0
        }
        
        audio_emotions = {
            row['emotion1_audio']: row['score1_audio'],
            row['emotion2_audio']: row['score2_audio'] if pd.notna(row['emotion2_audio']) else 0
        }
        
        video_weight = 0.7
        audio_weight = 0.3
        
        combined_emotions = {}
        
        for emotion, score in video_emotions.items():
            if emotion in combined_emotions:
                combined_emotions[emotion] += score * video_weight
            else:
                combined_emotions[emotion] = score * video_weight
        
        for emotion, score in audio_emotions.items():
            if emotion in combined_emotions:
                combined_emotions[emotion] += score * audio_weight
            else:
                combined_emotions[emotion] = score * audio_weight
        
        sorted_emotions = sorted(combined_emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:2]
        
        results.append({
            'timestamp': row['timestamp'],
            'emotion1': top_emotions[0][0],
            'score1': top_emotions[0][1],
            'emotion2': top_emotions[1][0] if len(top_emotions) > 1 else None,
            'score2': top_emotions[1][1] if len(top_emotions) > 1 else 0
        })
    
    results_df = pd.DataFrame(results)
    
    output_path = os.path.join('merge_emotions', 'merged_emotions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Merged emotions saved to {output_path}")
    
    return results_df

def main():
    merge_emotions()

if __name__ == "__main__":
    main()
