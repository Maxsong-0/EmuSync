import os
import cv2
import time
import pandas as pd
import numpy as np
from deepface import DeepFace
from datetime import datetime

def create_timestamp():
    """Generate a timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(directory):
    """Ensure the directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_video(video_path):
    """
    Process video for emotion analysis
    - Process every 10 frames
    - Aggregate per-second dominant emotions
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Duration: {duration:.2f} seconds")
    
    frame_idx = 0
    second_idx = 0
    emotions_per_second = {}
    
    emotion_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 10 == 0:
            current_second = int(frame_idx / fps)
            
            if current_second not in emotions_per_second:
                emotions_per_second[current_second] = {emotion: 0 for emotion in emotion_list}
            
            try:
                result = DeepFace.analyze(frame, 
                                         actions=['emotion'],
                                         detector_backend='mtcnn',
                                         enforce_detection=False)
                
                if isinstance(result, list):
                    emotion_data = result[0]['emotion']
                else:
                    emotion_data = result['emotion']
                
                for emotion, score in emotion_data.items():
                    if emotion in emotions_per_second[current_second]:
                        emotions_per_second[current_second][emotion] += score
                
                print(f"Processed frame {frame_idx} at second {current_second}")
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            progress = (frame_idx / frame_count) * 100
            print(f"Progress: {progress:.2f}%")
    
    cap.release()
    
    results = []
    for second, emotions in sorted(emotions_per_second.items()):
        total = sum(emotions.values())
        if total > 0:
            normalized_emotions = {e: s/total for e, s in emotions.items()}
        else:
            normalized_emotions = emotions
        
        sorted_emotions = sorted(normalized_emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:2]
        
        results.append({
            'timestamp': second,
            'emotion1': top_emotions[0][0],
            'score1': top_emotions[0][1],
            'emotion2': top_emotions[1][0] if len(top_emotions) > 1 else None,
            'score2': top_emotions[1][1] if len(top_emotions) > 1 else 0
        })
    
    return pd.DataFrame(results)

def main():
    ensure_directory('video_input')
    ensure_directory('video_emotion')
    
    video_files = [f for f in os.listdir('video_input') if f.endswith('.mp4')]
    
    if not video_files:
        print("No video files found in video_input directory")
        return
    
    for video_file in video_files:
        video_path = os.path.join('video_input', video_file)
        print(f"Processing video: {video_path}")
        
        base_filename = os.path.splitext(video_file)[0]
        
        results_df = process_video(video_path)
        
        if results_df is not None:
            timestamp = create_timestamp()
            output_path = os.path.join('video_emotion', f'emotion_analysis_{timestamp}.csv')
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
