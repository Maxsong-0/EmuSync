import os
import pandas as pd
import subprocess
import json
import time
from datetime import datetime

def ensure_directory(directory):
    """Ensure the directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_latest_file(directory, pattern):
    """Get the latest file matching the pattern in the directory"""
    import glob
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def format_emotion_data(df):
    """Format emotion data for prompt generation"""
    formatted_data = []
    
    for _, row in df.iterrows():
        line = f"{row['emotion1']} {row['score1']:.2f}"
        
        if pd.notna(row['emotion2']) and row['score2'] > 0:
            line += f" {row['emotion2']} {row['score2']:.2f}"
        
        formatted_data.append(line)
    
    return "\n".join(formatted_data)

def generate_prompt_with_ollama(emotion_data):
    """
    Generate Suno-compatible prompt using Ollama with Deepseek model
    """
    system_prompt = """
    You are an expert at analyzing emotional patterns and creating descriptive prompts for music generation.
    Given a sequence of emotions with their intensities, create a Suno-compatible English prompt that captures the emotional trajectory.
    Your prompt should be concise (1-3 sentences), evocative, and suitable for generating music that follows the emotional pattern.
    Focus on the emotional journey, transitions, and overall mood without explicitly mentioning all the technical emotion labels.
    """
    
    user_prompt = f"""
    Below is a sequence of emotions with their intensities (each line represents one second of content):
    
    {emotion_data}
    
    Create a Suno-compatible English prompt that captures this emotional trajectory for music generation.
    """
    
    payload = json.dumps({
        "model": "deepseek",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    })
    
    try:
        result = subprocess.run(["pgrep", "ollama"], capture_output=True, text=True)
        if not result.stdout.strip():
            print("Warning: Ollama does not appear to be running. Starting Ollama...")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)
        
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/chat", "-d", payload],
            capture_output=True,
            text=True
        )
        
        response = json.loads(result.stdout)
        
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"].strip()
        else:
            print("Error: Unexpected response format from Ollama")
            print(response)
            return None
    
    except Exception as e:
        print(f"Error generating prompt with Ollama: {str(e)}")
        return None

def main():
    ensure_directory('merge_emotions')
    ensure_directory('prompt')
    
    merged_file = get_latest_file('merge_emotions', 'merged_emotions.csv')
    
    if not merged_file:
        print("Error: Could not find merged emotions file")
        return
    
    print(f"Generating prompt from: {merged_file}")
    
    merged_df = pd.read_csv(merged_file)
    
    emotion_data = format_emotion_data(merged_df)
    
    prompt = generate_prompt_with_ollama(emotion_data)
    
    if prompt:
        base_filename = os.path.splitext(os.path.basename(merged_file))[0]
        
        output_path = os.path.join('prompt', f"{base_filename}.txt")
        with open(output_path, 'w') as f:
            f.write(prompt)
        
        print(f"Prompt saved to {output_path}")
        print("\nGenerated Prompt:")
        print(prompt)

if __name__ == "__main__":
    main()
