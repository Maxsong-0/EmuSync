import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_ollama():
    """Check if Ollama is installed and the Deepseek model is available"""
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if not result.stdout.strip():
            print("Warning: Ollama is not installed. Please install Ollama from https://ollama.ai/")
            return False
        
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "deepseek" not in result.stdout:
            print("Warning: Deepseek model is not available. Installing...")
            subprocess.run(["ollama", "pull", "deepseek"], check=True)
        
        return True
    except Exception as e:
        print(f"Error checking Ollama: {str(e)}")
        return False

def create_directories():
    """Create required directories"""
    directories = [
        'video_input',
        'separated_audio',
        'video_emotion',
        'audio_text_emotion',
        'merge_emotions',
        'prompt'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    """Main setup function"""
    print("Setting up EmuSync...")
    
    check_python_version()
    
    create_directories()
    
    install_requirements()
    
    ollama_available = check_ollama()
    
    print("\nSetup complete!")
    
    if not ollama_available:
        print("\nNote: Ollama is required for prompt generation. Please install Ollama and the Deepseek model.")
    
    print("\nUsage:")
    print("1. Place your input video at video_input/<filename>.mp4")
    print("2. Place the separated audio file at separated_audio/<filename>_audio.mp3")
    print("3. Run the scripts in order:")
    print("   python video_emotion.py")
    print("   python audio_text_emotion.py")
    print("   python merge_emotions.py")
    print("   python generate_prompt.py")
    print("4. Final Suno prompt will appear in prompt/<filename>.txt")

if __name__ == "__main__":
    main()
