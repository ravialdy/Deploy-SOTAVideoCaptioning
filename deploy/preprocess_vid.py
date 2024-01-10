import torch
import torchvision.transforms as transforms
from PIL import Image
import soundfile as sf
import numpy as np
import os
import subprocess

def extract_frames_and_audio(video_path, output_dir, fps=4, sr=22050):
    video_name = os.path.basename(video_path).replace(".mp4", "")
    
    # Extract video frames at fps
    fps_frame_dir = os.path.join(output_dir, f"frames_fps{fps}", video_name)
    os.makedirs(fps_frame_dir, exist_ok=True)
    cmd = f"ffmpeg -loglevel error -i {video_path} -vsync 0 -f image2 -vf fps=fps={fps} -qscale:v 2 {fps_frame_dir}/frame_%04d.jpg"
    subprocess.call(cmd, shell=True)

    # Extract audio at sr
    sr_audio_dir = os.path.join(output_dir, f"audio_{sr}hz")
    os.makedirs(sr_audio_dir, exist_ok=True)
    audio_file_path = os.path.join(sr_audio_dir, f"{video_name}.wav")
    cmd = f"ffmpeg -i {video_path} -loglevel error -f wav -vn -ac 1 -ab 16k -ar {sr} -y {audio_file_path}"
    subprocess.call(cmd, shell=True)

    return fps_frame_dir, audio_file_path

def preprocess_video(video_path, output_dir):
    # Extract frames and audio
    fps_frame_dir, audio_file_path = extract_frames_and_audio(video_path, output_dir)
    
    # Convert frames to tensor
    frame_files = sorted(os.listdir(fps_frame_dir))
    frames = [Image.open(os.path.join(fps_frame_dir, f)) for f in frame_files]
    transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize to the input size of your model
                                    transforms.ToTensor()])
    video_pixels = torch.stack([transform(frame) for frame in frames]).unsqueeze(0)  # Add batch dimension
    
    # Convert audio to tensor
    audio_data, _ = sf.read(audio_file_path)
    audio_spectrogram = np.abs(np.fft.fft(audio_data)[:audio_data.shape[0] // 2])
    audio_spectrograms = torch.tensor(audio_spectrogram).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    return video_pixels, audio_spectrograms