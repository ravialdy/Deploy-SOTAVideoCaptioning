import torch
import torchvision.transforms as transforms
from PIL import Image
import soundfile as sf
import numpy as np
import os
import cv2
import subprocess

def extract_frames_and_audio(video_path, output_dir, sr=22050):
    video_name = os.path.basename(video_path).replace(".mp4", "")
    
    # Extract all video frames
    all_frames_dir = os.path.join(output_dir, "all_frames", video_name)
    os.makedirs(all_frames_dir, exist_ok=True)
    cmd = f'ffmpeg -loglevel error -i "{video_path}" -vsync 0 -q:v 2 "{all_frames_dir}/frame_%04d.jpg"'
    subprocess.call(cmd, shell=True)

    # Check if video has audio stream
    has_audio = check_audio_stream(video_path)

    # Extract audio if present
    audio_file_path = None
    if has_audio:
        sr_audio_dir = os.path.join(output_dir, f"audio_{sr}hz")
        os.makedirs(sr_audio_dir, exist_ok=True)
        audio_file_path = os.path.join(sr_audio_dir, f"{video_name}.wav")
        cmd = f'ffmpeg -i "{video_path}" -loglevel error -f wav -vn -ac 1 -ab 16k -ar {sr} -y "{audio_file_path}"'
        subprocess.call(cmd, shell=True)

    return all_frames_dir, audio_file_path

def check_audio_stream(video_path):
    cmd = f'ffprobe -loglevel error -show_streams "{video_path}"'
    output = subprocess.check_output(cmd, shell=True, text=True)
    return 'codec_type=audio' in output

def preprocess_video(video_path, output_dir):
    all_frames_dir, audio_file_path = extract_frames_and_audio(video_path, output_dir)

    # Select 32 frames evenly
    frame_files = sorted(os.listdir(all_frames_dir))
    total_extracted_frames = len(frame_files)
    selected_frames = [frame_files[i] for i in sorted(np.round(np.linspace(0, total_extracted_frames - 1, 32)).astype(int))]

    # Convert selected frames to tensor
    frames = [Image.open(os.path.join(all_frames_dir, f)) for f in selected_frames]
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    video_pixels = torch.stack([transform(frame) for frame in frames]).unsqueeze(0)

    if audio_file_path and os.path.exists(audio_file_path):
        audio_data, _ = sf.read(audio_file_path)
        audio_spectrogram = np.abs(np.fft.fft(audio_data)[:audio_data.shape[0] // 2])
        audio_spectrograms = torch.tensor(audio_spectrogram).float().unsqueeze(0).unsqueeze(0)
    else:
        # Return empty tensor if no audio
        dummy_spatial_dim = (64, 512)
        audio_spectrograms = torch.zeros(1, 1, *dummy_spatial_dim)

    # import pdb; pdb.set_trace()

    return video_pixels, audio_spectrograms