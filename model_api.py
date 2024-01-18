import torch
import torch.nn.functional as F
from model.pretrain import VALOR
import json
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb=10"

import easydict
from utils.misc import NoOp, parse_with_config, set_random_seed, str2bool
from train_utils import initialize
from deploy.preprocess_vid import preprocess_video

def get_args(config_path='config_deploy/deploy_base.json'):
    """
    Load configuration from a JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Convert the dictionary to an argparse.Namespace-like object
    args = easydict.EasyDict(config)
    return args

def load_model(checkpoint_path, opts):
    """
    Load the pre-trained VALOR model from a checkpoint.
    """
    initialize(opts)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:1'))
    model = VALOR.from_pretrained(opts, checkpoint)
    model.eval()
    return model

def generate_caption(model, video_path, device):
    """
    Generate a caption for a given video using the VALOR model.
    """

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("output_dir", video_name)
    os.makedirs(output_dir, exist_ok=True)

    video_pixels, audio_spectrograms = preprocess_video(video_path, output_dir)
    
    batch = {
        'video_pixels': video_pixels.to(device),
        'audio_spectrograms': audio_spectrograms.to(device)
    }

    task_str = "cap%tva"
    evaluation_dict = model(batch, task_str, compute_loss=False, inference_only=True)
    
    sents = evaluation_dict['generated_sequences_t_va']

    # Decode the tensor into human-readable sentences
    sents = model.decode_sequence(sents.data)
    
    return sents[0]


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    opts = get_args()
    # import pdb; pdb.set_trace()
    model = load_model("checkpoint/VALOR_base_msr_cap.pt", opts).to(device)
    caption = generate_caption(model, "sample/video0.mp4", device)
    print("Generated Caption:", caption)