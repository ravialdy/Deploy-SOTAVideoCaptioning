from flask import Flask, request, jsonify
from model_api import load_model, generate_caption
from train_utils import get_args, initialize

app = Flask(__name__)

# Load the model when the app starts
opts = get_args()
initialize(opts)
model = load_model("checkpoint/VALOR_base_msr_cap.pt", opts)

@app.route('/caption', methods=['POST'])
def caption_video():
    video_path = request.files['video']  # Assume the video is uploaded as a file
    video_path.save("temp_video.mp4")  # Save the uploaded video to a temporary file
    
    caption = generate_caption(model, "temp_video.mp4", "output_dir/")
    
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)