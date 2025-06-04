# from flask import Flask, request, render_template
# import os
# import joblib
# import pandas as pd
# import numpy as np
# from collections import Counter
# from utils.keypoints import extract_keypoints
# from tensorflow.keras.models import load_model

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Ensure model files exist before loading
# model_paths = {
#     "Autoencoder_Classifier_CNN": "newModels/Autoencoder_Classifier.h5",
#     "CNN_GRU": "newModels/CNN_GRU.h5",
#     "CNN_LSTM": "newModels/CNN_LSTM.h5",
#     "RNN_CNN": "newModels/RNN.h5"
# }

# # Load models
# models = {}
# for model_name, path in model_paths.items():
#     if os.path.exists(path):
#         models[model_name] = load_model(path)
#         print(f"✅ Loaded {model_name}")
#     else:
#         print(f"❌ Model file missing: {path}")

# # Load scaler
# scaler_path = "newModels/scaler.pkl"
# scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# # Class labels
# class_labels = {
#     0: "Normal",
#     1: "Limping",
#     2: "Slouch",
#     3: "No Arm Swing",
#     4: "Circumduction",
# }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'video' not in request.files:
#         return render_template('index.html', error="❌ No video file uploaded.")

#     file = request.files['video']
#     video_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(video_path)

#     output_video_path = "static/output.mp4"

#     # Extract keypoints
#     keypoints = extract_keypoints(video_path, output_video_path)
#     if isinstance(keypoints, dict) and 'error' in keypoints:
#         return render_template('index.html', error=keypoints['error'])

#     keypoints_np = keypoints.to_numpy()

#     predictions_summary = {}
#     final_votes = Counter()

#     for model_name, model in models.items():
#         try:
#             if scaler:
#                 keypoints_scaled = scaler.transform(keypoints_np)

#             # Reshape input for CNN and LSTM models
#             if "CNN" in model_name or "LSTM" in model_name:
#                 keypoints_scaled = keypoints_scaled.reshape((keypoints_scaled.shape[0], 33, 4))

#             # Make prediction
#             model_output = model.predict(keypoints_scaled)

#             # Convert predictions to class labels
#             if "CNN" in model_name or "LSTM" in model_name:
#                 predicted_classes = np.argmax(model_output, axis=1)
#                 predicted_labels = [class_labels[pred] for pred in predicted_classes]
#             else:
#                 predicted_labels = [class_labels[pred] for pred in model_output]

#             # Count occurrences of each class
#             class_count = dict(Counter(predicted_labels))
#             predictions_summary[model_name] = class_count
#             final_votes.update(class_count)

#         except Exception as e:
#             predictions_summary[model_name] = {"Error": f"⚠️ {str(e)}"}

#     # Get the most common predicted class
#     final_prediction = final_votes.most_common(1)[0][0] if final_votes else "Unknown"

#     return render_template(
#         'index.html',
#         predictions=predictions_summary,
#         final_votes=dict(final_votes),
#         final_prediction=final_prediction,
#         video_url=output_video_path
#     )

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 10000))
#     app.run(debug=True, host='0.0.0.0', port=port, extra_files=['templates/index.html'])


from flask import Flask, request, render_template, url_for
import os
import joblib
import pandas as pd
import numpy as np
import glob
from collections import Counter
from utils.keypoints import extract_keypoints
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure model files exist before loading
model_paths = {
    "Autoencoder_Classifier_CNN": "Models/Autoencoder_Classifier.h5",
    "CNN_GRU": "Models/CNN_GRU.h5",
    "CNN_LSTM": "Models/CNN_LSTM.h5",
    "RNN_CNN": "Models/RNN.h5"
}

# Load models
models = {}
for model_name, path in model_paths.items():
    if os.path.exists(path):
        models[model_name] = load_model(path)
        print(f"✅ Loaded {model_name}")
    else:
        print(f"❌ Model file missing: {path}")

# Load scaler
scaler_path = "Models/scaler.pkl"
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Class labels
class_labels = {
    0: "Normal",
    1: "Limping",
    2: "Slouch",
    3: "No Arm Swing",
    4: "Circumduction",
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return render_template('index.html', error="❌ No video file uploaded.")

    file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # Remove old output video if it exists
    old_video = "static/output.mp4"
    if os.path.exists(old_video):
        os.remove(old_video)
    
    # Set new output video path
    output_video_path = "static/output.mp4"
    
    # Extract keypoints
    keypoints = extract_keypoints(video_path, output_video_path)
    if isinstance(keypoints, dict) and 'error' in keypoints:
        return render_template('index.html', error=keypoints['error'])

    keypoints_np = keypoints.to_numpy()
    print(f"Extracted keypoints shape: {keypoints_np.shape}")

    predictions_summary = {}
    final_votes = Counter()

    for model_name, model in models.items():
        try:
            if scaler:
                keypoints_scaled = scaler.transform(keypoints_np)
            else:
                keypoints_scaled = keypoints_np  # Use unscaled data if no scaler is available

            # print(f"Processing model: {model_name}")
            # print(f"Input shape before reshaping: {keypoints_scaled.shape}")

            # Reshape input for CNN and LSTM models
            if "CNN" in model_name or "LSTM" in model_name:
                keypoints_scaled = keypoints_scaled.reshape((keypoints_scaled.shape[0], 33, 4))
                # print(f"Reshaped input shape: {keypoints_scaled.shape}")

            # Make prediction
            model_output = model.predict(keypoints_scaled)
            # print(f"Raw model output shape: {model_output.shape}")

            # Convert predictions to class labels
            if "CNN" in model_name or "LSTM" in model_name:
                predicted_classes = np.argmax(model_output, axis=1)
                predicted_labels = [class_labels[pred] for pred in predicted_classes]
            else:
                predicted_labels = [class_labels[pred] for pred in model_output]

            # print(f"Predicted labels: {predicted_labels}")

            # Count occurrences of each class
            class_count = dict(Counter(predicted_labels))
            predictions_summary[model_name] = class_count
            final_votes.update(class_count)

        except Exception as e:
            predictions_summary[model_name] = {"Error": f"⚠️ {str(e)}"}
            print(f"Error in model {model_name}: {e}")

    # Get the most common predicted class
    final_prediction = final_votes.most_common(1)[0][0] if final_votes else "Unknown"  # Mode

    return render_template(
        'index.html',
        predictions=predictions_summary,
        final_votes=dict(final_votes),
        final_prediction=final_prediction,
        video_url = url_for('static', filename="output.mp4") + f"?t={int(os.path.getmtime(output_video_path))}"
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host='0.0.0.0', port=port, extra_files=['templates/index.html'])