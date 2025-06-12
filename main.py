from flask import Flask, request, send_file, jsonify
import os
from PIL import Image
from flask_cors import CORS
from segment_anything import SamPredictor, sam_model_registry
from io import BytesIO
import model
import json

import utils
import sam

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # for receiving reconstruction layers
CORS(app)  # Enable CORS for all routes


MASKS_DIR = 'data/saved_masks'
if not os.path.exists(MASKS_DIR):
    os.makedirs(MASKS_DIR)

@app.route('/get_SAM_embedding', methods=['POST'])
def get_SAM_embedding():
    try:
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400

        image_file = request.files['image']
        if image_file.filename == '':
            return {'error': 'Empty filename'}, 400

        # Read and preprocess image
        embedding_bytes=sam.get_embedding(image_file)

        # Return embedding as binary response
        return send_file(
            BytesIO(embedding_bytes),
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/save_mask', methods=['POST'])
def save_mask():
    try: 
        if 'mask' not in request.files:
            return jsonify({'error': 'No mask provided'}), 400
        
        mask_file = request.files['mask']
        # Generate unique filename (e.g., timestamp-based)
        filename = f"mask_{len(os.listdir(MASKS_DIR)) + 1}.png"
        filepath = os.path.join(MASKS_DIR, filename)
        # Save the mask image
        mask_image = Image.open(mask_file)
        mask_image.save(filepath, 'PNG')
        
        return jsonify({'message': f'Mask saved successfully as {filename}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict', methods=['POST'])
def get_predictions():
    # try:
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400

        image_file = request.files['image']
        if image_file.filename == '':
            return {'error': 'Empty filename'}, 400

        output=model.get_predictions(image_file)

        json_output = json.dumps(output)

        return json_output
    # except Exception as e:
    #     print(str(e))
    #     return {'error': "INTERNAL SERVER ERROR"}, 500

@app.route('/split', methods=['POST'])
def split_image_masks():
    num_horizontal_splits=10
    num_vertical_splits=8

    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400

    if 'num_horizontal_splits'  in request.form:
        num_horizontal_splits=int(request.form['num_horizontal_splits'])
    if 'num_vertical_splits'  in request.form:
        num_vertical_splits=int(request.form['num_vertical_splits'])
    

    response={'segmentation': {}}

    floorplan = request.files['image']
    if floorplan.filename == '':
            return {'error': 'Empty filename'}, 400
    print("Splitting the floorplan")
    response['floorplan'] = utils.split(input=utils.file_to_image(floorplan, to_PIL_Image=True), num_horizontal_splits=num_horizontal_splits, num_vertical_splits=num_vertical_splits)

    print("Form keys received:", list(request.form.keys()))  # Debug log
    for key in request.form:
        if key.startswith('segmentation_'):
            class_name = key[len('segmentation_'):]  # e.g., 'wall' from 'segmentation_wall'
            base64_string = request.form[key]
            if not base64_string:
                continue  # Skip empty strings

            # Decode base64 to image
            try:
                # Handle optional "data:image/png;base64," prefix
                if 'base64,' in base64_string:
                    base64_string = base64_string.split('base64,')[1]
                
                img=utils.base64_to_Image(base64_string)

                print(f'Splitting {key}')
                response['segmentation'][class_name] = utils.split(
                    img,  # Pass PIL Image directly
                    num_horizontal_splits,
                    num_vertical_splits
                )
            except Exception as e:
                print(f"Error processing {class_name}: {str(e)}")  # Debug log
                return {'error': f'Invalid base64 for {class_name}: {str(e)}'}, 400

    if not response['segmentation']:
        print("No valid segmentation masks processed")  # Debug log
        return {'error': 'No valid segmentation masks provided'}, 400

    return response

#implemented in fronted, big payload
@app.route('/reconstruct', methods=['POST'])
def reconstruct_image_masks():
    
    if 'segmentation' not in request.form:
        return {'error': 'No segmentation provided'}, 400

    segmentation_data = request.form['segmentation']
    try:
        segmentation = json.loads(segmentation_data)  # Parse JSON
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid segmentation JSON format'}), 400
    
    print(segmentation)

    return segmentation, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)