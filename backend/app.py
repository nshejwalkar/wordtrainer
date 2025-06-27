from flask import Flask, request, jsonify
from flask_cors import CORS
from opencv_processing import extract_board
import os

app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)  # enables 

UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER_TEMP = UPLOAD_FOLDER + '/temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_TEMP, exist_ok=True)

# @app.after_request
# def add_headers(response):
#    response.headers['Cache-Control'] = 'no-store'
#    return response

@app.route('/getScreenshot', methods=['POST'])
def get_screenshot():
   if 'video' not in request.files:
      return jsonify({'message': 'No video file provided'}), 400
   
   video = request.files['video']
   video_path = os.path.join(UPLOAD_FOLDER, video.filename)
   video.save(video_path)

   print(f'Video saved at {video_path}')
   print("processing video...")
   board = extract_board(video_path)
   return jsonify({'message': board, 'file': video.filename}), 200

@app.route('/upload', methods=['POST'])
def upload_video():
   if 'video' not in request.files:
      return jsonify({'messsage':'No video file provided'}), 400
   
   video = request.files['video']

   print(request)
   print(request.files)
   print(video)
   
   video_path = os.path.join(UPLOAD_FOLDER, video.filename)
   video.save(video_path)

   print(f'Video saved at {video_path}')
   print("processing video...")
   board = extract_board(video_path)
   return jsonify({'message': board, 'file': video.filename}), 200


if __name__ == '__main__':
   app.run(debug=True)

