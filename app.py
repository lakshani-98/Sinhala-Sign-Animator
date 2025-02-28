from flask import Flask, render_template, request, jsonify, send_from_directory
from model import preprocess_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/translate', methods=['POST'])
def translate():
    sinhala_text = request.form['sinhala_text']
    video_name = preprocess_and_predict(sinhala_text)  
    print("video name:: ",video_name)
    return jsonify({'video_name': video_name})  

@app.route('/output/<path:filename>')
def serve_output_file(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    app.run(debug=True)
