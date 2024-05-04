from flask import Flask, send_from_directory, jsonify, request, render_template
import os
import cv2
import numpy as np
from PIL import Image
import networkx as nx
import math
from skimage.measure import approximate_polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

app = Flask(__name__)

# HTML 페이지를 제공하는 라우트
@app.route('/map_touch.html')
def home():
    # 'templates' 폴더 내의 'index.html' 파일을 렌더링하여 반환
    return render_template('map_touch.html')

@app.route('/팀원지도최종본.png')
def serve_image():
    # 'static/img/' 디렉토리에서 '팀원지도최종본.png' 이미지를 클라이언트에게 전송
    return send_from_directory('static/img', '팀원지도최종본.png')

# 좌표를 받아 처리하는 라우트
@app.route('/coordinates', methods=['POST'])
def handle_coordinates():
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    print(f"Received coordinates: x={x}, y={y}")
    # 여기서 좌표에 따라 필요한 처리를 수행할 수 있습니다.
    return jsonify({"status": "success", "x": x, "y": y})
if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.2', port=4080)
    #host를 ipv4 address로 바꿀것
    #http://192.168.0.2:4080/map_touch.html 라고 검색창에 입력하세요