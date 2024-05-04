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
    capture_and_save_image(filename='map.png', camera_index=0)
    path_process = find_direction_and_adjust_angle('C:/Users/u/Desktop/팀원지도최종본_출발지만.png')
    find_and_draw_path('map.png', (path_process[0][0],path_process[0][1]), (x,y))
    return jsonify({"status": "success", "x": x, "y": y})

def capture_and_save_image(filename='map.png', camera_index=0):
    # USB 카메라 연결 (1은 첫 번째 연결된 USB 카메라를 의미합니다. 필요에 따라 변경 가능)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    else:
        # 비디오 캡처 속성 설정 (예시)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 카메라에서 사진 한 장을 읽음
    ret, img = cap.read()

    # 사진을 정상적으로 읽었다면 저장
    if ret:
        cv2.imwrite(filename, img)
        print(f"{filename}으로 사진이 저장되었습니다.")
    else:
        print("사진을 찍는 데 실패했습니다.")

    # 카메라 연결 해제
    cap.release()

def calculate_angle_with_horizontal(cx, cy, fx, fy):
    """
    수평선과 (cx, cy)에서 (fx, fy)로 이어지는 선 사이의 각도를 계산합니다.
    """
    dx = fx - cx
    dy = fy - cy
    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def find_direction_and_adjust_angle(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 빨간색의 HSV 범위 정의
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 영역 추출
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    path_process = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # 너무 작은 객체는 무시
            continue

        # 객체의 중심점 계산
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue  # 중심점을 구할 수 없는 경우는 무시

        # 가장 먼 점 찾기
        extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
        extRight = tuple(contour[contour[:, :, 0].argmax()][0])
        extTop = tuple(contour[contour[:, :, 1].argmin()][0])
        extBot = tuple(contour[contour[:, :, 1].argmax()][0])

        distances = [((extLeft[0]-cX)**2 + (extLeft[1]-cY)**2, extLeft),
                     ((extRight[0]-cX)**2 + (extRight[1]-cY)**2, extRight),
                     ((extTop[0]-cX)**2 + (extTop[1]-cY)**2, extTop),
                     ((extBot[0]-cX)**2 + (extBot[1]-cY)**2, extBot)]

        farthest_point = max(distances, key=lambda x: x[0])[1]

        # 중심점과 가장 먼 점 사이의 각도 계산
        angle = calculate_angle_with_horizontal(cX, cY, farthest_point[0], farthest_point[1])

        # 각도 조정. 여기서는 0도 방향이 수평 오른쪽이라 가정합니다.
        # 물방울의 뾰족한 부분이 정확히 오른쪽을 향하게 하려면, 그 각도에서
        adjustment_angle = -angle

        # 결과 저장
        path_process.append((cX, cY, adjustment_angle))

        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        img[(red_mask != 0)] = [255, 255, 255]

        # 이미지 저장
        cv2.imwrite('/content/map_delet_red.png', img)

    # 모든 물방울에 대한 중심점과 조정 각도 정보가 담긴 리스트 반환
    return path_process

def calculate_angle_and_distance(from_point, to_point):
    """두 점 사이의 각도(도)와 거리를 계산합니다."""
    delta_x = to_point[0] - from_point[0]
    delta_y = to_point[1] - from_point[1]
    angle = math.atan2(delta_y, delta_x) * (180 / math.pi)  # 라디안을 도로 변환
    distance = math.sqrt(delta_x**2 + delta_y**2)
    return angle, distance


@app.route('/get_command', methods=['GET'])
def get_command(angle, distance):
    command = angle, distance
    angle, distance = None  # 명령어를 전송한 후에는 초기화
    if command:
        return jsonify({'command': command}), 200
    else:
        return jsonify({'error': 'No command'}), 404

def navigate_path(path):
    """경로를 따라 이동하는 함수입니다."""
    for i in range(len(path) - 1):
        current_point = path[i]
        next_point = path[i + 1]
        
        angle, distance = calculate_angle_and_distance(current_point, next_point)
        
        # 여기서는 예시로 각도와 거리를 출력합니다.
        # 실제로는 이 값을 사용하여 로봇을 회전시키고 직진시키는 명령을 내려야 합니다.
        print(f"회전해야 할 각도: {angle}도, 직진해야 할 거리: {distance}m")
        get_command(angle,distance)
        
        # 예시: rotate(angle)  # 로봇을 angle도 만큼 회전시키는 함수
        # 예시: forward(distance)  # 로봇을 distance만큼 직진시키는 함수

    #최단거리구하기
def find_and_draw_path(image_path, start, end):
    # 이미지 불러오기
    img = Image.open(image_path)

    # 이미지를 흑백으로 변환
    img = img.convert('L')

    # 이미지의 픽셀 값을 가져오기
    pixels = img.load()

    # 임계값 설정
    threshold = 128

    # 이진화 수행
    for i in range(img.width):
        for j in range(img.height):
            if pixels[i, j] < threshold:
                pixels[i, j] = 0  # 통과 불가능한 지역
            else:
                pixels[i, j] = 1  # 통과 가능한 지역

    # 그래프 생성
    G = nx.Graph()

    # 노드 및 간선 추가
    for i in range(img.width):
        for j in range(img.height):
            if pixels[i, j] == 1:  # 통과 가능한 지역만 노드로 추가
                G.add_node((i, j))

    # 4방향 연결 고려
    for node in G.nodes:
        x, y = node
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            node_x, node_y = x + dx, y + dy
            if (node_x, node_y) in G.nodes:  # 연결 가능한 노드만 간선으로 추가
                G.add_edge((x, y), (node_x, node_y))

    # Dijkstra 알고리즘을 이용한 최단 경로 찾기
    path = nx.dijkstra_path(G, (start[1],start[0]), (end[1],end[0]))
    # 원본 이미지 불러오기
    img = Image.open(image_path)

    # 이미지 출력 준비
    fig, ax = plt.subplots()

    # 원본 이미지 출력
    ax.imshow(img, cmap='gray')

    # colormap 지정 (점점 진해지는 파란색 계열)
    cmap = plt.get_cmap('Blues')

    # path에 저장된 노드들을 원으로 그리기
    for i, node in enumerate(path):
        circle = plt.Circle((node[1], node[0]), radius=5, color=cmap(i / len(path)), fill=True)
        ax.add_patch(circle)

    # 그래프 출력
    plt.show()
    
    #가는길 단순화
    # 좌표를 NumPy 배열로 변환
    coords = np.array(path)

    # Douglas-Peucker 알고리즘 적용
    tolerance = 2.0  # 허용 오차 설정
    simplified_coords = approximate_polygon(coords, tolerance)

    # 결과 출력
    print(f"Original number of points: {len(coords)}")
    print(f"Simplified number of points: {len(simplified_coords)}")

    navigate_path(path)
        

if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.6', port=4080)
    #host를 ipv4 address로 바꿀것
