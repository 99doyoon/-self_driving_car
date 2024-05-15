import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from flask import request
import requests
import time

SERVER_URL = "http://192.168.137.1:4080/provide_list"

move_command =[]
move_state=0
move_distance=0

#모터 상태
STOP = 0
FORWARD = 1
BACKWARD = 2
ERROR_MOTOR = 20
RIGHTSPEED = 50
LEFTSPEED = RIGHTSPEED
TURN_SPEED = 30
motor_delay = 1
motor_delay_90 = 10

#모터 채널
CH1 = 0
CH2 = 1

#실제 핀 정의 (BCM 모드)
ENA = 26  # PWM 핀 설정
ENB = 0
IN1 = 19
IN2 = 13
IN3 = 6
IN4 = 5

GPIO.setmode(GPIO.BCM)  # BCM 모드 설정
GPIO.setwarnings(False)

#핀 설정 함수
def setPinConfig(EN, INA, INB):
    GPIO.setup(EN, GPIO.OUT)
    GPIO.setup(INA, GPIO.OUT)
    GPIO.setup(INB, GPIO.OUT)
    pwm = GPIO.PWM(EN, 100)  # 100Hz
    pwm.start(0)
    return pwm

#모터 제어 함수
def setMotorControl(pwm, INA, INB, speed, stat):
    pwm.ChangeDutyCycle(speed)
    if stat == FORWARD:
        GPIO.output(INA, GPIO.HIGH)
        GPIO.output(INB, GPIO.LOW)
    elif stat == BACKWARD:
        GPIO.output(INA, GPIO.LOW)
        GPIO.output(INB, GPIO.HIGH)
    elif stat == STOP:
        GPIO.output(INA, GPIO.LOW)
        GPIO.output(INB, GPIO.LOW)

#모터 제어함수 간단하게 사용하기 위해 한번더 래핑(감쌈)
def setMotor(ch, speed, stat):
    if ch == CH1:
        setMotorControl(pwmA, IN1, IN2, speed, stat)
    else:
        setMotorControl(pwmB, IN3, IN4, speed, stat)
        
#LEFTSPEED and Rightspeed change work
def control_motor(command):
    if command == 'f':
        setMotor(CH2, LEFTSPEED, FORWARD)  # 두 바퀴 모두 앞으로
        setMotor(CH1, RIGHTSPEED, FORWARD)

    elif command == 'b':
        setMotor(CH2, LEFTSPEED, BACKWORD)  # 두 바퀴 모두 뒤로
        setMotor(CH1, RIGHTSPEED, BACKWORD)

    elif command == 'r':
        setMotor(CH2, LEFTSPEED, FORWARD)

        setMotor(CH1, RIGHTSPEED-ERROR_MOTOR, FORWARD)

    elif command == 'l':
        setMotor(CH1, RIGHTSPEED, FORWARD)
        setMotor(CH2, LEFTSPEED-ERROR_MOTOR, FORWARD)

    elif command == 's':
        setMotor(CH2, LEFTSPEED, FORWARD)  # 두 바퀴 모두 앞으로
        setMotor(CH1, RIGHTSPEED-TURN_SPEED, FORWARD)

    else:
        setMotor(CH2, 0, STOP)  # 두 바퀴 모두 정지
        setMotor(CH1, 0, STOP)
        

def detect_lanes(frame):
    """
    입력 받은 프레임에서 차선을 감지하고 차량의 위치를 판단하는 함수
    """
    global move_command
    global move_state
    global move_distance

    # 이미지 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3),0)
    height, width = blur.shape[:2]
    lower_third = blur[int(60*height//100):height, 0:width]
    edges = cv2.Canny(image=gray, threshold1=500, threshold2=550, apertureSize=5)

    height, width = frame.shape[:2]
    # ROI를 이미지 하단 50%로 설정

    roi = np.array([[(0, 60*height//100), (0, height), (width, height), (width, 60*height//100)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
       
    # 허프 변환을 통한 직선 검출
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=10, maxLineGap=5)

    left_lines = []
    right_lines = []

    if lines is not None:        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # 기울기가 무한대가 아닌 경우만 처리
                slope = (y2 - y1)  # 기울기 계산
                if abs(slope) > 30:
                    if slope < 0:  # 기울기가 음수일 경우 왼쪽 차선
                        left_lines.append(line)
                    else:  # 기울기가 양수일 경우 오른쪽 차선
                        right_lines.append(line)
                        
    flag_end=0

    vehicle_position = "f"  # 초기 위치를 가운데로 설정
    if left_lines and not right_lines:  # 왼쪽 선만 있을 경우
        vehicle_position = "r"
        if flag_end == 1:
            flag_end=0
            move_state=move_state+1
    elif right_lines and not left_lines:  # 오른쪽 선만 있을 경우
        vehicle_position = "l"
        if flag_end == 1:
            flag_end=0
            move_state=move_state+1
    elif not left_lines and not right_lines:  # 선이 감지되지 않는 경우
        flag_end=1
        
    if move_distance < move_command[move_state][1] and flag_end == 1: 
        vehicle_position = "f"
    elif move_distance > move_command[move_state][1]:
        vehicle_position = move_command[move_state][0]
        
    if vehicle_position == "f":
        move_distance=move_distance+1
        if flag_end == 1:
            flag_end=0
            move_state=move_state+1
        print(move_distance)
    # 둘 다 있을 경우 기본값인 "Center" 유지
                
    return vehicle_position

pwmA = setPinConfig(ENA, IN1, IN2)
pwmB = setPinConfig(ENB, IN3, IN4)

cap = cv2.VideoCapture(0)
position=[]

try:  
    while True:
        response = requests.get(SERVER_URL)
        if response.status_code == 200:
            command_json = response.json()
            if command_json:
                move_command = command_json
                len_row=len(move_command)
                len_col=len(move_command) if len_row > 0 else 0                                
                
            print("받아온 데이터:", command_json)
        else:
            print("404 nogetdata")
        
        if move_command and move_distance != len_row and move_state != len_col:
            ret, frame = cap.read()
            frame = cv2.flip(frame, -1)
            vehicle_position_local=detect_lanes(frame)

            print(vehicle_position_local)
            control_motor(vehicle_position_local)
            
        if move_command and move_distance == len_row and move_state == len_col:
            len_row = 0
            len_col = 0
            move_command =[]
            move_state=0
            move_distance=0
        
except Exception:
    GPIO.cleanup()

cap.release()
cv2.destroyAllWindows()

