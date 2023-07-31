# 라이브러리 선언
import cv2 as cv
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import datetime
import torch
import mediapipe as mp
from pickle import load
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
import os
import threading
from PIL import ImageFont, ImageDraw, Image
import sys

# 임시 디렉토리 경로에서 데이터 받아오는 클래스
class ResourceHelper:
    def __init__(self):
        pass
    
    def resource_path(self, relative_path):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, relative_path)

# 모델 설정
class Settings:
    def __init__(self, base_dir, save_video_path, yolo_model_path, put_model_path, rotate_model_path,
                 put_scaler_path, rotate_scaler_path, put_seq_len, rotate_seq_len, static_image_mode, 
                 min_detection_confidence, min_tracking_confidence, max_num_hands, yolo_offset, 
                 yolo_conf, put_conf_threshold, rotate_conf_threshold, put_criterion_ratio, rotate_criterion_ratio,
                 input_dim, hidden_dim, layer_dim, output_dim, label_dict):
        
        # 데이터 로드 경로
        self.base_dir = base_dir
        self.save_video_path = save_video_path
        self.yolo_model_path = yolo_model_path
        self.put_model_path = put_model_path
        self.rotate_model_path = rotate_model_path
        self.put_scaler_path = put_scaler_path
        self.rotate_scaler_path = rotate_scaler_path
        
        # 미디어 파이프 설정
        self.max_num_hands = max_num_hands
        self.static_image_mode = static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # 검수 데이터 설정
        self.put_seq_len = put_seq_len
        self.rotate_seq_len = rotate_seq_len
        self.yolo_offset = yolo_offset
        self.yolo_conf = yolo_conf
        self.put_conf_threshold = put_conf_threshold
        self.rotate_conf_threshold = rotate_conf_threshold
        self.put_criterion_ratio = put_criterion_ratio
        self.rotate_criterion_ratio = rotate_criterion_ratio
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.label_dict = label_dict
        
# 스케일러
class Scaler:
    def __init__(self, scaler_path):
        self.scaler = load(open(scaler_path, 'rb'))
    # 스케일링된 데이터를 반환
    def transform(self, data):
        return self.scaler.transform(data)
        
# LSTM 선언
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self): 
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx) 
        gates = gates.squeeze() 
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, -1)

        ingate = F.sigmoid(ingate) 
        forgetgate = F.sigmoid(forgetgate) 
        cellgate = F.tanh(cellgate) 
        outgate = F.sigmoid(outgate) 

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate) 
        hy = torch.mul(outgate, F.tanh(cy)) 
        return(hy, cy)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim 

        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim) # ①
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available(): 
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()) 
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        if torch.cuda.is_available(): 
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()) 
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:] 

        for seq in range(x.size(1)): 
            hn, cn = self.lstm(x[:,seq,:], (hn,cn)) 
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        
        # sigmoid
        out = F.sigmoid(out)
        return out

# YOLO + MediaPipe + LSTM
class DetectedModel:
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        # github 경로에 있는 yolov5 모델 사용
        # self.yolo5 = torch.hub.load('ultralytics/yolov5', 'custom', path=self.settings.yolo_model_path)
        # local 경로에 있는 yolov5 모델 사용
        helper = ResourceHelper()  # ResourceHelper 클래스의 인스턴스 생성
        get_yolo5_path = helper.resource_path('ultralytics_yolov5_master')  # resource_path() 메서드 호출
        self.yolo5 = torch.hub.load(get_yolo5_path, 'custom', path = self.settings.yolo_model_path, source='local') # pt파일 경로 지정
        self.yolo5.amp = True
        self.yolo5.eval()
        self.yolo5.conf = self.settings.yolo_conf
        self.mp_hand = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hand = self.mp_hand.Hands(
            max_num_hands=self.settings.max_num_hands,
            static_image_mode=self.settings.static_image_mode,
            min_detection_confidence=self.settings.min_detection_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence
        )
        self.put_model = None
        self.rotate_model = None

    # lstm 모델 로드
    def lstm_load_model(self, weight_path):
        model = LSTMModel(self.settings.input_dim, self.settings.hidden_dim, self.settings.layer_dim, self.settings.output_dim)
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        return model

    # pyqt_process.py 파일에서 모델 정보 가져오는 함수
    def get_model(self):
        self.put_model = self.lstm_load_model(self.settings.put_model_path)
        self.rotate_model = self.lstm_load_model(self.settings.rotate_model_path)
        return self.yolo5, self.hand, self.mp_hand, self.mp_drawing, self.mp_styles, self.put_model, self.rotate_model
    
# UI 세팅
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        # # 현재 파일 경로로 이동
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        # os.chdir(self.current_directory)
        
        # UI 불러오기
        self.helper = ResourceHelper()  # ResourceHelper 클래스의 인스턴스 생성
        get_ui_path = self.helper.resource_path("pyqt_qt_v2.ui")  # resource_path() 메서드 호출
        loadUi(get_ui_path, self)
        
        # 비디오 출력창 설정
        self.video_frame = self.findChild(QtWidgets.QLabel, 'label_frame')
        self.error_list = ["미검사", "미검사", "미검사", "미검사", "미검사", "미검사", "미검사"]

        # 기본값 설정
        self.width = self.video_frame.size().width()    # 960
        self.height = self.video_frame.size().height()  # 540
        self.img_width = int(self.width * 0.2)
        self.img_height = int(self.height * 0.2)
        
        self.settings = None
        self.video_device = 0
        self.video_test_flag = False
        self.video_name = "미설정"
        
        self.video_path = "None"
        self.outputPath = "save_video"
        self.modelPath = self.helper.resource_path("0627002.onnx")
        self.put_model_path = self.helper.resource_path('put_lstm_weight_seq_10_.pth') #put모델 웨이트
        self.put_scaler_path = self.helper.resource_path('put_lstm_ss.pkl') #put모델 스케일러
        self.rotate_model_path = self.helper.resource_path('rotate_lstm_weight_seq_10_.pth') #rotate모델 웨이트
        self.rotate_scaler_path = self.helper.resource_path('rotate_lstm_ss.pkl') #rotate스케일러
        self.yolo5 = None
        self.hand_model = None
        self.put_model = None
        self.rotate_model = None
        self.put_scaler = None
        self.rotate_scaler = None
        
        # 작업공간과 rotate 비율 설정
        self.box_x_ul = int(self.width * 0.2)
        self.box_y_ul = int(self.height * 0.2)
        self.box_x_lr = int(self.width * 0.8)
        self.box_y_lr = int(self.height * 0.9)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
            if torch.backends.mps.is_available():
                self.device = 'mps'
                
        self.box_distance = int(self.width * 0.20)
        
        self.height_10_persent = int(self.height * 0.1)
        self.height_20_persent = int(self.height * 0.2)
        self.height_30_persent = int(self.height * 0.3)
        self.height_50_persent = int(self.height * 0.5)
        self.height_60_persent = int(self.height * 0.6)
        self.height_70_persent = int(self.height * 0.7)
        self.height_90_persent = int(self.height * 0.9)
        
        self.width_10_persent = int(self.width * 0.1)
        self.width_30_persent = int(self.width * 0.3)
        self.width_40_persent = int(self.width * 0.4)
        self.width_60_persent = int(self.width * 0.6)
        self.width_70_persent = int(self.width * 0.7)
        self.width_90_persent = int(self.width * 0.9)
        
        
        # yolo_class_idx
        self.BACK_IDX = 0
        self.BOTTOM_IDX = 1
        self.ERROR_IDX = 2
        self.FRONT_IDX = 3
        self.HAND_IDX = 4
        self.HEAD_IDX = 5
        self.SIDE_IDX = 6
        
        # put_rotate_idx
        self.PUT_IDX = 0
        self.ROTATE_IDX = 0
        
        # step_check
        self.none_to_put = []
        self.front_to_side = []
        self.side_to_back = []
        self.back_to_side = []
        self.side_to_front = []
        
        # step_data
        self.STEP = 1
        self.ROTATE_STEP = 1
        self.SUB_STEP = 1
        self.WAIT_COUNT = 0
        
        # Error_check
        self.error_list = ["미검사", "미검사", "미검사", "미검사", "미검사", "미검사", "미검사"]
        self.error_flag = False        
        
        # 스텝별 검사 화면
        self.FRONT_IMG = np.zeros((self.box_x_lr-self.box_x_ul, self.box_y_lr-self.box_y_ul, 3), np.uint8)
        self.SIDE1_IMG = np.zeros((self.box_x_lr-self.box_x_ul, self.box_y_lr-self.box_y_ul, 3), np.uint8)
        self.BACK_IMG = np.zeros((self.box_x_lr-self.box_x_ul, self.box_y_lr-self.box_y_ul, 3), np.uint8)
        self.SIDE2_IMG = np.zeros((self.box_x_lr-self.box_x_ul, self.box_y_lr-self.box_y_ul, 3), np.uint8)
        self.HEAD_IMG = np.zeros((self.box_x_lr-self.box_x_ul, self.box_y_lr-self.box_y_ul, 3), np.uint8)
        self.BOTTOM_IMG = np.zeros((self.box_x_lr-self.box_x_ul, self.box_y_lr-self.box_y_ul, 3), np.uint8)
        
        # # 테스트 삭제용
        # self.FRONT_IMG = cv.resize(self.FRONT_IMG, (self.img_width, self.img_height))
        # self.SIDE1_IMG = cv.resize(self.SIDE1_IMG, (self.img_width, self.img_height))
        # self.BACK_IMG = cv.resize(self.BACK_IMG, (self.img_width, self.img_height))
        # self.SIDE2_IMG = cv.resize(self.SIDE2_IMG, (self.img_width, self.img_height))
        # self.HEAD_IMG = cv.resize(self.HEAD_IMG, (self.img_width, self.img_height))
        # self.BOTTOM_IMG = cv.resize(self.BOTTOM_IMG, (self.img_width, self.img_height))
        
        # 타이머
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.thread_ = None
        self.thread_model_load = True
        
        # 버튼 동작
        self.pushButton_start.clicked.connect(self.start_video) # 시작버튼
        self.pushButton_end.clicked.connect(self.end_video)     # 종료버튼
        self.pushButton_videoTest.clicked.connect(self.test_video) # 비디오테스트
        
        # 결과 출력 버튼
        self.pushButton_resultDisplay.clicked.connect(self.display_result)
        
        # 검수모델 설정
        self.toolButton_editVideoPath.clicked.connect(self.set_videoPath)   # 비디오 경로 설정
        self.toolButton_editOutputPath.clicked.connect(self.set_outputPath) # 검수영상 저장 경로 설정
        self.toolButton_editModelPath.clicked.connect(self.set_modelPath)   # 객체인식 가중치 경로 설정
        self.toolButton_editPutModelPath.clicked.connect(self.set_putModelPath) # PUT 인식 가중치 경로 설정
        self.toolButton_editPutScalerPath.clicked.connect(self.set_putScalerPath) # PUT 스케일러 경로 설정
        self.toolButton_editRotateModelPath.clicked.connect(self.set_rotateModelPath) # ROTATE 인식 가중치 경로 설정
        self.toolButton_editRotateScalerPath.clicked.connect(self.set_rotateScalerPath) # ROTATE 스케일러 경로 설정
    
    # 검수 영상 저장 디렉토리 생성
    def make_save_video_dir(self):
        if self.outputPath == "save_video":
            save_video_directory = os.path.join(self.current_directory, "save_video")
            if not os.path.exists(save_video_directory):
                os.makedirs(save_video_directory)
    
    # 값 설정
    def data_settings(self):
        self.settings = Settings(
            base_dir=self.video_path,
            save_video_path = self.outputPath,
            yolo_model_path = self.modelPath,
            put_model_path = self.put_model_path,
            rotate_model_path = self.rotate_model_path,
            put_scaler_path = self.put_scaler_path,
            rotate_scaler_path = self.rotate_scaler_path,
            put_seq_len = int(self.put_model_path.split("_")[-2]), # put 모델 프레임 개수 10개
            rotate_seq_len = int(self.rotate_model_path.split("_")[-2]), # rotate 모델 프레임 개수 10개
            max_num_hands = 1,
            static_image_mode = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5,
            yolo_offset = 75,
            yolo_conf = 0.5,
            put_conf_threshold = 0.35,
            rotate_conf_threshold = 0.35,
            put_criterion_ratio = 0.5,
            rotate_criterion_ratio = 0.2,
            input_dim = 12,
            hidden_dim = 128,
            layer_dim = 1,
            output_dim = 2,
            label_dict = {
                0 : 'back',
                1 : 'bottom',
                2 : 'error',
                3 : 'front',
                4 : 'hand',
                5 : 'head',
                6 : 'side'}
        )
    
    # 영상 확인
    def test_video(self):
        self.video_test_flag = True
        self.video_frame.setText("0. Loading...")
        if self.radioButton_camera.isChecked(): # 실시간 캠 사용
            self.video_device = self.spinBox_camera.value()
            self.cap = cv.VideoCapture(self.video_device)            
            self.timer.start(30)
            self.video_name = "실시간 영상 테스트"
        else: # 비디오 사용
            self.video_device = self.label_videoPath.text()
            if (not self.video_device) or (not any(ext in self.video_device for ext in ['.mp4', '.avi', '.mkv', '.mov', '.MOV'])):
                self.video_frame.clear()
                self.label_directive.setText("비디오 경로가 잘못되었습니다")
            else:
                self.cap = cv.VideoCapture(self.video_device)
                self.timer.start(30)
                self.video_name = self.video_device.split("/")[-1]
    
    # 영상 시작
    def start_video(self):
        self.label_directive.setText("검수모델을 불러오고 있습니다.. 잠시만 기다려주세요...")
        self.make_save_video_dir()
        self.timer.stop()
        self.video_test_flag = False
        self.data_settings()
        if self.thread_model_load:
            self.load_model_thread()
        else:
            # 값 초기화
            
            # 검수 list 초기화
            self.none_to_put = []
            self.front_to_side = []
            self.side_to_back = []
            self.back_to_side = []
            self.side_to_front = []
            
            # 단계 초기화
            self.STEP = 1
            self.ROTATE_STEP = 1
            self.SUB_STEP = 1
            self.WAIT_COUNT = 0
            self.text = "Loading..."
            self.progressBar_stepProgress.setValue(0)
            
            # 실시간 캠 사용
            if self.radioButton_camera.isChecked():
                self.video_device = self.spinBox_camera.value()
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
                self.cap = cv.VideoCapture(self.video_device)
                self.video_name = f"{formatted_time}.mp4"
                self.inspection_process()
                
            # 비디오 사용
            else:
                self.video_device = self.label_videoPath.text()
                if (not self.video_device) or (not any(ext in self.video_device for ext in ['.mp4', '.avi', '.mkv', '.mov', '.MOV'])):
                    self.video_frame.clear()
                    self.label_directive.setText("비디오 경로가 잘못되었습니다")
                else:
                    self.cap = cv.VideoCapture(self.video_device)
                    self.video_name = self.video_device.split("/")[-1]
                    self.inspection_process()

    # 영상 종료
    def end_video(self):
        self.video_test_flag = False
        self.video_frame.clear()
        if self.cap is not None:
            ret, _ = self.cap.read()
            if ret:
                self.timer.stop()
                self.cap.release()
            else:
                self.video_frame.setText("영상이 존재하지 않습니다")
        else:
            self.video_frame.setText("영상이 존재하지 않습니다")
            
    # 프레임 갱신
    def update_frame(self):
        if self.video_test_flag:
            self.ret, self.frame = self.cap.read()
            self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        else:
            self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
            self.progressBar_stepProgress.setValue(self.STEP * 20 + 5 * (self.ROTATE_STEP-1))
            self.update_directive()
        h, w, ch = self.frame.shape
        q_image = QImage(self.frame.data, w, h, ch * w, QImage.Format_RGB888).scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)            
        self.video_frame.setPixmap(QPixmap.fromImage(q_image))
    
    # 지시문
    def update_directive(self):
        self.label_directive.setText(self.text)
    
    # 단일 비디오 경로
    def set_videoPath(self):
        editVideoPath = QFileDialog()
        editVideoPath.setNameFilter("Video file (*.mp4 *.avi *.mkv *.mov *.MOV)")
        self.video_path, _ = editVideoPath.getOpenFileName(self, "Select Video File", "", "Video file (*.mp4 *.avi *.mkv *.mov *.MOV)")
        if len(self.video_path)==0:  # default 값 설정
            self.video_path = "None"
        self.label_videoPath.setText(self.video_path)

    # 검수영상 저장 경로
    def set_outputPath(self):
        editOutputPath = QFileDialog()
        editOutputPath.setFileMode(QFileDialog.Directory)        
        self.outputPath = editOutputPath.getExistingDirectory(self, "Select Directory")
        if len(self.outputPath)==0: # default 값 설정
            self.outputPath = "save_video"
        self.label_outputPath.setText(self.outputPath)
    
    # 객체인식 가중치 경로
    def set_modelPath(self):
        editModelPath = QFileDialog()
        editModelPath.setNameFilter("Model weigth (*.pt *.onnx)")
        self.modelPath, _ = editModelPath.getOpenFileName(self, "Select Model weigth", "", "Model weigth (*.pt *.onnx)")
        if len(self.modelPath)==0:  # default 값 설정
            self.modelPath = self.helper.resource_path("0627002.onnx")
        self.label_modelPath.setText(self.modelPath)
        
    # PUT 가중치 경로
    def set_putModelPath(self):
        editPutModelPath = QFileDialog()
        editPutModelPath.setNameFilter("Model weigth (*.pth)")
        self.putModelPath, _ = editPutModelPath.getOpenFileName(self, "Select Model weigth", "", "Model weigth (*.pth)")
        if len(self.putModelPath)==0:  # default 값 설정
            self.putModelPath = self.helper.resource_path("put_lstm_weight_seq_10_.pth")
        self.label_putModelPath.setText(self.putModelPath)
        
    # PUT 스케일러 경로
    def set_putScalerPath(self):
        editPutScalerPath = QFileDialog()
        editPutScalerPath.setNameFilter("Scaler file (*.pkl)")
        self.putScalerPath, _ = editPutScalerPath.getOpenFileName(self, "Select Scaler file", "", "Scaler file (*.pkl)")
        if len(self.putScalerPath)==0:  # default 값 설정
            self.putScalerPath = self.helper.resource_path("put_lstm_ss.pkl")
        self.label_putScalerPath.setText(self.putScalerPath)
    
    # ROTATE 인식 가중치 경로
    def set_rotateModelPath(self):
        editRotateModelPath = QFileDialog()
        editRotateModelPath.setNameFilter("Model weigth (*.pth)")
        self.rotateModelPath, _ = editRotateModelPath.getOpenFileName(self, "Select Model weigth", "", "Model weigth (*.pth)")
        if len(self.rotateModelPath)==0:  # default 값 설정
            self.rotateModelPath = self.helper.resource_path("rotate_lstm_weight_seq_10_.pth")
        self.label_rotateModelPath.setText(self.rotateModelPath)
        
    # ROTATE 스케일러 경로
    def set_rotateScalerPath(self):
        editRotateScalerPath = QFileDialog()
        editRotateScalerPath.setNameFilter("Scaler file (*.pkl)")
        self.rotateScalerPath, _ = editRotateScalerPath.getOpenFileName(self, "Select Scaler file", "", "Scaler file (*.pkl)")
        if len(self.rotateScalerPath)==0:  # default 값 설정
            self.rotateScalerPath = self.helper.resource_path("rotate_lstm_ss.pkl")
        self.label_rotateScalerPath.setText(self.rotateScalerPath)
        
    # 결과 출력
    def display_result(self):
        result_text = \
f"""
<pre><h1>검사결과</h1>
<h2>{self.video_name}</h2>
준비      {self.error_list[0]}
옆면(1)   {self.error_list[1]}
뒷면      {self.error_list[2]}
옆면(2)   {self.error_list[3]}
정면      {self.error_list[4]}
윗면      {self.error_list[5]}
아랫면    {self.error_list[6]}

영상은 "검수영상 저장경로"에 저장되었습니다
검수를 지속하시려면 영상 지정 후 "실행"을 누르세요
종료를 원하시면 우측 상단의 "닫기" 버튼을 누르세요</pre>
"""
        self.video_frame.setText(f"{result_text}")
    
    # detect_model.py에서 모델 정보 가져오기
    def load_model(self):
        detectModel = DetectedModel(self.settings)
        self.yolo5, self.hand, self.mp_hand, self.mp_drawing, self.mp_styles, self.put_model, self.rotate_model = detectModel.get_model()
        self.yolo5.to(self.device)
        self.pred = 0
        self.hand_frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.res = self.hand.process(cv.cvtColor(self.hand_frame, cv.COLOR_BGR2RGB))
        self.put_preds = 0
        self.rotate_preds = 0
        self.hand_class_text = 'None'
        self.put_scaler = Scaler(self.settings.put_scaler_path)
        self.rotate_scaler = Scaler(self.settings.rotate_scaler_path)
        self.thread_model_load = False
        self.label_directive.setText("0. 검수모델을 성공적으로 불러왔습니다")
        
    # 작업공간 빨간 상자로 그리기
    def draw_workspace(self):
        cv.rectangle(self.frame,
                     (self.box_x_ul, self.box_y_ul),
                     (self.box_x_lr, self.box_y_lr),
                     (0, 0, 255),
                     5)
    
    # STEP 1. 대상배치 및 정면
    def place_object(self, step_list, next_class_idx):
        # 데이터 저장
        if (np.isnan(self.put_preds) == False):
            step_list.append(self.put_preds)
            if (len(step_list) > self.settings.put_seq_len):
                del step_list[0]

        if self.pred.shape[0] != 0:
            self.text = f'{self.STEP}. 작업공간에 \"{self.settings.label_dict[next_class_idx]}\"으로 대상을 배치하세요.'
            if (len(step_list) >= 3) and (self.HAND_IDX not in self.pred[:, 5]):
                put_ratio = step_list.count(self.PUT_IDX) / len(step_list)
                if put_ratio > self.settings.put_criterion_ratio:
                    for i in range(self.pred.shape[0]):
                        if self.pred[i, 5] == next_class_idx:
                            x_ul, y_ul, x_lr, y_lr = list(map(int, (self.pred[i, :4])))
                            if (x_ul > self.box_x_ul) & (y_ul > self.box_y_ul) & (x_lr < self.box_x_lr) & (y_lr < self.box_y_lr): 
                                self.WAIT_COUNT += 1    # wait 단계로 이동
                                break    
    
    # STEP wait. error 확인
    def wait_sequence(self, next_class_idx, step_num):
        if self.pred.shape[0] != 0:
            if next_class_idx not in self.pred[:, 5]: # 검수하고자 하는 물품의 방향이 확인되지 않으면
                self.text = f"{step_num}. \"{self.settings.label_dict[next_class_idx]}\" 상태의 대상이 확인되지 않습니다 "
            else: # 검수하고자 하는 물품의 방향이 확인되면서
                if self.HAND_IDX not in self.pred[:, 5]: # 손이 검출되지 않은 상태이고
                    for i in range(self.pred.shape[0]):
                        if (self.pred[i, 5] == next_class_idx): # 검수하고자 하는 물품의 방향이 확인되면
                            x_ul, y_ul, x_lr, y_lr = list(map(int, (self.pred[i, :4])))
                            if (x_ul > self.box_x_ul) & (y_ul > self.box_y_ul) & (x_lr < self.box_x_lr) & (y_lr < self.box_y_lr):
                                self.WAIT_COUNT += 9 # wating_count에 3을 더함 (100이 될때까지)
                                self.text = f'{step_num}. 대상을 확인하고 있습니다. [{self.WAIT_COUNT%101}%]'
                                if self.ERROR_IDX in self.pred[:, 5]: # 오류가 보이면 check
                                    self.error_flag = True
                else: # 손이 검출되는 상황이라면
                    self.text = f"{step_num}. 손을 화면 밖으로 치워주세요"

    # STEP 2. 측면
    def rotate_step_by_step(self, step_list, pre_class_idx, next_class_idx):
        if self.SUB_STEP == 1: # 전단계의 방향을 보여주는 단계
            self.text = f'2-{self.ROTATE_STEP}. \"{self.settings.label_dict[pre_class_idx]}\"을 보여주세요'
            if self.HAND_IDX not in self.pred[:, 5]: # 손이 검출되지 않은 상태이고
                if pre_class_idx in self.pred[:, 5]: # 전 단계의 물품 방향이 검출되면
                    self.SUB_STEP += 1 # 다음 Substep으로 진행
                    
        elif self.SUB_STEP==2: # 다음 방향으로 회전하는 단계
            self.text = f'2-{self.ROTATE_STEP}. 대상을 돌려서 \"{self.settings.label_dict[next_class_idx]}\"을 보여주세요'
            if (np.isnan(self.rotate_preds) == False):
                step_list.append(self.rotate_preds) # LSTM을 통해 분류한 행동값을 step_list에 삽입
            if (len(step_list) >= 3) and (next_class_idx in self.pred[:, 5]) and (self.HAND_IDX not in self.pred[:, 5]): 
                # 다음 방향이 검출되고, 손이 검출되지 않을 때
                rotate_ratio = step_list.count(self.ROTATE_IDX) / len(step_list)
                step_list = []  # 리스트 초기화
                if rotate_ratio >= self.settings.rotate_criterion_ratio:
                    # 그동안 step_list에 쌓아두었던 행동값들 중에 Rotate의 비율이 일정 % 이상이면
                    self.WAIT_COUNT += 1 # wait step으로 이동
                    self.SUB_STEP = 1
                else:
                    self.SUB_STEP = 1
                        
        else: # 다음 방향이 모두 확인되어 sub_step이 3의 값을 가지면
            self.SUB_STEP = 1 # substep은 1로 초기화

    # STEP 3,4. head, bottom 확인
    def show_head_bottom(self, next_class_idx):
        self.text = f'{self.STEP}. \"{self.settings.label_dict[next_class_idx]}\"을 보여주세요'
        if next_class_idx in self.pred[:, 5]: # 다음 방향이 검출되면
            self.WAIT_COUNT += 1 # wait step으로 이동

    # STEP 3,4의 wait. error 확인
    def wait_sequence2(self, next_class_idx):
        if next_class_idx in self.pred[:, 5]: # 다음 방향이 검출되고
            for i in range(self.pred.shape[0]):
                if (self.pred[i, 5] == next_class_idx):
                    x_ul, y_ul, x_lr, y_lr = list(map(int, (self.pred[i, :4])))
                    # 해당 방향이 검출 영역 내에 위치하면
                    if (x_ul > self.box_x_ul) & (y_ul > self.box_y_ul) & (x_lr < self.box_x_lr) & (y_lr < self.box_y_lr):
                        self.WAIT_COUNT += 9 # wating_count 더하기
                        self.text = f'{self.STEP}. 대상을 확인하고 있습니다. [{self.WAIT_COUNT%101}%]'
                        if self.ERROR_IDX in self.pred[:, 5]: # 오류가 보이면 check
                            self.error_flag = True
                        break
                    # 해당 방향이 검출 영역 내에 위치하지 않으면
                    else:
                        self.text = f'{self.STEP}. 대상 위치를 재조정 해주세요'

        else: # 다음 방향이 검출되지 않으면 대기시간 초기화
            self.text = f"{self.STEP}. \"{self.settings.label_dict[next_class_idx]}\"을 찾을 수 없습니다"
            if self.STEP == 3:
                self.WAIT_COUNT = 500
            elif self.STEP == 4:
                self.WAIT_COUNT = 600
            else:
                self.text = f"{self.STEP}. Program Error!"
    
    # wait 단계에서 오류 확인
    def error_check(self, seq_num):
        if self.error_flag:
            self.error_list[seq_num] = "Fail"
        else:
            self.error_list[seq_num] = "Pass"
    
    # 모델 불러오기
    def load_model_thread(self):
        self.thread_ = threading.Thread(target=self.load_model)
        self.thread_.start()
    
    # CV에서 한글 출력 함수
    def korean_text(self, text='',font_size=30, font_color=(255,51,255), font_rocation=(50,50), bold=False): 
        if bold:
            fontpath =  self.helper.resource_path('KoPubWorld Dotum Bold.ttf') #폰트가 저장된 경로 지정
        else:
            fontpath =  self.helper.resource_path('KoPubWorld Dotum Light.ttf') #폰트가 저장된 경로 지정
        font = ImageFont.truetype(fontpath, font_size)
        img_pil = Image.fromarray(self.frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text(font_rocation,  text, font=font, fill=font_color) #(B,G,R)
        self.frame = np.array(img_pil)
    
    # 상단 step 박스 변경 함수
    def step_box_draw(self): #480, 960, 1440, 1920
        cv.rectangle(self.frame, (0, 0), (self.width, self.height_10_persent), (0,0,0), -1) #최상단 검은 박스
        text_offset = 0.04
        step_box_type = []
        step_font_color = []
        bold_type = []
        for n in range(5):
            if n == (self.STEP-1):
                step_box_type.append(-1)
                bold_type.append(True)
                step_font_color.append((0,0,0))
            else:
                step_font_color.append((255,255,255))
                step_box_type.append(1)
                bold_type.append(False)
        
        cv.rectangle(self.frame,                     (1, 0),     (self.box_distance, self.height_10_persent), (255,255,255), step_box_type[0]) # Ready
        cv.rectangle(self.frame, (self.box_distance * 1, 0), (self.box_distance * 2, self.height_10_persent), (255,255,255), step_box_type[1]) # Rotate
        cv.rectangle(self.frame, (self.box_distance * 2, 0), (self.box_distance * 3, self.height_10_persent), (255,255,255), step_box_type[2]) # Head
        cv.rectangle(self.frame, (self.box_distance * 3, 0), (self.box_distance * 4, self.height_10_persent), (255,255,255), step_box_type[3]) # Bottom
        cv.rectangle(self.frame, (self.box_distance * 4, 0), (self.box_distance * 5, self.height_10_persent), (255,255,255), step_box_type[4]) # Finish
        self.korean_text(text='Ready', font_size=30, font_color=step_font_color[0], font_rocation=(int(self.width * (0.1-text_offset)),0), bold=bold_type[0])   
        self.korean_text(text='Rotate',font_size=30, font_color=step_font_color[1], font_rocation=(int(self.width * (0.3-text_offset)),0), bold=bold_type[1])  
        self.korean_text(text='Head',  font_size=30, font_color=step_font_color[2], font_rocation=(int(self.width * (0.5-text_offset)),0), bold=bold_type[2])  
        self.korean_text(text='Bottom',font_size=30, font_color=step_font_color[3], font_rocation=(int(self.width * (0.7-text_offset)),0), bold=bold_type[3])
        self.korean_text(text='Finish',font_size=30, font_color=step_font_color[4], font_rocation=(int(self.width * (0.9-text_offset)),0), bold=bold_type[4]) 
    
    
    # 검수시작
    def inspection_process(self):
        # 검수영상 저장 설정
        fps = self.cap.get(cv.CAP_PROP_FPS)
        codec = 'mp4v' if self.video_name.split('.')[1] in ['mp4', 'avi', 'mkv', 'mov', 'MOV'] else 'MP4V' #코덱
        fourcc = cv.VideoWriter_fourcc(*codec) # 코덱 적용
        self.out = cv.VideoWriter(self.settings.save_video_path+"/"+self.video_name, fourcc, fps * 0.5, (self.width, self.height)) # 비디오 저장
        
        if not self.thread_model_load:
            self.load_model()   # 검수모델 가져오기
        yolo_offset = self.settings.yolo_offset
        frame_num = 1
        
        # put 모델 미디어파이프 관절 값
        put_landmark = [0, 1, 5, 9, 13, 17]
        
        put_info = []   # put
        rotate_info = [] # rotate
        
        self.text = '1. 작업공간에 \"front\" 상태의 대상을 배치하세요.'
        
        # 검수 시작
        while True:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                break
            
            self.frame = cv.resize(self.frame, (self.width, self.height))   # 영상 크기를 UI 크기로 조정
            
            self.pred = self.yolo5(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)).pred[0].numpy()   # 객체인식
            self.put_preds = np.nan
            self.rotate_preds = np.nan
            self.hand_class_text = 'None'
            
            # 객체인식에 성공하면 검수 시작
            if self.pred.shape[0] != 0:
                
                # 인식에 성공했으나 손이 없는 경우
                if self.HAND_IDX not in self.pred[:, 5]:
                    tmp = [np.nan for i in range(12)]
                    rotate_info.append([self.video_name, frame_num] + tmp)
                    put_info.append([self.video_name, frame_num] + tmp)
                    
                for i in range(self.pred.shape[0]):
                    
                    x_ul, y_ul, x_lr, y_lr, conf, class_ = list(map(int, (self.pred[i, :4]))) + list(self.pred[i, 4:])

                    # 인식 값 중 손이 있는 경우
                    if class_ == self.HAND_IDX:
                        
                        self.hand_frame = np.zeros((self.frame.shape[0], self.frame.shape[1], 3), np.uint8)
                        
                        # 손 bbox 에 offset 적용
                        hand_x_ul = x_ul if x_ul - yolo_offset <= 0 else x_ul - yolo_offset
                        hand_x_lr = x_lr + yolo_offset if x_lr + yolo_offset <= self.frame.shape[1] else x_lr
                        hand_y_ul = y_ul if y_ul - yolo_offset <= 0 else y_ul - yolo_offset
                        hand_y_lr = y_lr
                        self.hand_frame[hand_y_ul:hand_y_lr, hand_x_ul:hand_x_lr] = self.frame[hand_y_ul:hand_y_lr, hand_x_ul:hand_x_lr]
                        
                        # rotate step에서만 media-pipe 작동
                        if ((self.STEP == 2) or (self.STEP == 1)) and True:
                            # 손 검출 수행
                            self.res = self.hand.process(cv.cvtColor(self.hand_frame, cv.COLOR_BGR2RGB))

                            # 손 랜드마크 좌표값 가져오기
                            if self.res.multi_hand_landmarks:

                                # for landmarks in self.res.multi_hand_landmarks:
                                    landmarks = self.res.multi_hand_landmarks[0]

                                    joint = np.zeros((21, 3))
                                    self.mp_drawing.draw_landmarks(self.frame,
                                                            landmarks,
                                                            self.mp_hand.HAND_CONNECTIONS,
                                                            self.mp_styles.get_default_hand_landmarks_style(),
                                                            self.mp_styles.get_default_hand_connections_style())
                                    
                                    # 랜드마크 포인트 좌표값 채우기
                                    put_xy = [] # put인식용 리스트
                                    for j, lm in enumerate(landmarks.landmark):
                                        joint[j] = [lm.x, lm.y, lm.z]
                                        if j in put_landmark:
                                            put_xy.extend([lm.x, lm.y])
                                            

                                    # 랜드마크 좌표값 벡터 내적 계산을 이용한 각도 계산 (엄지, 손바닥) | 10개 벡터 => 12개의 각도
                                    v1 = joint[[0, 1, 0, 5, 0, 0 , 0 , 5, 9 , 13]]
                                    v2 = joint[[1, 2, 5, 6, 9, 13, 17, 9, 13, 17]]
                                    v = v2 - v1

                                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                                    angle = np.arccos(np.einsum('nt,nt->n',
                                                                v[[1, 0, 2, 4, 5, 7, 3, 4, 8, 5, 9, 6],:], 
                                                                v[[0, 2, 4, 5, 6, 2, 7, 7, 4, 8, 5, 9],:]))

                                    # radian to degree
                                    angle = np.degrees(angle)

                                    rotate_info.append([self.video_name, frame_num] + list(angle))
                                    put_info.append([self.video_name, frame_num] + put_xy)

                            # 손 좌표값이 없을 경우 NaN 처리
                            else:
                                tmp = [np.nan for i in range(12)]
                                rotate_info.append([self.video_name, frame_num] + tmp)
                                put_info.append([self.video_name, frame_num] + tmp)

                            # STEP 1 에서만 실행
                            if self.STEP == 1:
                                # put 모델
                                if len(put_info) >= self.settings.put_seq_len:
                                    lms = [ tmp[2:] if np.isnan(tmp[2:][0]) ==False else [ 0 for n in range(12)] for tmp in put_info[-self.settings.put_seq_len:] ]
                                    lms = np.array(lms)
                                    scaled_lms = self.put_scaler.transform(lms)
                                    lms_tensor = torch.from_numpy(np.expand_dims(scaled_lms, axis=0).astype('float32'))
                                    
                                    self.put_model.eval()
                                    lms_input = lms_tensor.view(-1, self.settings.put_seq_len, 12).to(self.device)
                                    lms_output = self.put_model(lms_input).to(self.device)
                                    
                                    del put_info[0]
                                    if lms_output[self.PUT_IDX] > self.settings.put_conf_threshold:
                                        self.put_preds = 0
                                        self.hand_class_text = 'Put'
                                    else:
                                        self.put_preds = 1
                                        self.hand_class_text = 'Stop'


                            # STEP 2 에서만 실행
                            elif self.STEP == 2:
                                # rotate 모델
                                if len(rotate_info) >= self.settings.rotate_seq_len:
                                    angles = [ tmp[2:] if np.isnan(tmp[2:][0]) == False else [ 0 for n in range(12)] for tmp in rotate_info[-self.settings.rotate_seq_len:] ]
                                    angles = np.array(angles)
                                    scaled_angles = self.rotate_scaler.transform(angles)
                                    angles_tensor = torch.from_numpy(np.expand_dims(scaled_angles, axis=0).astype('float32'))

                                    self.rotate_model.eval()
                                    angles_input = angles_tensor.view(-1, self.settings.rotate_seq_len, 12).to(self.device)
                                    angles_output = self.rotate_model(angles_input).to(self.device)
                                    
                                    del rotate_info[0] # 메모리 관리
                                    if angles_output[self.ROTATE_IDX] > self.settings.rotate_conf_threshold:
                                        self.rotate_preds = 0
                                        self.hand_class_text = 'Rotate'
                                    else:
                                        self.rotate_preds = 1
                                        self.hand_class_text = 'Other'

                    # 바운딩 박스 그리기
                    cv.rectangle(self.frame, (x_ul, y_ul), (x_lr, y_lr), (100, 100, 100), 5)
                    # Class 표시
                    cv.putText(self.frame, self.settings.label_dict[int(class_)], (x_ul, y_ul-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    # 신뢰도 표시
                    cv.putText(self.frame, str(round(conf, 2)), (x_ul + 100, y_ul-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                tmp = [np.nan for i in range(12)]
                rotate_info.append([self.video_name, frame_num] + tmp)
                put_info.append([self.video_name, frame_num] + tmp)  
            
                        
            # 프레임 표시
            if self.STEP != 5:
                # WorkSpace 생성
                cv.rectangle(self.frame, (self.box_x_ul, self.box_y_ul), (self.box_x_lr, self.box_y_lr), (0, 0, 255), 5)
            
            # 프레임, 손 라벨 표기
            cv.rectangle(self.frame, (0, int(self.height * 0.92)), (self.width, self.height), (204,204,204), -1) # 뒷배경
            cv.putText(self.frame, f'Frame : {frame_num}', (5, (self.height - 10)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
            cv.putText(self.frame, f'| Hand : {self.hand_class_text}', (int(self.width * 0.3), (self.height - 10)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
                
            # CV step box 생성
            self.step_box_draw()
                    
            # 1. put 수행
            if self.STEP == 1:
                if self.WAIT_COUNT == 0:
                    self.place_object(step_list=self.none_to_put, next_class_idx=self.FRONT_IDX)                                  
                elif 0 < self.WAIT_COUNT < 100:
                    self.wait_sequence(next_class_idx=self.FRONT_IDX, step_num=self.STEP)
                    self.error_check(0)
                    self.error_flag = False
                else:
                    # self.FRONT_IMG = self.frame[self.box_y_ul:self.box_y_lr, self.box_x_ul:self.box_x_lr]
                    # self.FRONT_IMG = cv.resize(self.FRONT_IMG, (self.img_width, self.img_height))
                    self.STEP += 1
                    # 테스트
                    # self.STEP = 5
                                            
            # 2. rotate 수행
            elif self.STEP == 2:
                # 2-1. front -> side
                if self.ROTATE_STEP == 1:
                    if self.WAIT_COUNT == 100:
                        self.rotate_step_by_step(step_list=self.front_to_side, pre_class_idx=self.FRONT_IDX, next_class_idx=self.SIDE_IDX)
                    elif 100 < self.WAIT_COUNT < 200:
                        self.wait_sequence(next_class_idx=self.SIDE_IDX, step_num=f'{self.STEP}-{self.ROTATE_STEP}')
                        self.error_check(1)
                        self.error_flag = False
                    else:
                        self.SIDE1_IMG = self.frame[self.box_y_ul:self.box_y_lr, self.box_x_ul:self.box_x_lr]
                        self.SIDE1_IMG = cv.resize(self.SIDE1_IMG, (self.img_width, self.img_height))
                        self.ROTATE_STEP += 1
                            
                # 2-2. side -> back
                elif self.ROTATE_STEP == 2:
                    if self.WAIT_COUNT == 200:
                        self.rotate_step_by_step(step_list=self.side_to_back, pre_class_idx=self.SIDE_IDX, next_class_idx=self.BACK_IDX)
                    elif 200 < self.WAIT_COUNT < 300:
                        self.wait_sequence(next_class_idx=self.BACK_IDX, step_num=f'{self.STEP}-{self.ROTATE_STEP}')
                        self.error_check(2)
                        self.error_flag = False
                    else:
                        self.BACK_IMG = self.frame[self.box_y_ul:self.box_y_lr, self.box_x_ul:self.box_x_lr]
                        self.BACK_IMG = cv.resize(self.BACK_IMG, (self.img_width, self.img_height))
                        self.ROTATE_STEP += 1
                        
                # 2-3. back -> side
                elif self.ROTATE_STEP == 3:
                    if self.WAIT_COUNT == 300:
                        self.rotate_step_by_step(step_list=self.back_to_side, pre_class_idx=self.BACK_IDX, next_class_idx=self.SIDE_IDX)
                    elif 300 < self.WAIT_COUNT < 400:
                        self.wait_sequence(next_class_idx=self.SIDE_IDX, step_num=f'{self.STEP}-{self.ROTATE_STEP}')
                        self.error_check(3)
                        self.error_flag = False
                    else:
                        self.SIDE2_IMG = self.frame[self.box_y_ul:self.box_y_lr, self.box_x_ul:self.box_x_lr]
                        self.SIDE2_IMG = cv.resize(self.SIDE2_IMG, (self.img_width, self.img_height))
                        self.ROTATE_STEP += 1
                    
                # 2-4. side -> front
                elif self.ROTATE_STEP == 4:
                    if self.WAIT_COUNT == 400:
                        self.rotate_step_by_step(step_list=self.side_to_front, pre_class_idx=self.SIDE_IDX, next_class_idx=self.FRONT_IDX)
                    elif 400 < self.WAIT_COUNT < 500:
                        self.wait_sequence(next_class_idx=self.FRONT_IDX, step_num=f'{self.STEP}-{self.ROTATE_STEP}')
                        self.error_check(4)
                        self.error_flag = False
                    else:
                        self.FRONT_IMG = self.frame[self.box_y_ul:self.box_y_lr, self.box_x_ul:self.box_x_lr]
                        self.FRONT_IMG = cv.resize(self.FRONT_IMG, (self.img_width, self.img_height))
                        self.ROTATE_STEP += 1
                else:
                    self.STEP += 1
                    self.ROTATE_STEP = 1
                
            # 3. head 수행
            elif self.STEP == 3:
                if self.WAIT_COUNT == 500:
                    self.show_head_bottom(next_class_idx=self.HEAD_IDX)         
                elif 500 < self.WAIT_COUNT < 600:
                    self.wait_sequence2(self.HEAD_IDX)
                    self.error_check(5)
                    self.error_flag = False
                else:
                    self.HEAD_IMG = self.frame[self.box_y_ul:self.box_y_lr, self.box_x_ul:self.box_x_lr]
                    self.HEAD_IMG = cv.resize(self.HEAD_IMG, (self.img_width, self.img_height))
                    self.STEP += 1
                
            # 4. bottom 수행
            elif self.STEP == 4:
                if self.WAIT_COUNT == 600:
                    self.show_head_bottom(next_class_idx=self.BOTTOM_IDX)
                elif 600 <self.WAIT_COUNT < 700:
                    self.wait_sequence2(self.BOTTOM_IDX)
                    self.error_check(6)
                    self.error_flag = False
                else:
                    self.BOTTOM_IMG = self.frame[self.box_y_ul:self.box_y_lr, self.box_x_ul:self.box_x_lr]
                    self.BOTTOM_IMG = cv.resize(self.BOTTOM_IMG, (self.img_width, self.img_height))
                    self.STEP += 1
                    
            # 5. 결과 출력
            elif self.STEP == 5:
                # 6면에 대해서 결과 출력
                self.text = f"{self.STEP}. 검사가 종료되었습니다."
                cv.rectangle(self.frame, (0, self.height_10_persent), (self.width, int(self.height * 0.17)), (255, 255, 255), -1) # 지시문박스
                self.korean_text(text='제품의 6개의 면을 확인 완료하였습니다',font_size=30, font_color=(0,0,0), font_rocation=(10,self.height_10_persent), bold=True) 
                
                cv.rectangle(self.frame,  (self.width_10_persent, self.height_20_persent), (self.width_30_persent, self.height_50_persent), (0,0,0), -1) 
                self.korean_text(text=f'앞면: {self.error_list[4]}',font_size=20, font_color=(255,255,255), font_rocation=(self.width_10_persent, self.height_20_persent), bold=True)
                self.frame[self.height_30_persent:self.height_30_persent+self.img_height, self.width_10_persent:self.width_10_persent+self.img_width] = self.FRONT_IMG
                
                cv.rectangle(self.frame, (self.width_40_persent, self.height_20_persent), (self.width_60_persent,  self.height_50_persent), (0,0,0), -1) 
                self.korean_text(text=f'뒷면: {self.error_list[2]}',font_size=20, font_color=(255,255,255), font_rocation=(self.width_40_persent, self.height_20_persent), bold=True)
                self.frame[self.height_30_persent:self.height_30_persent+self.img_height, self.width_40_persent:self.width_40_persent+self.img_width] = self.BACK_IMG
                
                cv.rectangle(self.frame, (self.width_70_persent, self.height_20_persent), (self.width_90_persent,  self.height_50_persent), (0,0,0), -1) 
                self.korean_text(text=f'윗면: {self.error_list[5]}',font_size=20, font_color=(255,255,255), font_rocation=(self.width_70_persent, self.height_20_persent), bold=True)
                self.frame[self.height_30_persent:self.height_30_persent+self.img_height, self.width_70_persent:self.width_70_persent+self.img_width] = self.HEAD_IMG
                
                cv.rectangle(self.frame, (self.width_10_persent, self.height_60_persent), (self.width_30_persent,  self.height_90_persent), (0,0,0), -1) 
                self.korean_text(text=f'옆면(1): {self.error_list[1]}',font_size=20, font_color=(255,255,255), font_rocation=(self.width_10_persent, self.height_60_persent), bold=True)
                self.frame[self.height_70_persent:self.height_70_persent+self.img_height, self.width_10_persent:self.width_10_persent+self.img_width] = self.SIDE1_IMG
                
                cv.rectangle(self.frame, (self.width_40_persent, self.height_60_persent), (self.width_60_persent, self.height_90_persent), (0,0,0), -1) 
                self.korean_text(text=f'옆면(2): {self.error_list[3]}',font_size=20, font_color=(255,255,255), font_rocation=(self.width_40_persent, self.height_60_persent), bold=True)
                self.frame[self.height_70_persent:self.height_70_persent+self.img_height, self.width_40_persent:self.width_40_persent+self.img_width] = self.SIDE2_IMG
                
                cv.rectangle(self.frame, (self.width_70_persent, self.height_60_persent), (self.width_90_persent, self.height_90_persent), (0,0,0), -1) 
                self.korean_text(text=f'아랫면: {self.error_list[6]}',font_size=20, font_color=(255,255,255), font_rocation=(self.width_70_persent, self.height_60_persent), bold=True)
                self.frame[self.height_70_persent:self.height_70_persent+self.img_height, self.width_70_persent:self.width_70_persent+self.img_width] = self.BOTTOM_IMG
            

            frame_num += 1

            # cv 영상 확인
            # cv.imshow(self.video_name, self.frame)
            
            #비디오 저장
            self.out.write(self.frame)
            self.update_frame()
            cv.waitKey(1)

        self.out.release()
        self.end_video()
        self.display_result()
    
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()