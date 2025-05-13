import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os

app = FaceAnalysis(providers=['CPUExecutionProvider'])  # CUDA 제거
app.prepare(ctx_id=0, det_size=(640, 640))

# 현재 디렉토리의 이미지 경로 지정
img_path = "cha.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

faces = app.get(img)
if len(faces) == 0:
    print("얼굴을 찾을 수 없습니다.")
else:
    # 얼굴 분석 결과를 직접 그리기
    for face in faces:
        bbox = face.bbox.astype(int)  # np.int 대신 int 사용
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

    cv2.imwrite("./t1_output.jpg", img)
    print(f"분석된 얼굴 수: {len(faces)}")