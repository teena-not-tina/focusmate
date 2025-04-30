import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

if __name__ == '__main__':
    # 이미지 로드
    img_path = "kdw.jpg"  # 분석할 이미지 경로
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

    # Face Analysis 앱 초기화
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'],
                      providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 얼굴 감지
    faces = app.get(img)
    if len(faces) == 0:
        print("얼굴을 찾을 수 없습니다.")
        exit()

    # 랜드마크 그리기
    tim = img.copy()
    color = (200, 160, 75)
    for face in faces:
        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(int)  # np.int 대신 int 사용
        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
    
    # 결과 저장
    cv2.imwrite('./test_out.jpg', tim)
    print(f"분석된 얼굴 수: {len(faces)}")
    print("결과가 'test_out.jpg'에 저장되었습니다.")