import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time
import os
import subprocess

# YOLO 모델 로드
model = YOLO("yolo11n.pt")

# 비디오 파일 경로
video_path = "/Users/gimminjin/Downloads/TownCentreXVID.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 정보 추출
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 출력 비디오 설정
output_video_path = "trajectory_visualization.mp4"
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# 궤적 저장 및 색상 저장
track_history = defaultdict(lambda: [])
track_colors = {}

# 흰색 배경 초기화
canvas = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)  # 흰색 배경

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # YOLO 객체 탐지 및 추적
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()  # 바운딩 박스
        track_ids = results[0].boxes.id.int().cpu().tolist()  # 추적 ID

        

        # 궤적 업데이트
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x, y, w, h = map(int, (x, y, w, h))
            track = track_history[track_id]
            track.append((x, y))

            # 궤적 색상을 ID별로 저장 (상의 색상 기반)
            if track_id not in track_colors:
                # 바운딩 박스 상단 중앙의 작은 영역의 평균 색상 추출
                top_area = frame[max(0, y - h // 4):y, x:x + w // 2]
                if top_area.size > 0:
                    avg_color = np.mean(top_area, axis=(0, 1)).astype(int).tolist()
                    track_colors[track_id] = tuple(avg_color)

            # 궤적을 흰색 배경에 그리기
            if track_id in track_colors:
                color = track_colors[track_id]
                for i in range(1, len(track)):
                    cv2.line(
                        canvas,
                        (track[i - 1][0], track[i - 1][1]),
                        (track[i][0], track[i][1]),
                        color,
                        2,  # 선 두께
                        cv2.LINE_AA  # 안티 앨리어싱 처리
                    )

        # 흰색 배경에 궤적이 추가된 캔버스와 현재 프레임을 합성
        overlay = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)

        # 비디오 저장
        out.write(overlay)

        # 화면에 표시
        cv2.imshow("Trajectory Visualization", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# 최종 궤적만 남겨진 이미지를 저장
output_image_path = "final_trajectory_image.png"
cv2.imwrite(output_image_path, canvas)

# MacOS에서 저장된 이미지 및 비디오 열기
subprocess.call(["open", output_video_path])
subprocess.call(["open", output_image_path])
