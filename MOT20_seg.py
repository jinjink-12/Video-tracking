import cv2
import numpy as np
from ultralytics import YOLO
import os

# YOLOv8 모델 로드
model = YOLO("yolov8n-seg.pt")  # Segmentation 모델 가중치 (Nano 버전 사용)

# 비디오 파일 경로
video_path = "/Users/gimminjin/Downloads/TownCentreXVID.mp4"
cap = cv2.VideoCapture(video_path)

# 출력 비디오 설정
output_path = "segmented_output.mp4"
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# 새로운 이미지를 저장할 디렉토리
output_dir = "./segmented_frames/"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv8로 추론
    results = model(frame)
    masks = results[0].masks.data.cpu().numpy()  # 세그멘테이션 마스크
    classes = results[0].boxes.cls.cpu().numpy()  # 클래스 IDs

    # 사람만 세그멘테이션 (클래스 ID: 0이 'person')
    person_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    person_count = 0  # 사람 수 초기화
    for i, class_id in enumerate(classes):
        if class_id == 0:  # 'person' class
            # 마스크 리사이즈
            resized_mask = cv2.resize(masks[i], (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            person_mask = np.logical_or(person_mask, resized_mask.astype(np.uint8))
            person_count += 1  # 사람 수 증가

    # 배경을 하얀색으로 설정
    person_mask = (person_mask * 255).astype(np.uint8)  # 마스크 스케일 변환
    white_background = np.full_like(frame, 255)  # 하얀색 배경 생성
    segmented_frame = np.where(person_mask[:, :, None], frame, white_background)

    # 사람 개수 표시
    cv2.putText(
        segmented_frame,
        f"Person Count: {person_count}",  # 표시할 텍스트
        (10, 50),  # 텍스트 위치 (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # 폰트
        1.5,  # 폰트 크기
        (0, 0, 255),  # 텍스트 색상 (B, G, R)
        3,  # 텍스트 두께
        cv2.LINE_AA  # 선 종류
    )

    # 출력 비디오 저장
    out.write(segmented_frame)

    # 프레임 저장
    cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.png", segmented_frame)
    frame_count += 1

    # 화면에 표시
    cv2.imshow("Person Segmentation", segmented_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

