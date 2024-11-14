import json
import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from datasets import load_dataset

# JSON 형식의 ground_truth를 딕셔너리로 변환하는 함수
def parse_ground_truth(ground_truth_str):
    return json.loads(ground_truth_str)

# BBox를 그리기 위해 이미지 변환
def draw_bboxes(image, bboxes):
    # 이미지가 PIL 객체라면 np.array로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)

    # OpenCV가 BGR 형식을 사용하기 때문에 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for bbox in bboxes:
        quad = bbox["quad"]
        x1, y1 = quad["x1"], quad["y1"]
        x2, y2 = quad["x2"], quad["y2"]
        x3, y3 = quad["x3"], quad["y3"]
        x4, y4 = quad["x4"], quad["y4"]

        # 사각형으로 연결할 점들
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # BGR 이미지를 다시 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 이미지와 어노테이션 정보를 표시하는 함수 정의
def display_sample(sample):
    # 이미지 로드
    image = sample["image"]
    if isinstance(image, str):
        image = Image.open(image)

    # ground_truth JSON 문자열 파싱
    ground_truth_str = sample["ground_truth"]
    ground_truth = parse_ground_truth(ground_truth_str)

    # valid_line에서 BBox 정보 추출, words 키가 있는 경우만 사용
    valid_lines = ground_truth.get("valid_line", [])
    bboxes = [
        word for line in valid_lines if "words" in line 
        for word in line["words"] if "quad" in word
    ]

    # 이미지에 BBox 그리기
    image_with_bboxes = draw_bboxes(image, bboxes)
    st.image(image_with_bboxes, caption="Sample Image with BBoxes", use_column_width=True)

    # 메뉴 항목 표시
    menu_items = ground_truth.get("gt_parse", {}).get("menu", [])
    
    st.write("### Menu Items")
    if menu_items:
        for item in menu_items:
            st.write(f"Name: {item['nm']}, Count: {item['cnt']}, Price: {item['price']}")
    
    # Sub-total, service, tax, etc. 정보 표시
    sub_total_info = ground_truth.get("gt_parse", {}).get("sub_total", {})
    st.write("### Sub-total Information")
    for key, value in sub_total_info.items():
        st.write(f"{key.replace('_', ' ').title()}: {value}")
    
    # Total 가격 정보 표시
    total_info = ground_truth.get("gt_parse", {}).get("total", {})
    st.write("### Total Information")
    for key, value in total_info.items():
        st.write(f"{key.replace('_', ' ').title()}: {value}")

# Streamlit 앱 구조 정의
st.title("CORD Dataset Sample Viewer")

# 데이터셋 불러오기
dataset_url = "https://huggingface.co/datasets/naver-clova-ix/cord-v2"
st.write(f"Dataset source: [CORD v2]({dataset_url})")

try:
    # 데이터셋 로드
    dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
    
    sample_index = st.slider("Sample Index", min_value=0, max_value=len(dataset)-1, value=0)
    sample = dataset[sample_index]
    
    display_sample(sample)
except Exception as e:
    st.error("Error loading dataset or sample.")
    st.write(e)
