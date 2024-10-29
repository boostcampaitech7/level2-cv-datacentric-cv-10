import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import json
import os
import albumentations as A
import numpy as np

# 데이터 디렉토리 경로 설정
data_dir = '/data/ephemeral/home/level2-cv-datacentric-cv-10/data'

# 사이드바에서 언어 선택
languages = ['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt']
language = st.sidebar.selectbox('언어 선택', languages)

# 이미지 및 어노테이션 경로 설정
img_dir = os.path.join(data_dir, language, 'img', 'train')
anno_file = os.path.join(data_dir, language, 'ufo', 'train.json')

# 어노테이션 로드
with open(anno_file, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# 이미지 파일명 목록 가져오기
image_files = sorted(os.listdir(img_dir))

# 세션 상태에서 이미지 인덱스 초기화
if 'index' not in st.session_state:
    st.session_state.index = 0

# 증강 옵션 설정
st.sidebar.subheader("증강 옵션")
apply_clahe = st.sidebar.checkbox("CLAHE 적용")
clahe_clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 4.0, 2.0) if apply_clahe else 2.0

apply_rotation = st.sidebar.checkbox("랜덤 회전 적용")
rotation_limit = st.sidebar.slider("회전 각도", -45, 45, 15) if apply_rotation else 15

apply_random_crop = st.sidebar.checkbox("랜덤 크롭 적용")
crop_scale = st.sidebar.slider("크롭 비율", 0.5, 1.0, 0.9) if apply_random_crop else 0.9

apply_optical_distortion = st.sidebar.checkbox("광학 왜곡 적용")
distort_limit = st.sidebar.slider("왜곡 강도", 0.1, 0.5, 0.2) if apply_optical_distortion else 0.2

# 증강 함수 정의
def apply_augmentations(image_np):
    augmentations = []

    if apply_clahe:
        augmentations.append(A.CLAHE(clip_limit=clahe_clip_limit, tile_grid_size=(8, 8), p=1.0))
    
    if apply_rotation:
        augmentations.append(A.Rotate(limit=rotation_limit, p=1.0))
    
    if apply_random_crop:
        height, width = image_np.shape[:2]
        crop_size = int(min(height, width) * crop_scale)
        augmentations.append(A.RandomCrop(width=crop_size, height=crop_size, p=1.0))
    
    if apply_optical_distortion:
        augmentations.append(A.OpticalDistortion(distort_limit=distort_limit, shift_limit=0.2, p=1.0))

    # Albumentations 시퀀스로 이미지 변환
    augmentation_pipeline = A.Compose(augmentations)
    augmented = augmentation_pipeline(image=image_np)
    return augmented['image']

# 이미지 로드 및 바운딩 박스 그리기 함수
def load_image_with_boxes(image_path, image_filename, line_width=5):
    image = Image.open(image_path).convert('RGB')
    # EXIF 정보를 기준으로 이미지 회전 수정
    image = ImageOps.exif_transpose(image)
    draw = ImageDraw.Draw(image)

    # 현재 이미지의 어노테이션 가져오기
    image_annotation = annotations['images'].get(image_filename, {})

    # 바운딩 박스 그리기
    if 'words' in image_annotation:
        for word_info in image_annotation['words'].values():
            points = word_info['points']
            num_points = len(points)
            for i in range(num_points):
                start_point = tuple(points[i])
                end_point = tuple(points[(i + 1) % num_points])
                draw.line([start_point, end_point], fill='red', width=line_width)
    return np.array(image)

# 네비게이션 버튼
col1, col2 = st.columns([1, 1])
with col1:
    if st.button('이전'):
        if st.session_state.index > 0:
            st.session_state.index -= 1
with col2:
    if st.button('다음'):
        if st.session_state.index < len(image_files) - 1:
            st.session_state.index += 1

# 현재 이미지 파일명 가져오기
current_image_file = image_files[st.session_state.index]
image_path = os.path.join(img_dir, current_image_file)

# 원본 이미지와 증강된 이미지 로드
image_np = load_image_with_boxes(image_path, current_image_file, line_width=5)  # 선 두께 조절
augmented_image_np = apply_augmentations(image_np)  # 증강 적용

# NumPy 배열을 Pillow 이미지로 변환
image = Image.fromarray(image_np)
augmented_image = Image.fromarray(augmented_image_np)

# 원본 및 증강된 이미지 비교 출력 (좌우 배치)
st.subheader(f"{language} - {current_image_file}")
col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="원본 이미지", use_column_width=True)
with col2:
    st.image(augmented_image, caption="증강 이미지", use_column_width=True)
