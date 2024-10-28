import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import json
import os

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
            # 점을 연결하여 선 그리기
            num_points = len(points)
            for i in range(num_points):
                start_point = tuple(points[i])
                end_point = tuple(points[(i + 1) % num_points])
                draw.line([start_point, end_point], fill='red', width=line_width)
    return image

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

# 이미지 로드 및 바운딩 박스 그리기
image = load_image_with_boxes(image_path, current_image_file, line_width=5)  # 선 두께 조절

st.image(image, caption=f"{language} - {current_image_file}")
