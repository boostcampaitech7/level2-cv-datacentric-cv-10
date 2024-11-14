import json
import argparse
import os
from xml.etree import ElementTree as ET
from xml.dom import minidom

def ufo_to_cvat_xml(input_path, output_path):
    """
    UFO 형식의 JSON/CSV 파일을 CVAT XML 형식으로 변환하는 함수
    
    Args:
        input_path (str): 입력 JSON/CSV 파일 경로
        output_path (str): 출력 XML 파일 경로
    """
    # JSON 파일 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        ufo_data = json.load(f)
    
    # XML 기본 구조 생성
    annotations = ET.Element('annotations')
    version = ET.SubElement(annotations, 'version')
    version.text = '1.1'
    
    # 메타데이터 및 라벨 정보 추가
    meta = ET.SubElement(annotations, 'meta')
    task = ET.SubElement(meta, 'task')
    labels = ET.SubElement(task, 'labels')
    label = ET.SubElement(labels, 'label')
    name = ET.SubElement(label, 'name')
    name.text = 'text'
    
    # 이미지별 annotation 처리
    box_id = 0 
    for img_name, img_data in ufo_data["images"].items():
        # 이미지 정보 추가
        image = ET.SubElement(annotations, 'image')
        image.set('name', img_name)
        image.set('width', str(img_data.get('img_w', 1000)))
        image.set('height', str(img_data.get('img_h', 1000)))
        image.set('id', str(len(annotations.findall('image'))))
        
        # 단어(텍스트 박스) 정보 처리
        if "words" in img_data:
            for word_data in img_data["words"].values():
                points = word_data["points"]
                if len(points) == 4:  # 4개의 점으로 구성된 박스만 처리
                    # 점 좌표를 CVAT 형식의 문자열로 변환
                    points_str = ';'.join([f'{p[0]:.1f},{p[1]:.1f}' for p in points])
                    
                    # 폴리곤 정보 추가
                    polygon = ET.SubElement(image, 'polygon')
                    polygon.set('label', 'text')
                    polygon.set('points', points_str)
                    polygon.set('z_order', '0')
                    polygon.set('occluded', '0')
                    polygon.set('id', str(box_id))
                    box_id += 1
    
    # XML 파일로 저장
    xml_str = minidom.parseString(ET.tostring(annotations)).toprettyxml(indent="  ")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

def cvat_to_ufo(input_path, output_path):
    """
    CVAT XML 파일을 UFO JSON 형식으로 변환하는 함수
    
    Args:
        input_path (str): 입력 XML 파일 경로
        output_path (str): 출력 JSON 파일 경로
    """
    # XML 파일 파싱
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # UFO 형식의 기본 구조 생성
    ufo_data = {
        "images": {}
    }
    
    # 이미지별 데이터 처리
    for image in root.findall(".//image"):
        img_name = image.get('name')
        img_width = int(image.get('width'))
        img_height = int(image.get('height'))
        
        # UFO 형식의 이미지 메타데이터 생성
        image_data = {
            "paragraphs": {},
            "words": {},
            "chars": {},
            "img_w": img_width,
            "img_h": img_height,
            "num_patches": None,
            "tags": [],
            "relations": {},
            "annotation_log": {
                "worker": "worker",
                "timestamp": "2024-06-07",
                "tool_version": "",
                "source": None
            },
            "license_tag": {
                "usability": True,
                "public": False,
                "commercial": True,
                "type": None,
                "holder": "Upstage"
            }
        }
        
        # 폴리곤(텍스트 박스) 정보 처리
        for idx, polygon in enumerate(image.findall(".//polygon")):
            points_str = polygon.get('points')
            # 점 좌표 파싱 및 변환
            points = [
                [float(x), float(y)] 
                for x, y in [point.split(',') for point in points_str.split(';')]
            ]
            
            # word ID 생성 및 정보 추가
            word_id = str(idx + 1).zfill(4)
            image_data["words"][word_id] = {
                "transcription": "",  
                "points": points
            }
        
        ufo_data["images"][img_name] = image_data
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ufo_data, f, indent=4, ensure_ascii=False)

def convert_format(input_path, output_path, conversion_type):
    """
    파일 형식 변환을 처리하는 메인 함수
    
    Args:
        input_path (str): 입력 파일 경로
        output_path (str): 출력 파일 경로
        conversion_type (str): 변환 타입 ('to_xml' 또는 'to_json')
    """
    # 입력 파일 존재 확인
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 변환 타입에 따른 처리
    if conversion_type == "to_xml":
        ufo_to_cvat_xml(input_path, output_path)
 