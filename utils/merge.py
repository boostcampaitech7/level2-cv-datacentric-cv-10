import json
import os

# JSON 파일 경로
json_path_1 = 'json 파일 경로 1'
json_path_2 = 'json 파일 경로 2'
output_path = '결과 저장 directory 경로'

# JSON 파일 읽기
def load_json(file_path):
   with open(file_path, 'r', encoding='utf-8') as f:
       return json.load(f)

try:
   # 두 JSON 파일 읽기
   json1 = load_json(json_path_1)
   json2 = load_json(json_path_2)
   
   # 두 JSON 병합
   merged_json = json1.copy()  # 첫 번째 파일 복사
   
   # 두 번째 파일의 내용 추가
   for img_name, img_data in json2.items():
       if img_name in merged_json:
           # 이미지가 이미 있으면 words 병합
           if 'words' in img_data:
               max_word_id = max([int(word_id) for word_id in merged_json[img_name]['words'].keys()]) if merged_json[img_name]['words'] else 0
               
               for word_data in img_data['words'].values():
                   max_word_id += 1
                   merged_json[img_name]['words'][str(max_word_id).zfill(4)] = word_data
       else:
           # 새로운 이미지면 그대로 추가
           merged_json[img_name] = img_data
   
   # 병합된 JSON 저장
   with open(output_path, 'w', encoding='utf-8') as f:
       json.dump(merged_json, f, indent=2, ensure_ascii=False)
   
   # 이미지 개수 출력
   total_images = len(merged_json)
   print(f"\n병합된 JSON 파일 생성 완료: {output_path}")
   print(f"총 어노테이션된 이미지 개수: {total_images}")
   
   # 원본 파일들의 이미지 개수도 출력
   print(f"\n원본 파일 이미지 개수:")
   print(f"첫번째 json 파일: {len(json1)}")
   print(f"두번째 json 파일: {len(json2)}")
   
except Exception as e:
   print(f"에러 발생: {str(e)}")