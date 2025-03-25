import os
import glob
from PIL import Image

# 스케치 이미지 목록을 기준으로 인덱스를 가져옵니다.
sketch_files = sorted(glob.glob('/data/Anime/test_data/sketch/*.jpg'))
pdf_pages = []

for sketch_file in sketch_files:
    basename = os.path.basename(sketch_file)  # 예: "1.jpg"
    idx = os.path.splitext(basename)[0]
    
    ref_file = f'/data/Anime/test_data/reference/{idx}.jpg'
    output_file = f'/root/AnimeDiffusion/result_same/ret_{idx}.jpg'
    
    if os.path.exists(ref_file) and os.path.exists(output_file):
        # 각 이미지 열기
        sketch = Image.open(sketch_file)
        reference = Image.open(ref_file)
        output = Image.open(output_file)
        
        # 동일한 높이로 리사이즈 (세 이미지 중 가장 작은 높이에 맞춤)
        height = min(sketch.height, reference.height, output.height)
        sketch = sketch.resize((int(sketch.width * height / sketch.height), height))
        reference = reference.resize((int(reference.width * height / reference.height), height))
        output = output.resize((int(output.width * height / output.height), height))
        
        # 세 이미지를 좌우로 합치기
        total_width = sketch.width + reference.width + output.width
        combined = Image.new('RGB', (total_width, height))
        combined.paste(sketch, (0, 0))
        combined.paste(reference, (sketch.width, 0))
        combined.paste(output, (sketch.width + reference.width, 0))
        
        pdf_pages.append(combined)
    else:
        print(f"파일이 누락되었습니다. idx: {idx}")

# PDF로 저장 (첫 페이지 이미지에 나머지 이미지들을 추가)
if pdf_pages:
    pdf_pages[0].save('combined_output.pdf', "PDF", resolution=100.0, save_all=True, append_images=pdf_pages[1:])
    print("PDF 저장 완료: combined_output.pdf")
else:
    print("PDF로 저장할 이미지가 없습니다.")
