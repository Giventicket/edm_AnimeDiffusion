import os
from skimage.metrics import structural_similarity as ssim
import cv2

# 경로 설정
real_dir = '/data/Anime/test_data/reference'
generated_dir = './result_same_no_weight_log_uniform_importance'

# 모든 이미지의 SSIM을 저장할 리스트
ssim_scores = []

# 이미지 파일 리스트
real_files = sorted(os.listdir(real_dir))
generated_files = sorted(os.listdir(generated_dir))

for real_file, gen_file in zip(real_files, generated_files):
    real_path = os.path.join(real_dir, real_file)
    gen_path = os.path.join(generated_dir, gen_file)

    img_real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    
    img_gen = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)

    img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))

    score, _ = ssim(img_real, img_gen, full=True)
    ssim_scores.append(score)

    print(f"{real_file} vs {gen_file} ➔ SSIM: {score:.4f}")

average_ssim = sum(ssim_scores) / len(ssim_scores)
print(f"📊 평균 SSIM Score: {average_ssim:.4f}")
