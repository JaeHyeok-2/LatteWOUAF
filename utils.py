import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2 



def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 프레임을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 프레임을 Tensor로 변환하고 정규화 (0-1 범위)
        frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        frames.append(frame_tensor)
    
    cap.release()

    # 모든 프레임을 스택으로 쌓아 하나의 Tensor로 변환
    frames_tensor = torch.stack(frames)  # [T, C, H, W] 형태

    return frames_tensor

class VideoLatentDataset(Dataset):
    def __init__(self, latent_z, video_paths):
        self.latent_z = latent_z
        self.video_paths = video_paths

    def __len__(self):
        return len(self.latent_z)

    def __getitem__(self, idx):
        latent_z = self.latent_z[idx]  # [16, 4, 32, 32] 형태
        video_path = self.video_paths[idx]  # 해당 비디오 경로
        
        return latent_z, video_path

# # JSON 파일 불러오기
# with open('latent_z.json', 'r') as f:
#     data = json.load(f)

# latent_z = torch.tensor(data['latent_z'])  # [66, 16, 4, 32, 32] 형태
# video_paths = data['video_paths']  # 66개의 비디오 경로

# # 66개의 비디오 중 3/4는 training, 1/4는 validation으로 분할
# train_latent_z, val_latent_z, train_video_paths, val_video_paths = train_test_split(
#     latent_z, video_paths, test_size=0.25, random_state=42
# )

# # Dataset 인스턴스 생성
# train_dataset = VideoLatentDataset(train_latent_z, train_video_paths)
# val_dataset = VideoLatentDataset(val_latent_z, val_video_paths)

# # DataLoader 생성
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# DataLoader 사용 예시
# for batch_idx, (latent_batch, video_paths_batch) in enumerate(train_loader):
#     print(f"Training Batch {batch_idx + 1}:")
#     print(f"Latent batch shape: {latent_batch.shape}")  # 기대: [4, 16, 4, 32, 32]
#     print(f"Video paths: {video_paths_batch}")
#     print()

# for batch_idx, (latent_batch, video_paths_batch) in enumerate(val_loader):
#     print(f"Validation Batch {batch_idx + 1}:")
#     print(f"Latent batch shape: {latent_batch.shape}")  # 기대: [4, 16, 4, 32, 32]
#     print(f"Video paths: {video_paths_batch}")
#     print()


# tensor = load_video_frames('test/sample_0.mp4')



