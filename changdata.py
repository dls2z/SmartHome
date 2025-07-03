import os
import numpy as np
from face_recognition import extract_face, get_embedding
import cv2

face_folder = 'face'
data_folder = 'facedata'
os.makedirs(data_folder, exist_ok=True)

grouped = {}

for filename in os.listdir(face_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(face_folder, filename)
        image = cv2.imread(path)
        face = extract_face(image)
        if face is not None:
            embedding = get_embedding(face)
            name = os.path.splitext(filename)[0].split('_')[0]
            grouped.setdefault(name, []).append(embedding)
        else:
            print(f"[警告] 無法從 {filename} 擷取人臉")

for name, embeddings in grouped.items():
    avg_emb = np.mean(embeddings, axis=0)
    np.save(os.path.join(data_folder, f"{name}.npy"), avg_emb)
    print(f"✅ 儲存 {name}.npy 完成")