import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import os
from sklearn.preprocessing import Normalizer
from keras_facenet import FaceNet

# 載入模型
embedder = FaceNet()
model = embedder.model
l2_normalizer = Normalizer('l2')
mp_face_detection = mp.solutions.face_detection

def extract_face(image):
    """ 使用 MediaPipe 偵測人臉並裁切 """
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = image.shape[:2]
                x, y, w_box, h_box = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                x, y = max(x, 0), max(y, 0)
                return cv2.resize(image[y:y+h_box, x:x+w_box], (160, 160))
    return None

def get_embedding(face_img):
    """ 把臉部圖像轉為128維向量 """
    face = face_img.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    embedding = model.predict(face)[0]
    return l2_normalizer.transform([embedding])[0]

def build_white_list_embeddings(white_folder='white'):
    """將同一人多張圖片的 embeddings 合併為平均向量"""
    database = {}
    grouped = {}  # key: 人名, value: list of embeddings

    for filename in os.listdir(white_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(white_folder, filename)
            image = cv2.imread(path)
            face = extract_face(image)
            if face is not None:
                embedding = get_embedding(face)
                # 支援同名多張，例如 myface1_1.jpg、myface1_2.jpg → name = myface1
                name = os.path.splitext(filename)[0].split('_')[0]
                if name not in grouped:
                    grouped[name] = []
                grouped[name].append(embedding)
            else:
                print(f"[警告] 無法從 {filename} 擷取人臉")

    for name, embeddings in grouped.items():
        avg_embedding = np.mean(embeddings, axis=0)  # 可改 median 更穩定
        database[name] = avg_embedding

    return database

def recognize_face(image, database, threshold=0.7):
    """ 與 white list 比對，返回最接近的名字與距離 """
    face = extract_face(image)
    if face is None:
        return "No face", None
    embedding = get_embedding(face)
    min_dist = float('inf')
    identity = "Unknown"
    for name, db_emb in database.items():
        dist = np.linalg.norm(embedding - db_emb)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > threshold:
        return "Unknown", min_dist
    return identity, min_dist
