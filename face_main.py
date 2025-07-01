import cv2
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import load_model
import time

class FaceRecognitionSystem:
    def __init__(self, model_path=None, embedding_db_path="face_embeddings.pkl", 
                 confidence_threshold=0.6, k_neighbors=3):
        """
        初始化臉部辨識系統
        
        Args:
            model_path: FaceNet模型路徑
            embedding_db_path: 已知臉部embedding資料庫路徑
            confidence_threshold: 辨識信心度閾值
            k_neighbors: KNN中的k值
        """
        self.model_path = model_path
        self.embedding_db_path = embedding_db_path
        self.confidence_threshold = confidence_threshold
        self.k_neighbors = k_neighbors
        
        # 初始化模型和分類器
        self.facenet_model = None
        self.knn_classifier = None
        self.label_encoder = None
        self.known_embeddings = []
        self.known_labels = []
        
        # 人臉檢測器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 載入模型和資料庫
        self._load_facenet_model()
        self._load_embedding_database()
    
    def _load_facenet_model(self):
        """載入FaceNet模型"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.facenet_model = load_model(self.model_path)
                print("FaceNet模型載入成功")
            except Exception as e:
                print(f"載入FaceNet模型失敗: {e}")
                # 使用預訓練模型或其他替代方案
                # TODO: 實作替代的embedding方法
                pass
        else:
            print("未指定FaceNet模型路徑或檔案不存在")
            # TODO: 下載或使用預設的FaceNet模型
            pass
    
    def _load_embedding_database(self):
        """載入已知臉部embedding資料庫"""
        if os.path.exists(self.embedding_db_path):
            try:
                with open(self.embedding_db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_embeddings = data['embeddings']
                    self.known_labels = data['labels']
                
                # 訓練KNN分類器
                if len(self.known_embeddings) > 0:
                    self.label_encoder = LabelEncoder()
                    encoded_labels = self.label_encoder.fit_transform(self.known_labels)
                    self.knn_classifier = KNeighborsClassifier(n_neighbors=self.k_neighbors)
                    self.knn_classifier.fit(self.known_embeddings, encoded_labels)
                    print(f"載入{len(self.known_embeddings)}個已知臉部embedding")
                else:
                    print("embedding資料庫為空")
            except Exception as e:
                print(f"載入embedding資料庫失敗: {e}")
        else:
            print("embedding資料庫不存在，將建立新的資料庫")
            self.known_embeddings = []
            self.known_labels = []
    
    def preprocess_face(self, face_img):
        """
        預處理臉部影像以供FaceNet使用
        
        Args:
            face_img: 裁切後的臉部影像
            
        Returns:
            preprocessed_img: 預處理後的影像
        """
        # 調整大小到FaceNet輸入尺寸 (通常是160x160)
        face_resized = cv2.resize(face_img, (160, 160))
        
        # 正規化像素值到[-1, 1]
        face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
        
        # 增加batch維度
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def get_face_embedding(self, face_img):
        """
        使用FaceNet獲取臉部embedding
        
        Args:
            face_img: 臉部影像
            
        Returns:
            embedding: 臉部embedding向量
        """
        if self.facenet_model is None:
            # TODO: 實作替代的embedding方法
            print("FaceNet模型未載入，無法生成embedding")
            return None
        
        try:
            preprocessed_face = self.preprocess_face(face_img)
            embedding = self.facenet_model.predict(preprocessed_face)
            return embedding[0]  # 移除batch維度
        except Exception as e:
            print(f"生成embedding失敗: {e}")
            return None
    
    def detect_faces(self, frame):
        """
        檢測影像中的人臉
        
        Args:
            frame: 輸入影像
            
        Returns:
            faces: 檢測到的人臉座標列表
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def recognize_face(self, face_embedding):
        """
        使用KNN辨識臉部
        
        Args:
            face_embedding: 臉部embedding向量
            
        Returns:
            name: 辨識結果名稱
            confidence: 信心度
        """
        if self.knn_classifier is None or face_embedding is None:
            return "Unknown", 0.0
        
        try:
            # 預測
            distances, indices = self.knn_classifier.kneighbors([face_embedding])
            
            # 計算信心度（距離越小信心度越高）
            avg_distance = np.mean(distances[0])
            confidence = max(0, 1 - avg_distance)  # 簡單的信心度計算
            
            if confidence > self.confidence_threshold:
                predicted_label = self.knn_classifier.predict([face_embedding])[0]
                name = self.label_encoder.inverse_transform([predicted_label])[0]
                return name, confidence
            else:
                return "Unknown", confidence
                
        except Exception as e:
            print(f"臉部辨識失敗: {e}")
            return "Unknown", 0.0
    
    def add_known_face(self, face_img, name):
        """
        新增已知臉部到資料庫
        
        Args:
            face_img: 臉部影像
            name: 人名
        """
        embedding = self.get_face_embedding(face_img)
        if embedding is not None:
            self.known_embeddings.append(embedding)
            self.known_labels.append(name)
            
            # 重新訓練KNN分類器
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(self.known_labels)
            self.knn_classifier = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            self.knn_classifier.fit(self.known_embeddings, encoded_labels)
            
            print(f"已新增 {name} 到臉部資料庫")
            return True
        return False
    
    def save_embedding_database(self):
        """儲存embedding資料庫"""
        try:
            data = {
                'embeddings': self.known_embeddings,
                'labels': self.known_labels
            }
            with open(self.embedding_db_path, 'wb') as f:
                pickle.dump(data, f)
            print("embedding資料庫已儲存")
        except Exception as e:
            print(f"儲存embedding資料庫失敗: {e}")
    
    def process_frame(self, frame):
        """
        處理單一影像幀
        
        Args:
            frame: 輸入影像幀
            
        Returns:
            processed_frame: 處理後的影像幀
            recognition_results: 辨識結果列表
        """
        recognition_results = []
        
        # 檢測人臉
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # 裁切臉部
            face_img = frame[y:y+h, x:x+w]
            
            # 獲取embedding並辨識
            embedding = self.get_face_embedding(face_img)
            name, confidence = self.recognize_face(embedding)
            
            # 儲存辨識結果
            result = {
                'bbox': (x, y, w, h),
                'name': name,
                'confidence': confidence,
                'embedding': embedding
            }
            recognition_results.append(result)
            
            # 在影像上繪製結果
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # 顯示名稱和信心度
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
        
        return frame, recognition_results
    
    def run_recognition(self):
        """執行即時臉部辨識"""
        # 開啟攝影機
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("無法開啟攝影機")
            return
        
        print("開始臉部辨識系統...")
        print("按 'q' 退出")
        print("按 's' 儲存當前辨識到的臉部")
        print("按 'a' 新增新的臉部到資料庫")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                break
            
            # 處理影像幀
            processed_frame, results = self.process_frame(frame)
            
            # 顯示結果
            cv2.imshow('Face Recognition System', processed_frame)
            
            # 處理按鍵事件
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 儲存embedding資料庫
                self.save_embedding_database()
            elif key == ord('a'):
                # 新增臉部到資料庫
                if len(results) > 0:
                    # TODO: 實作新增臉部的界面
                    print("請實作新增臉部的功能")
                    pass
            
            # 執行辨識後的動作
            self.perform_recognition_actions(results)
        
        # 清理資源
        cap.release()
        cv2.destroyAllWindows()
    
    def perform_recognition_actions(self, recognition_results):
        """
        根據辨識結果執行相應動作
        
        Args:
            recognition_results: 辨識結果列表
        """
        for result in recognition_results:
            name = result['name']
            confidence = result['confidence']
            
            # TODO: 根據不同的辨識結果執行不同動作
            if name != "Unknown" and confidence > self.confidence_threshold:
                # 已知人員的處理邏輯
                # 例如：記錄出入時間、發送通知、控制門禁等
                pass
            else:
                # 未知人員的處理邏輯
                # 例如：記錄陌生人、發送警告等
                pass

def main():
    """主程式"""
    # 初始化臉部辨識系統
    # TODO: 設定正確的FaceNet模型路徑
    face_recognition = FaceRecognitionSystem(
        model_path=None,  # 請設定FaceNet模型路徑
        embedding_db_path="face_embeddings.pkl",
        confidence_threshold=0.6,
        k_neighbors=3
    )
    
    # 執行臉部辨識
    face_recognition.run_recognition()

if __name__ == "__main__":
    main()