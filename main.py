import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import threading
import queue
import time
from scipy.signal import butter, filtfilt

class VoiceCommandSystem:
    def __init__(self):
        # 音頻參數
        self.sample_rate = 16000  # 取樣率
        self.duration = 2.0  # 錄音時長（秒）
        self.n_mels = 128  # 梅爾頻譜圖的頻帶數
        self.n_fft = 2048  # FFT窗口大小
        self.hop_length = 512  # 跳躍長度
        
        # 系統參數
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.model = None
        
        # 指令類別（需要根據實際需求定義）
        self.command_classes = [
            # 待補充：定義您的指令類別
            # 例如：'開燈', '關燈', '播放音樂', '停止', '音量調高', '音量調低'
        ]
        
        self.load_model()
    
    def preprocess_audio(self, audio_data):
        """
        預處理音頻數據
        """
        # 正規化音頻
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 應用預加重濾波器
        pre_emphasis = 0.97
        audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # 去除靜音部分
        audio_data = self.remove_silence(audio_data)
        
        # 確保音頻長度一致
        target_length = int(self.sample_rate * self.duration)
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        else:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
        
        return audio_data
    
    def remove_silence(self, audio_data, threshold=0.01):
        """
        移除音頻中的靜音部分
        """
        # 計算短時能量
        frame_length = int(0.025 * self.sample_rate)  # 25ms
        frame_shift = int(0.01 * self.sample_rate)   # 10ms
        
        energy = []
        for i in range(0, len(audio_data) - frame_length, frame_shift):
            frame = audio_data[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        # 找到語音活動區域
        energy = np.array(energy)
        voice_activity = energy > (threshold * np.max(energy))
        
        if np.any(voice_activity):
            start_idx = np.where(voice_activity)[0][0] * frame_shift
            end_idx = np.where(voice_activity)[0][-1] * frame_shift + frame_length
            return audio_data[start_idx:end_idx]
        else:
            return audio_data
    
    def audio_to_mel_spectrogram(self, audio_data):
        """
        將音頻轉換成梅爾頻譜圖
        """
        # 計算梅爾頻譜圖
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # 轉換為對數刻度
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 正規化到 [0, 1] 範圍
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        return mel_spec_normalized
    
    def build_cnn_model(self):
        """
        建立CNN分類器模型
        """
        model = keras.Sequential([
            # 輸入層
            layers.Input(shape=(self.n_mels, None, 1)),
            
            # 第一層卷積
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二層卷積
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三層卷積
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 全域平均池化
            layers.GlobalAveragePooling2D(),
            
            # 全連接層
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.command_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_model(self):
        """
        載入預訓練模型
        """
        try:
            # 嘗試載入預訓練模型
            self.model = keras.models.load_model('voice_command_model.h5')
            print("成功載入預訓練模型")
        except:
            # 如果沒有預訓練模型，建立新模型
            print("未找到預訓練模型，建立新模型")
            self.model = self.build_cnn_model()
            print("模型架構:")
            self.model.summary()
    
    def predict_command(self, mel_spectrogram):
        """
        使用CNN模型預測指令
        """
        # 準備輸入數據
        input_data = np.expand_dims(mel_spectrogram, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        
        # 進行預測
        predictions = self.model.predict(input_data)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return predicted_class, confidence
    
    def execute_command(self, command_index, confidence):
        """
        執行對應的指令
        """
        if confidence < 0.7:  # 信心度閾值
            print(f"指令識別信心度不足: {confidence:.2f}")
            return
        
        if command_index < len(self.command_classes):
            command = self.command_classes[command_index]
            print(f"執行指令: {command} (信心度: {confidence:.2f})")
            
            # 指令執行邏輯
            if command == '開燈':
                # 開燈指令
                pass
            elif command == '關燈':
                # 關燈指令
                pass
            elif command == '播放音樂':
                # 播放音樂指令
                pass
            elif command == '停止':
                # 停止指令
                pass
            elif command == '音量調高':
                # 音量調高指令
                pass
            elif command == '音量調低':
                # 音量調低指令
                pass
            else:
                print(f"未定義的指令: {command}")
        else:
            print("無效的指令索引")
    
    def audio_callback(self, indata, frames, time, status):
        """
        音頻回調函數
        """
        if status:
            print(f"音頻錄製狀態: {status}")
        
        # 將音頻數據放入隊列
        self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        """
        開始錄音
        """
        self.is_recording = True
        
        # 開始音頻串流
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms緩衝
        ):
            print("開始語音指令識別...")
            print("按 'q' 退出")
            
            while self.is_recording:
                try:
                    # 收集音頻數據
                    audio_buffer = []
                    start_time = time.time()
                    
                    while time.time() - start_time < self.duration:
                        if not self.audio_queue.empty():
                            audio_chunk = self.audio_queue.get()
                            audio_buffer.append(audio_chunk.flatten())
                    
                    if audio_buffer:
                        # 合併音頻數據
                        audio_data = np.concatenate(audio_buffer)
                        
                        # 預處理音頻
                        processed_audio = self.preprocess_audio(audio_data)
                        
                        # 轉換為梅爾頻譜圖
                        mel_spec = self.audio_to_mel_spectrogram(processed_audio)
                        
                        # 預測指令
                        command_index, confidence = self.predict_command(mel_spec)
                        
                        # 執行指令
                        self.execute_command(command_index, confidence)
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"處理音頻時發生錯誤: {e}")
    
    def stop_recording(self):
        """
        停止錄音
        """
        self.is_recording = False
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        訓練模型
        """
        # 資料預處理
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        
        # 設定回調函數
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint('voice_command_model.h5', save_best_only=True)
        ]
        
        # 訓練模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def visualize_mel_spectrogram(self, mel_spectrogram, title="梅爾頻譜圖"):
        """
        視覺化梅爾頻譜圖
        """
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(
            mel_spectrogram,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()

def main():
    """
    主程式
    """
    # 建立語音指令系統
    voice_system = VoiceCommandSystem()
    
    try:
        # 開始錄音和指令識別
        voice_system.start_recording()
    except KeyboardInterrupt:
        print("\n停止語音指令識別")
    finally:
        voice_system.stop_recording()

if __name__ == "__main__":
    main()

# 所需套件安裝：
# pip install numpy librosa sounddevice tensorflow matplotlib scipy