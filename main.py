import cv2
import time
from face_recognition import build_white_list_embeddings, recognize_face

print("🔍 建立允許通行的 embedding 資料庫...")
database = build_white_list_embeddings('face')

if not database:
    print("❌ 沒有找到有效的人臉資料，請將照片放入 white/ 資料夾")
    exit()

print("✅ 資料庫載入完成，啟動攝影機辨識中...")
cap = cv2.VideoCapture(0)
recognized_name = None
start_time = None
required_duration = 3  # 秒

while True:
    ret, frame = cap.read()
    if not ret:
        break

    name, dist = recognize_face(frame, database)

    if name not in ["Unknown", "No face"]:
        if recognized_name == name:
            # 已經持續偵測中
            elapsed = time.time() - start_time
            label = f"✅ {name} welcome（datected {elapsed:.1f} s）"
            color = (0, 255, 0)
            if elapsed >= required_duration:
                print(f"🎉 確認 {name} 持續存在 {required_duration} 秒，自動關閉攝影機")
                break
        else:
            # 首次偵測或辨識到不同人
            recognized_name = name
            start_time = time.time()
            label = f"✅ {name} 允許進入（開始計時）"
            color = (0, 255, 0)
    else:
        # 沒偵測到 or 是未知人 → 重置
        recognized_name = None
        start_time = None
        if name == "Unknown":
            label = "❌ no"
            color = (0, 0, 255)
        else:
            label = "no face"
            color = (128, 128, 128)

    cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Access Control System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 手動結束辨識")
        break

cap.release()
cv2.destroyAllWindows()