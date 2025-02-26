import cv2
import joblib
import numpy as np

# Muat model Decision Tree dan scaler
dtree = joblib.load('dtree_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Variabel untuk menghitung akurasi
correct_predictions = 0
total_predictions = 0

# Warna tambahan dari dataset terbaru
color_map = {
    'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0),
    'yellow': (0, 255, 255), 'orange': (255, 165, 0), 'purple': (128, 0, 128),
    'pink': (255, 192, 203), 'cyan': (0, 255, 255), 'gray': (128, 128, 128),
    'brown': (139, 69, 19), 'white': (255, 255, 255), 'black': (0, 0, 0),
    'light blue': (173, 216, 230), 'steel blue': (70, 130, 180), 'deep sky blue': (0, 191, 255),
    'pale green': (152, 251, 152), 'medium sea green': (60, 179, 113), 'forest green': (34, 139, 34),
    'dark orange': (255, 140, 0), 'tomato': (255, 99, 71), 'red orange': (255, 69, 0),
    'indigo': (75, 0, 130), 'medium orchid': (186, 85, 211), 'plum': (221, 160, 221),
    'khaki': (240, 230, 140), 'goldenrod': (218, 165, 32), 'dark goldenrod': (184, 134, 11),
    'silver': (192, 192, 192), 'peach': (255, 223, 186), 'lime': (173, 255, 47),
    'teal': (0, 128, 128), 'charcoal': (70, 70, 70), 'brick red': (165, 42, 42)
}

# Tetapkan posisi tetap untuk kotak deteksi
box_size = 50
center_x, center_y = 320, 240  # Posisi tetap di tengah layar

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi frame ke HSV untuk deteksi warna yang lebih akurat
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Ambil area kotak untuk deteksi warna
    roi = frame[center_y-box_size//2:center_y+box_size//2, center_x-box_size//2:center_x+box_size//2]
    avg_color = np.mean(roi, axis=(0, 1)).astype(int)
    avg_color = avg_color.reshape(1, -1)
    avg_color_scaled = scaler.transform(avg_color)
    
    # Prediksi warna dari model Decision Tree
    predicted_color = dtree.predict(avg_color_scaled)[0]
    
    # Perhitungan akurasi
    if predicted_color in color_map:
        correct_predictions += 1
    
    total_predictions += 1
    accuracy = max(0, min((correct_predictions / total_predictions) * 100, 100))  # Jaga dalam rentang 0-100%
    
    # Gambar kotak deteksi warna
    cv2.rectangle(frame, (center_x - box_size//2, center_y - box_size//2), 
                  (center_x + box_size//2, center_y + box_size//2), (255, 255, 255), 2)
    
    # Tampilkan warna yang terdeteksi dan akurasi
    if predicted_color in color_map:
        cv2.putText(frame, f'Color: {predicted_color.capitalize()}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map[predicted_color], 2)
    
    cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Tampilkan frame
    cv2.imshow('Color Detection', frame)
    print(f'Detected Color: {predicted_color}, Accuracy: {accuracy:.2f}%')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
