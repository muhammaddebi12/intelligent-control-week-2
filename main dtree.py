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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi frame ke HSV untuk deteksi warna yang lebih akurat
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Ambil ukuran frame
    height, width, _ = frame.shape
    
    # Temukan area dengan warna dominan
    blurred = cv2.GaussianBlur(hsv_frame, (15, 15), 0)
    pixel_center = blurred[height // 2, width // 2]

    # Normalisasi pixel sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center])
    
    # Prediksi warna
    color_pred = dtree.predict(pixel_center_scaled)[0]

    # Tentukan warna berdasarkan prediksi
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0)
    }
    box_color = color_map.get(color_pred, (255, 255, 255))  # Default putih jika tidak dikenali
    
    # Gunakan deteksi kontur untuk menemukan area warna dominan
    mask = cv2.inRange(hsv_frame, np.array([0, 120, 70]), np.array([10, 255, 255])) if color_pred == 'red' else \
           cv2.inRange(hsv_frame, np.array([36, 25, 25]), np.array([86, 255, 255])) if color_pred == 'green' else \
           cv2.inRange(hsv_frame, np.array([100, 150, 0]), np.array([140, 255, 255])) if color_pred == 'blue' else None
    
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, f'Color: {color_pred.capitalize()}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
    
    # Perhitungan akurasi
    if color_pred in color_map:
        correct_predictions += 1
    
    total_predictions += 1
    accuracy = max(0, min((correct_predictions / total_predictions) * 100, 100))  # Jaga dalam rentang 0-100%
    
    # Tampilkan akurasi
    cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Tampilkan frame
    cv2.imshow('Color Detection', frame)
    print(f'Accuracy: {accuracy:.2f}%')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
