import cv2
import joblib
import numpy as np

# muat model decision tree dan scaler
dtree = joblib.load('dtree_model.pkl')
scaler = joblib.load('scaler.pkl')

# inisialisasi kamera
cap = cv2.VideoCapture(0)

# Variabel untuk menghitung akurasi
correct_predictions = 0
total_predictions = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height // 2, width // 2]

    # normalisasi pixel sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center])
    
    # prediksi warna
    color_pred = dtree.predict(pixel_center_scaled)[0]

    # Tentukan kotak dan warna teks berdasarkan prediksi
    if color_pred == 'red':
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 0, 255), 2)  # kotak merah
        cv2.putText(frame, f'color: Red', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif color_pred == 'green':
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)  # kotak hijau
        cv2.putText(frame, f'color: Green', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Menampilkan akurasi (di terminal dan pada gambar)
    accuracy = 0
    if color_pred == 'red':
        correct_predictions += 1
    elif color_pred == 'green':
        correct_predictions += 1
    
    total_predictions += 1
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100

    # Tampilkan akurasi di dekat kotak
    cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # tampilkan frame dengan kotak
    cv2.imshow('frame', frame)

    # menampilkan akurasi di terminal
    print(f'Accuracy: {accuracy:.2f}%')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
