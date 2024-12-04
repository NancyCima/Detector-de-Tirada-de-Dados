# Importamos librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import cv2

def detectar_movimiento(video_path, umbral_movimiento=5000, escala=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("No se pudo abrir el video.")

    ret, frame_actual = cap.read()
    if not ret:
        raise IOError("No se pudo leer el primer frame.")
    
    frame_actual = cv2.resize(frame_actual, None, fx=escala, fy=escala)
    frame_actual_gray = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    frame_number = 0

    while cap.isOpened():
        ret, frame_siguiente = cap.read()
        if not ret:
            break
        
        frame_siguiente = cv2.resize(frame_siguiente, (frame_actual.shape[1], frame_actual.shape[0]))
        frame_siguiente_gray = cv2.cvtColor(frame_siguiente, cv2.COLOR_BGR2GRAY)

        diferencia = cv2.absdiff(frame_actual_gray, frame_siguiente_gray)
        _, diferencia_bin = cv2.threshold(diferencia, 30, 255, cv2.THRESH_BINARY)

        num_pixeles_movimiento = np.sum(diferencia_bin > 0)
        if num_pixeles_movimiento < umbral_movimiento and frame_number > 50:
            cap.release()
            return frame_actual
        
        frame_actual_gray = frame_siguiente_gray
        frame_number += 1

    cap.release()
    raise ValueError("No se detectó el cese de movimiento en el video.")

def extraer_area_interes(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fondo = frame[30:200, 30:200]
    fondo_hsv = cv2.cvtColor(fondo, cv2.COLOR_BGR2HSV)

    colores, counts = np.unique(fondo_hsv.reshape(-1, 3), axis=0, return_counts=True)
    color_fondo = colores[np.argmax(counts)]

    delta_bg = np.array([10, 150, 150])
    lower_lim = np.clip(color_fondo - delta_bg, 0, 255)
    upper_lim = np.clip(color_fondo + delta_bg, 0, 255)

    mask = cv2.inRange(frame_hsv, lower_lim, upper_lim)
    return mask

def detectar_dados(mask, frame):
    # Aplicar limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Detectar componentes conectadas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    dados = []
    for i in range(1, num_labels):  # Saltar el fondo
        x, y, w, h, area = stats[i]
        if area > 600:  # Filtrar por tamaño mínimo
            dado = frame[y:y+h, x:x+w]
            dados.append(dado)
    
    return dados

def contar_puntos(dado):
    dado_gray = cv2.cvtColor(dado, cv2.COLOR_BGR2GRAY)
    dado_blur = cv2.GaussianBlur(dado_gray, (5, 5), 0)
    circles = cv2.HoughCircles(dado_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=15, minRadius=5, maxRadius=15)
    return len(circles[0]) if circles is not None else 0

# Procesar cada video
for i in range(1, 5):
    video_path = r'videos\tirada_' + str(i) + '.mp4'
    print("Procesando: tirada_" + str(i))
    frame_final = detectar_movimiento(video_path)
    mask = extraer_area_interes(frame_final)
    dados = detectar_dados(mask, frame_final)

    for idx, dado in enumerate(dados):
        puntos = contar_puntos(dado)
        print(f"Dado {idx+1}: {puntos} puntos.")
