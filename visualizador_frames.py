import cv2
import numpy as np
import os

"""                            Descripción:
Este script fue hecho para visualizar con mas detalle un video frame a frame.
Mediante una GUI interactiva de OpenCV se obtiene una visualizacion frame por frame
de manera que se pueden ver mejor los detalles de lo que sucede en el video.
Teclas para usar GUI:
    # Flechas izquierda y derecha para pasar de frame
    # Flecha hacia arriba para cambiar a modo de 2 frames(previo y actual)
    # Escape para salir
"""

def guarda_frames(n_tirada,detalles = False, view = False):
    """
    Recibe el numero de tirada de un video y guarda todos los frames
    del mismo en la carpeta frames.
    Además, se puede especificar si se desea ver los detalles del video y 
    el video en cuestion
    """
    # Creamos carpeta frames si no existe
    os.makedirs("frames", exist_ok = True)  
    # Capturamos el video en cuestion y calculamos
    cap = cv2.VideoCapture(r'inputs\tirada_' + str(n_tirada) + '.mp4')
    # Si se requiere mostramos detalles del video
    if detalles:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cant_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("-" * 20)
        print("Detalles del video")
        print(" - Ancho:", width, "px")
        print(" - Alto:", height, "px")
        print(" - FPS:", fps)
        print(" - Cantidad de frames:", cant_frames)
        print("-" * 20)
    # Guardamos los frames en la carpeta correspondiente y mostramos los mismos
    frame_number = 0
    while (cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True:
            frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
            if view:
                cv2.imshow('Frame', frame_show) 
            if view and cv2.waitKey(): 
                break
            cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), frame) 
            frame_number += 1
        else:  
            break
    cv2.destroyAllWindows()
    cap.release()
    print(frame_number, "Frames guardados exitosamente")
    return cant_frames

def cargar_frames(folder, cant_frames):
    """
    Carga frames con nombres predefinidos (frames/frame_X.jpg) y los guarda en una lista.
    """
    frames = []
    for i in range(0, cant_frames):
        filepath = f"{folder}/frame_{i}.jpg"  # Construir la ruta manualmente
        print(f"Procesando archivo: {filepath}")  # Depuración
        frame = cv2.imread(filepath)  # Cargar el frame
        if frame is not None:  # Verificar que se cargó correctamente
            frames.append(frame)
        else:
            print(f"Error al cargar la imagen: {filepath}")  # Mensaje si falla
    return frames

def visualizador_framesBGR(frames):
    """
    Recibe una lista de frames(BGR) y muestra frame a frame en una GUI interactiva.
    Teclas para usar GUI:
    # Flechas izquierda y derecha para pasar de frame
    # Flecha hacia arriba para cambiar a modo de 2 frames(previo y actual)
    # Escape para salir 
    """
    # Escalo porque los frames son muy grandes y no se ven bien
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    frames = [cv2.resize(frame, dsize=(int(width/3), int(height/3))) for frame in frames]
    index = 0  # Índice del frame actual
    total_frames = len(frames)
    mode = 1  # 1: Modo 1 frame, 2: Modo 2 frames
    while True:
        if mode == 1:  # Mostrar 1 frame
            frame = frames[index].copy()
            display_frame = frame
            text = f"Frame {index + 1}/{total_frames}"
            cv2.putText(display_frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        elif mode == 2 and index > 0:  # Mostrar 2 frames (anterior y actual)
            prev_frame = frames[index - 1].copy()
            curr_frame = frames[index].copy()
            # Crear espacio negro entre las imágenes
            height = max(prev_frame.shape[0], curr_frame.shape[0])
            black_separator = (np.ones((height, 100, 3), dtype=np.uint8)*255)  # Espacio negro de 10px
            prev_text = f"Frame {index}/{total_frames}"
            curr_text = f"Frame {index + 1}/{total_frames}"
            cv2.putText(prev_frame, prev_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(curr_frame, curr_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            display_frame = cv2.hconcat([prev_frame, black_separator, curr_frame])
        else:  # Si no hay frame anterior en modo 2
            frame = frames[index].copy()
            display_frame = frame
            text = f"Frame {index + 1}/{total_frames}"
            cv2.putText(display_frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # Agregar número del frame al título de la imagen
        text = f"Frame {index + 1}/{total_frames}"
        # Mostrar el frame
        cv2.imshow('Frame Viewer', display_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Salir con ESC
            break
        elif key == 83:  # Flecha derecha (avanzar)
            index = (index + 1) % total_frames
        elif key == 81:  # Flecha izquierda (retroceder)
            index = (index - 1 + total_frames) % total_frames
        elif key == 82:  # Flecha hacia arriba (cambiar modo)
            mode = 2 if mode == 1 else 1
    cv2.destroyAllWindows()

def visualizador_framesRGB(frames):
    """Recibe una lista con frames(RGB) y muestra frame a frame en una GUI interactiva.
    Teclas para usar GUI:
    # Flechas izquierda y derecha para pasar de frame
    # Flecha hacia arriba para cambiar a modo de 2 frames(previo y actual)
    # Escape para salir"""
    # Paso de RGB a BGR para usar imshow de cv2
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    # Escalo porque los frames son muy grandes y no se ven bien
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    frames = [cv2.resize(frame, dsize=(int(width/3), int(height/3))) for frame in frames]
    index = 0  # Índice del frame actual
    total_frames = len(frames)
    mode = 1  # 1: Modo 1 frame, 2: Modo 2 frames
    while True:
        if mode == 1:  # Mostrar 1 frame
            frame = frames[index].copy()
            display_frame = frame
            text = f"Frame {index + 1}/{total_frames}"
            cv2.putText(display_frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        elif mode == 2 and index > 0:  # Mostrar 2 frames (anterior y actual)
            prev_frame = frames[index - 1].copy()
            curr_frame = frames[index].copy()
            # Crear espacio negro entre las imágenes
            height = max(prev_frame.shape[0], curr_frame.shape[0])
            black_separator = (np.ones((height, 100, 3), dtype=np.uint8)*255)  # Espacio negro de 10px
            prev_text = f"Frame {index}/{total_frames}"
            curr_text = f"Frame {index + 1}/{total_frames}"
            cv2.putText(prev_frame, prev_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(curr_frame, curr_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            display_frame = cv2.hconcat([prev_frame, black_separator, curr_frame])
        else:  # Si no hay frame anterior en modo 2
            frame = frames[index].copy()
            display_frame = frame
            text = f"Frame {index + 1}/{total_frames}"
            cv2.putText(display_frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # Agregar número del frame al título de la imagen
        text = f"Frame {index + 1}/{total_frames}"
        # Mostrar el frame
        cv2.imshow('Frame Viewer', display_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Salir con ESC
            break
        elif key == 83:  # Flecha derecha (avanzar)
            index = (index + 1) % total_frames
        elif key == 81:  # Flecha izquierda (retroceder)
            index = (index - 1 + total_frames) % total_frames
        elif key == 82:  # Flecha hacia arriba (cambiar modo)
            mode = 2 if mode == 1 else 1
    cv2.destroyAllWindows()










if __name__ == "__main__":
    # Guardo frames de video que deseamos inspeccionar
    n_tirada = 4
    cant_frames = guarda_frames(n_tirada,True,True)

    # Llamada al visualizador
    frames = cargar_frames("frames", cant_frames)
    visualizador_framesBGR(frames)