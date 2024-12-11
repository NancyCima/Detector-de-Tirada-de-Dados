import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

"""                            Descripción:
Este script fue hecho para calcular de forma dinamica los umbrales de valores HSV.
Mediante una GUI interactiva de OpenCV se obtiene una segmentacion mas visual y 
eficiente a la vez.
"""

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    """
    Definimos funcion para simplificar codigo de visualizaciones
    """
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

def subplot4(img1, img2, img3, img4, share = False, t1 = "", t2 = "", t3 = "", t4=""):
    """
    Recibe 4 imagenes y 4 titulos y las muestra en forma de subplot de 4x4
    """
    plt.figure()
    if share:
        ax = plt.subplot(141); imshow(img1, new_fig=False, title=t1)
        plt.subplot(142, sharex=ax, sharey=ax); imshow(img2, new_fig=False, title=t2)
        plt.subplot(143, sharex=ax, sharey=ax); imshow(img3, new_fig=False, title=t3)
        plt.subplot(144, sharex=ax, sharey=ax); imshow(img4, new_fig=False, title=t4)
        plt.show(block=False)
    else:
        plt.subplot(141); imshow(img1, new_fig=False, title=t1)
        plt.subplot(142); imshow(img2, new_fig=False, title=t2)
        plt.subplot(143); imshow(img3, new_fig=False, title=t3)
        plt.subplot(144); imshow(img4, new_fig=False, title=t4)
        plt.show(block=False)

def subplot3(img1, img2, img3, share = False, t1 = "", t2 = "", t3 = ""):
    """
    Recibe 3 imagenes y 3 titulos y las muestra en forma de subplot de 3x3
    """
    plt.figure()
    if share:
        ax = plt.subplot(131); imshow(img1, new_fig=False, title=t1)
        plt.subplot(132, sharex=ax, sharey=ax); imshow(img2, new_fig=False, title=t2)
        plt.subplot(133, sharex=ax, sharey=ax); imshow(img3, new_fig=False, title=t3)
        plt.show(block=False)
    else: 
        plt.subplot(131); imshow(img1, new_fig=False, title=t1)
        plt.subplot(132); imshow(img2, new_fig=False, title=t2)
        plt.subplot(133); imshow(img3, new_fig=False, title=t3)
        plt.show(block=False)

def guarda_frames(n_tirada, detalles = False, view = False):
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
        print("-" * 20)
        print("Detalles del video")
        print(" - Ancho:", width, "px")
        print(" - Alto:", height, "px")
        print(" - FPS:", fps)
        print("-" * 20)
    # Guardamos los frames en la carpeta correspondiente y mostramos los mismos
    frame_number = 0
    while (cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True:
            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
            if view:
                cv2.imshow('Frame', frame) 
            cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), frame) 
            frame_number += 1
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        else:  
            break
    cv2.destroyAllWindows()
    cap.release()
    print(frame_number, "Frames guardados exitosamente")

def mask_y_mask_inv_segHSV(img, h_min = 65, h_max = 85, s_min = 100, s_max = 255, view = False):
    """
    Recibe una imagen de una mesa de dados y calcula la mascara binaria de aplicar
    segmentacion de color verde y la mascara inversa de la misma
    """
    # Define a range for segmentation based on H and S values
    lims_inf = np.array([h_min, s_min, 50])
    lims_sup = np.array([h_max, s_max, 255])
    # Convert the image to HSV and create a mask
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Creo la mascara y normalizo a imagen binaria con 0 y 1
    mask = np.uint8(cv2.inRange(hsv_image, lims_inf, lims_sup) / 255)
    mask_inv = 1 - mask
    if view:
        subplot3(img, mask, mask_inv, "Imagen original", "Mascara segmentacion color verde", "Mascara inversa segmentacion color verde")
    return mask, mask_inv

def mask_y_mask_inv_segHSV2(img, h1_min = 65, h1_max = 85, s1_min = 100, s1_max = 255, h2_min = None, h2_max = None, s2_min = None, s2_max = None):
    """
    Recibe una imagen en formato RGB y calcula la mascara binaria de aplicar segmentacion de
    color especificado en formato HSV. Ademas devuelve la mascara inversa de la misma.
    Por default calcula la mascara binaria de segmentar por color verde
    """
    # Definimos los limites infereriores y superirores
    lims_inf = np.array([h1_min, s1_min, 50])
    lims_sup = np.array([h1_max, s1_max, 255])
    # Convertimos la imagen a formato HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if h2_min: # Si hay 2 intervalos para cada valor HSV (por ejemplo para segmentar el color rojo)
        lims2_inf = np.array([h2_min, s2_min, 50])
        lims2_sup = np.array([h2_max, s2_max, 255])
        # Calculamos las mascaras binarias con los limites dados
        mask1 = cv2.inRange(hsv_img, lims_inf, lims_sup)
        mask2 = cv2.inRange(hsv_img, lims2_inf, lims2_sup)
        # Combinamos ambas máscaras
        mask = cv2.bitwise_or(mask1, mask2)
    else: # Si hay un solo intervalo para cada valor HSV
        mask = cv2.inRange(hsv_img, lims_inf, lims_sup)
    # Normalizo la mascara a 0s y 1s y calculo la mascara inversa 
    mask = np.uint8(mask / 255) 
    mask_inv = 1 - mask
    return mask, mask_inv

def update_image(img):
    """
    Funcion para segmentar imagen por color mediante valores de H, S y V del formato HSV
    """
    titulo = "Segmentacion dinamica HSV"
    h_min = cv2.getTrackbarPos("Hue Min", titulo)
    h_max = cv2.getTrackbarPos("Hue Max", titulo)
    s_min = cv2.getTrackbarPos("Saturation Min", titulo)
    s_max = cv2.getTrackbarPos("Saturation Max", titulo)
    v_min = cv2.getTrackbarPos("Value Min", titulo)
    v_max = cv2.getTrackbarPos("Value Max", titulo)
    # Define a range for segmentation based on H, S, and V values
    lims_inf = np.array([h_min, s_min, v_min])
    lims_sup = np.array([h_max, s_max, v_max])
    # Convert the image to HSV and create a mask
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lims_inf, lims_sup)
    # Apply the mask to the original img
    segmentacion = cv2.bitwise_and(img, img, mask=mask)
    # Show the segmented result
    cv2.imshow(titulo, segmentacion)

def segmentacion_dinamica(img):
    """
    Recibe una imagen en formato RGB y crea la GUI interactiva para segmentar por colores
    en formato HSV
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Callback con la imagen específica
    def callback(_):
        update_image(img)
    # Convertimos la imagen a hsv para aplicar la segmentacion de colores
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Creamos la GUI interactiva
    titulo = "Segmentacion dinamica HSV"
    cv2.namedWindow(titulo)
    cv2.createTrackbar("Hue Min", titulo, 0, 179, callback)
    cv2.createTrackbar("Hue Max", titulo, 179, 179, callback)
    cv2.createTrackbar("Saturation Min", titulo, 0, 255, callback)
    cv2.createTrackbar("Saturation Max", titulo, 255, 255, callback)
    cv2.createTrackbar("Value Min", titulo, 0, 255, callback)
    cv2.createTrackbar("Value Max", titulo, 255, 255, callback)
    # Calculamos valores minimos y maximos para cada valor HSV de la imagen
    h_min = int(np.min(hsv_img[:, :, 0]))
    h_max = int(np.max(hsv_img[:, :, 0]))
    s_min = int(np.min(hsv_img[:, :, 1]))
    s_max = int(np.max(hsv_img[:, :, 1]))
    v_min = int(np.min(hsv_img[:, :, 2]))
    v_max = int(np.max(hsv_img[:, :, 2]))
    # Creamos los trackbars 
    cv2.setTrackbarPos("Hue Min", titulo, h_min)
    cv2.setTrackbarPos("Hue Max", titulo, h_max)
    cv2.setTrackbarPos("Saturation Min", titulo, s_min)
    cv2.setTrackbarPos("Saturation Max", titulo, s_max)
    cv2.setTrackbarPos("Value Min", titulo, v_min)
    cv2.setTrackbarPos("Value Max", titulo, v_max)
    # Mostramos la imagen original segmentada
    cv2.imshow(titulo, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == "__main__":
    # Guardamos los frames de la tirada de dados deseada
    n_tirada = 1
    guarda_frames(n_tirada, True, True)

    # Cargamos el frame que deseamos segmentar
    n_frame = 82 # Probar con frame 0, 20, 70, 100
    path = "frames/frame_" + str(n_frame) +".jpg"  
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Pasamos a formato RGB
    if img is None:
        print("Error: imagen inexistente")
        exit(1)
    imshow(img)

    # Segmentamos dinamicamente la image con la GUI interactiva
    segmentacion_dinamica(img)

    # A partir de la segmentacion dinamica anterior obtuvimos los siguientes umbrales:
    # Umbrales para color:

    # VERDE
        # Hmin = 70
        # Hmax = 85
        # Smin = 100
        # Smax = 255
        # Vmin = 50
        # Vmax = 255
    # Con la funcion queda:
    mask, mask_inv = mask_y_mask_inv_segHSV(img) # O especificando valores mask_y_mask_inv_segHSV(img, 65, 85, 100, 255)
    seg = cv2.bitwise_and(img, img, mask=mask)
    # Visualizamos el resultado
    subplot4(img,mask, mask_inv, seg,True, "Imagen original", "Mascara binaria color verde","Mascara binaria inversa color verde", "Segmentación")

    # ROJO
        # H1min = 0
        # H1max = 15
        # S1min = 150
        # S1max = 255
        # H2min = 165
        # H2max = 179
        # S1min = 25
        # S1max = 255
        # Vmin  = 50
        # Vmax  = 255

    # Con la funcion queda:
    mask, mask_inv = mask_y_mask_inv_segHSV2(img, 0, 15, 150, 255, 165, 179, 25, 255)
    seg = cv2.bitwise_and(img, img, mask=mask)
    # Visualizamos el resultado
    subplot4(img,mask, mask_inv, seg,True, "Imagen original", "Mascara binaria color rojo","Mascara binaria inversa color rojo", "Segmentación")


    # MEJORAMIENTO DE MASCARA BINARIA DADOS.
    mask, mask_inv = mask_y_mask_inv_segHSV(img)
    # subplot3(img,mask,mask_inv,True)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) 
    op = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    dil = cv2.dilate(op, kernel2)
    subplot3(img,op,dil)
    seg = cv2.bitwise_and(img, img, mask=dil)
    subplot4(img,mask,dil,seg,True)   