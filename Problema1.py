import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
--------------------------------------------------------------------------------
                    DETECCION AUTOMATICA DE TIRADAS DE DADOS 
--------------------------------------------------------------------------------

Descripción:
Este script utiliza procesamiento de imágenes para detectar tiradas de dados 
en videos. Incluye funciones para cargar frames, procesarlos y detectar los 
dados con sus respectivos valores.
"""

def imshow(img, new_fig=True, title=None, color_img=False, blocking=True, colorbar=False, ticks=False):
    """Definimos funcion para simplificar codigo de visualizaciones"""
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

def subplot3(img1, img2, img3, share = False, t1 = "", t2 = "", t3 = ""):
    """Recibe 3 imagenes y 3 titulos y las muestra en forma de subplot de 3x3"""
    plt.figure()
    if share:
        ax = plt.subplot(131); imshow(img1, new_fig=False, title=t1)
        plt.subplot(132, sharex=ax, sharey=ax); imshow(img2, new_fig=False, title=t2)
        plt.subplot(133, sharex=ax, sharey=ax); imshow(img3, new_fig=False, title=t3)
        plt.show(block=True)
    else: 
        plt.subplot(131); imshow(img1, new_fig=False, title=t1)
        plt.subplot(132); imshow(img2, new_fig=False, title=t2)
        plt.subplot(133); imshow(img3, new_fig=False, title=t3)
        plt.show(block=True)

def subplot4(img1, img2, img3, img4, share = False, t1 = "", t2 = "", t3 = "", t4=""):
    """Recibe 4 imagenes y 4 titulos y las muestra en forma de subplot de 4x4"""
    plt.figure()
    if share:
        ax = plt.subplot(141); imshow(img1, new_fig=False, title=t1)
        plt.subplot(142, sharex=ax, sharey=ax); imshow(img2, new_fig=False, title=t2)
        plt.subplot(143, sharex=ax, sharey=ax); imshow(img3, new_fig=False, title=t3)
        plt.subplot(144, sharex=ax, sharey=ax); imshow(img4, new_fig=False, title=t4)
        plt.show(block=True)
    else:
        plt.subplot(141); imshow(img1, new_fig=False, title=t1)
        plt.subplot(142); imshow(img2, new_fig=False, title=t2)
        plt.subplot(143); imshow(img3, new_fig=False, title=t3)
        plt.subplot(144); imshow(img4, new_fig=False, title=t4)
        plt.show(block=True)


def carga_frames(path):
    """Funcion encargada de cargar todos los frames de un video dado"""
    # Verifico la existencia del archivo
    if path is None:
        print("Error: Archivo inexistente")
        return -1
    # Capturo el video del path
    cap = cv2.VideoCapture(path)  
    # Almaceno en una lista los frames que tengan dados en la mesa
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Pasamos a RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:  
            break
    return frames

# Segmentación y enmascaramiento
def mask_y_mask_inv_segHSV(img, h_min = 65, h_max = 85, s_min = 100, s_max = 255):
    """Recibe una imagen de una mesa de dados y calcula la mascara binaria de aplicar
    segmentacion de color verde y la mascara inversa de la misma"""
    # Define a range for segmentation based on H and S values
    lims_inf = np.array([h_min, s_min, 50])
    lims_sup = np.array([h_max, s_max, 255])
    # Convert the image to HSV and create a mask
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Creo la mascara y normalizo a imagen binaria con 0 y 1
    mask = np.uint8(cv2.inRange(hsv_image, lims_inf, lims_sup) / 255)
    mask_inv = 1 - mask
    return mask, mask_inv

# Filtrado de componentes conectadas
def filtrar_area(img_bin, th1 = 4500, th2 = 5500):
    """Recibe una imagen binaria, calcula las componentes conectadas y filtra las
    mismas por area entre [th1,th2]"""
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity, cv2.CV_32S)
    for i in range(num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < th1 or area > th2:
            img_bin[labels==i] = 0
            num_labels -= 1
    return num_labels

def filtrar_relAsp(img_bin, th1 = 0.8, th2 = 1.2):
    """Filtra componentes conectadas por relación de aspecto."""
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity, cv2.CV_32S)
    for i in range(num_labels):
        alto = stats[i, cv2.CC_STAT_HEIGHT]
        ancho = stats[i, cv2.CC_STAT_WIDTH]
        relAsp = alto / ancho
        if relAsp < th1 or relAsp > th2:
            img_bin[labels==i] = 0
            num_labels -= 1
    return num_labels

def dados_en_mesa(img):
    """Determina si la imagen tiene dados en la mesa. Con
    las siguientes condiciones:
        #Cantidad de pixels verdes mayor a umbral adecuado
        #Area de los dados entre umbral adecuado"""
    if img is None:
        print("Error: debe pasar una imagen")
        return -1
    # imshow(img) # Mostrar paso a paso
    mask, mask_inv = mask_y_mask_inv_segHSV(img, 65, 85, 100, 255)
    # imshow(mask) # Mostrar paso a paso
    cant_verde = mask.sum()
    # imshow(mask_inv) # Mostrar paso a paso
    # Calculamos componentes conectadas y filtramos por area para quedarnos solo con dados
    filtrar_area(mask_inv, 4400, 6000) # Trabaja por referencia
    # imshow(mask_inv) # Mostrar paso a paso
    # Calculamos nuevamente las componentes conectadas y verificamos tener mas de 1 y menos de 6
    # es decir, mas de 1 dado (por si algunos estan muy juntos o por si queda un dedo de la mano en la imagen)
    # y menor o igual a 5 (ya que son 5 dados)
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inv, connectivity, cv2.CV_32S)
    cant_dados = num_labels - 1
    min_dados = 1
    max_dados = 6 #Esto es por si hay un dedo en la imagen con el area de un dado lo tome igual
    CANT_VERDE_TH = 1600000 # Se puede ajustar un poco mas bajo para obtener algunos frames con partes de manos al agarrar los dados, por ejemplos 1650000. Default 1.7 mill
    cumple = cant_dados > min_dados and cant_dados <= max_dados and cant_verde > CANT_VERDE_TH
    # Si esto no basta, podria obtener el umbral de cantidad de pixels verdes con dados arriba
    # y agregarlo a la filtracion o condicion
    return cumple

# Filtrado de frames útiles
def filtrar_frames_dados_estaticos(frames):
    """Funcion encargada de filtrar los frames utiles de un video de tirada particular
    Se filtran solo los frames que cumplen con la funcion predicado "dados_en_mesa()"
    Luego se queda solo con la mitad superior de los frames para quedarse con los
    frames que tienen dados estaticos en la mesa."""
    # Almaceno en una lista los frames que tengan dados en la mesa
    frames_dados = []
    i_frame = 0
    cant_frames_dados = 0
    inicio = 0
    for frame in frames:
        if dados_en_mesa(frame):
            if cant_frames_dados == 0:
                inicio = i_frame
            frames_dados.append(frame)
            cant_frames_dados += 1
        i_frame += 1
    # Retorno la mitad superior -10 de la lista de frames con dados ya que es donde estan estaticos
    largo = cant_frames_dados
    k = 0
    indice_inf = inicio + largo//2 - k
    indice_sup = inicio + cant_frames_dados - k
    frames_dados_estaticos = frames_dados[largo//2 - k:]
    return frames_dados_estaticos,indice_inf, indice_sup


def detecta_dados(frames_dados_estaticos):
    """Recibe los frames con dados estaticos en la mesa, detecta los dados y los
    identifica con un bounding box. Para esto toma un frame de referencia, en especifico
    el frame del medio de la lista, y arma los bounding box a partir del mismo para
    todas los demas frames con dados estaticos ya que estaran en el mismo lugar aprox.
    Devuelve todos los dados identificados y la mascara de los bounding box"""
    dados_identificados = [frame.copy() for frame in frames_dados_estaticos]
    indice_ref = len(frames_dados_estaticos) // 2
    frame_ref = frames_dados_estaticos[indice_ref]
    frame_ref_c = frame_ref.copy()
    frame_ref_gray = cv2.cvtColor(frame_ref_c, cv2.COLOR_RGB2GRAY)
    mask_box = np.zeros_like(frame_ref_gray)
    mask, mask_inv = mask_y_mask_inv_segHSV(frame_ref_c, 60, 85, 100, 255) 
    if filtrar_area(mask_inv, 4400, 6200) != 5: #Si todavia no se consiguieron los 5 dados filtro por relacion de aspecto
        filtrar_relAsp(mask_inv, 0.8, 1.2)
    subplot3(frame_ref_c, mask, mask_inv, True) # Mostrar paso a paso !
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_inv, connectivity, cv2.CV_32S)
    cant_dados = num_labels - 1
    for st in stats[1:]:
            x1, y1, x2, y2 = st[0], st[1], st[0]+st[2],  st[1]+st[3]
            dado = frame_ref_c[y1:y2, x1:x2].copy()
            dado_gray = cv2.cvtColor(dado, cv2.COLOR_RGB2GRAY)
            dado_gray_blur = cv2.medianBlur(dado_gray,7)
            _,mask = cv2.threshold(dado_gray_blur, 170, 255,cv2.THRESH_BINARY)
            # subplot4(dado, dado_gray, dado_gray_blur, mask, True) # Mostrar paso a paso !
            connectivity = 8
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S) #Detectamos circulos con este metodo porque no funciono houghcircles probando con diferentes parametros, ademas es mas eficiente
            numero_dado = num_labels - 1
            cv2.rectangle(mask_box, (x1, y1), (x2, y2), 255, thickness=3)
            k1 = 19
            k2 = 5
            pos = (x1 + (x2-x1)//2 - k1, y1-k2)
            text = str(numero_dado)
            cv2.putText(mask_box, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    frame_ref_c[mask_box == 255] = [0,0,255]
    subplot3(frame_ref,mask_box,frame_ref_c, True)
    imshow(frame_ref_c)
    # Bounding box a todos los frames
    for frame in dados_identificados:
        frame[mask_box == 255] = [0,0,255]
    return dados_identificados, mask_box
    

def grabar_video(frames_out, tirada):
    """Recibe frames con dados estaticos en la mesa y detecta los dados
    Para la deteccion se usa el frame del medio
    Para la identificacion se usan todos los frames"""
    cap = cv2.VideoCapture(r'inputs\tirada_' + str(tirada) + '.mp4')  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    out = cv2.VideoWriter(r'outputs\output_tirada_' + str(tirada) +'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    for frame in frames_out:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
        # frame_show = cv2.cvtColor(frame_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame",frame_show)
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            cv2.destroyAllWindows()
        out.write(frame)   
    cap.release() 
    out.release() 
    cv2.destroyAllWindows() 



tirada = 3
# 1 # Se cargan todos los frames en una lista
frames = carga_frames(r'inputs\tirada_' + str(tirada) + '.mp4')
# len(frames)
copy_frames = [frame.copy() for frame in frames] # Se crea una copia para no modificar las imagenes originales con bounding box

# 2 # Se obtienen los frames con dados y los indices con respecto la lista frames.
    # Estos indices nos serviran luego para grabar el video final, modificando los 
    # frames que solo esten dentro de esos indices y dejando el resto iguales
frames_dados_estaticos, indice_inf, indice_sup = filtrar_frames_dados_estaticos(copy_frames)
# Vemos los primeros frames hasta el ultimo con dados estaticos
# subplot4(frames_dados_estaticos[0], frames_dados_estaticos[1], frames_dados_estaticos[-2], frames_dados_estaticos[-1])

# 3 # Identificamos los dados con bounding box en todos los frames para poder grabar el video
dados_identificados, mask_box = detecta_dados(frames_dados_estaticos)
# subplot4(dados_identificados[0], dados_identificados[1], dados_identificados[-2], dados_identificados[-1])


# 4 # Agregamos los frames con los bounding box a todos los frames en su posicion correspondiente
copy_frames[indice_inf:indice_sup] = dados_identificados
frames_out = copy_frames

# 5 # Grabamos el video con los frames resultantes
grabar_video(frames_out, tirada)