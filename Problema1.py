import numpy as np
import matplotlib.pyplot as plt
import cv2
from segmentacionHSV import mask_y_mask_inv_segHSV
from visualizador_frames import visualizador_framesRGB

"""
--------------------------------------------------------------------------------
                    DETECCION AUTOMATICA DE TIRADAS DE DADOS 
--------------------------------------------------------------------------------

Este codigo permite el procesamiento de videos con tiradas de dados mediante tecnicas
de procesamiento de imagenes con el objetivo de identificar el numero de cada dado
y modificar el video para visualizar el resultado con bounding box.
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


def subplot3(img1, img2, img3, t1 = "", t2 = "", t3 = ""):
    """
    Recibe 3 imagenes y 3 titulos y las muestra en forma de subplot de 3x3
    """
    plt.figure()
    ax = plt.subplot(131); imshow(img1, new_fig=False, title=t1)
    plt.subplot(132, sharex=ax, sharey=ax); imshow(img2, new_fig=False, title=t2)
    plt.subplot(133, sharex=ax, sharey=ax); imshow(img3, new_fig=False, title=t3)
    plt.show(block=False)


def subplot4(img1, img2, img3, img4, t1 = "", t2 = "", t3 = "", t4=""):
    """
    Recibe 4 imagenes y 4 titulos y las muestra en forma de subplot de 4x4
    """
    plt.figure()
    ax = plt.subplot(141); imshow(img1, new_fig=False, title=t1)
    plt.subplot(142, sharex=ax, sharey=ax); imshow(img2, new_fig=False, title=t2)
    plt.subplot(143, sharex=ax, sharey=ax); imshow(img3, new_fig=False, title=t3)
    plt.subplot(144, sharex=ax, sharey=ax); imshow(img4, new_fig=False, title=t4)
    plt.show(block=False)

def carga_frames(n_tirada, view = False, detalles = False):
    """
    Funcion encargada de cargar todos los frames de un video dado
    """
    # Verifico la existencia del archivo
    if path is None:
        print("Error: Archivo inexistente")
        return -1
    # Capturo el video del path
    cap = cv2.VideoCapture(path)  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cant_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    if detalles:
        print("-" * 20)
        print("Detalles del video")
        print(" - Ancho:", width, "px")
        print(" - Alto:", height, "px")
        print(" - FPS:", fps)
        print(" - Cantidad de frames:", cant_frames)
        print("-" * 20)
    # Almaceno en una lista los frames que tengan dados en la mesa
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Si corresponde muestro el video
            if view:
                frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
                cv2.imshow('Frame', frame_show)
            # Pasamos a RGB y guardamos en la lista
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        else:  
            break
    cv2.destroyAllWindows()
    print("Se cargaron ", cant_frames, "frames exitosamente")
    return frames

# Cargamos frames para ir mostrando el funcionamiento de las funciones
# n_tirada = 1
# path = r'inputs\tirada_' + str(n_tirada) + '.mp4'
# frames = carga_frames(path, detalles = True)

def filtrar_area(img_bin, th1 = 4300, th2 = 6300, view = False):
    """
    Recibe una imagen binaria, calcula las componentes conectadas y filtra las
    mismas por area entre [th1,th2]
    """
    img_bin_out = img_bin.copy()
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin_out, connectivity, cv2.CV_32S)
    cont = 0
    for i in range(num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < th1 or area > th2:
            cont += 1
            img_bin_out[labels==i] = 0
            num_labels -= 1
    if view:
        imshow(img_bin_out, title="Filtrado de area entre [" + str(th1) + "," + str(th2) + "]")
    return img_bin_out, num_labels

def filtrar_relAsp(img_bin, th1 = 0.7, th2 = 1.3, view = False):
    """
    Recibe una imagen binaria, calcula las componentes conectadas y filtra las
    mismas por relacion de aspecto entre [th1,th2]"""
    img_bin_out = img_bin.copy()
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin_out, connectivity, cv2.CV_32S)
    cont = 0
    for i in range(num_labels):
        alto = stats[i, cv2.CC_STAT_HEIGHT]
        ancho = stats[i, cv2.CC_STAT_WIDTH]
        relAsp = alto / ancho
        if relAsp < th1 or relAsp > th2:
            cont += 1
            img_bin_out[labels==i] = 0
            num_labels -= 1
    if view:
        imshow(img_bin_out, title="Filtrado de relacion de aspecto entre [" + str(th1) + "," + str(th2) + "]")
    return img_bin_out, num_labels

# Para ir probando funciones anteriores
# i = 87  # Probar con 0, 70, 87
# img = frames[i] 
# imshow(img, title="Frame " + str(i))
# mask, mask_inv = mask_y_mask_inv_segHSV(img) # Por default segmenta por color verde
# mask_dados, cant_dados = filtrar_area(mask_inv, 4300, 6300,  view = True)
# mask_dados, cant_dados = filtrar_relAsp(mask_dados, 0.7, 1.3,  view = True)
# seg = cv2.bitwise_and(img, img, mask=mask_dados)
# subplot3(img, mask_inv, mask_dados, "Imagen original", "Mascara inversa color verde", "Mascara inversa + filtrado")
# cant_dados

def dados_en_mesa(img, views = False):
    """
    Determina si la imagen tiene los dados estaticos en la mesa. Con
    las siguientes condiciones:
        #Cantidad de pixels verdes mayor a umbral adecuado
        #Tengo 5 dados
    """
    if img is None:
        print("Error: debe pasar una imagen")
        return -1
    # Calculamos las mascaras a partir de la segmentacion de color verde
    mask, mask_inv = mask_y_mask_inv_segHSV(img, 65, 85, 100, 255, views)
    cant_verde = mask.sum()
    # Calculamos componentes conectadas y filtramos por area para quedarnos solo con dados
    mask_dados, cant_dados = filtrar_area(mask_inv, 4300, 6300)
    if cant_dados != 5: # Si no consegui 5 dados filtro por relacion de aspecto
        mask_dados, cant_dados = filtrar_relAsp(mask_dados, 0.7, 1.3)
    if views:
        seg = cv2.bitwise_and(img, img, mask=mask_dados)
        subplot3(img, mask_dados, seg,"Imagen original", "Mascara inversa + filtrado", "Segmentacion dados")
    # Calculamos nuevamente las componentes conectadas y verificamos tener mas de 5 y menos de 6
    # es decir, mas de 1 dado (por si algunos estan muy juntos o por si queda un dedo de la mano en la imagen)
    # y menor o igual a 5 (ya que son 5 dados)
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dados, connectivity, cv2.CV_32S)
    cant_dados = num_labels - 1
    min_dados = 5
    max_dados = 6 #Esto es por si hay un dedo en la imagen con el area de un dado lo tome igual
    CANT_VERDE_TH = 1600000 # Se puede ajustar un poco mas bajo para obtener algunos frames con partes de manos al agarrar los dados, por ejemplos 1650000. Default 1.7 mill
    cumple = cant_dados >= min_dados and cant_dados <= max_dados and cant_verde > CANT_VERDE_TH
    # Si esto no basta, podria obtener el umbral de cantidad de pixels verdes con dados arriba
    # y agregarlo a la filtracion o condicion
    return cumple, mask_dados

# Para ir probando funciones anteriores
# i1, i2, i3 = 0, 70, 87
# img1 = frames[i1] 
# img2 = frames[i2]
# img3 = frames[i3]
# subplot3(img1, img2, img3, "Frame " + str(i1), "Frame " + str(i2), "Frame " + str(i3))
# dados_en_mesa(img1, True)
# dados_en_mesa(img2, True)
# dados_en_mesa(img3, True)

def filtrar_frames_dados_estaticos(frames, view = False):
    """
    Funcion encargada de filtrar los frames utiles de un video de tirada particular
    Se filtran solo los frames que cumplen con la funcion predicado "dados_en_mesa()"
    Luego, evalua la similitud entre frames consecuitvos para quedarse con los
    frames que tienen dados estaticos en la mesa.
    """
    # Almaceno en una lista los frames que tengan dados en la mesa
    cant_frames_dados = 0
    inicio = 0
    indice_estatico = 0 # Indice de cuando se detienen los dados
    th_similitud = 0.9998
    mask_prev = np.uint8(np.zeros((frames[0].shape[0],frames[0].shape[1])))
    similitud_max = 0 # Valor minimo de similitud
    ###
    for i, frame in enumerate(frames):
        cumple, mask_dados = dados_en_mesa(frame)
        if cumple:
            cumple, mask_dados = dados_en_mesa(frame)
            # Comparar las imágenes para ver dónde son iguales
            resultado = cv2.compare(mask_prev, mask_dados, cv2.CMP_EQ)
            # Contar el número de píxeles iguales (donde el resultado es 255)
            numero_de_pixeles_iguales = np.sum(resultado == 255)
            # Calcular la similitud como un porcentaje
            similitud = (numero_de_pixeles_iguales / resultado.size)
            # print(similitud)
            if similitud > similitud_max and similitud_max < th_similitud:
                # print(i, similitud)
                indice_estatico = i - 1 #El frame previo es igual al actual, por lo tanto estan quietos
                similitud_max = similitud
            mask_prev = mask_dados
            if cant_frames_dados == 0:
                inicio = i
            cant_frames_dados += 1
    # Retorno la mitad superior de la lista de frames con dados ya que es donde estan estaticos
    indice_inf = indice_estatico
    indice_sup = inicio + cant_frames_dados
    frames_dados_estaticos = frames[indice_inf:indice_sup]
    if view:
        visualizador_framesRGB(frames_dados_estaticos)
    return frames_dados_estaticos,indice_inf, indice_sup


def detecta_dados(frames_dados_estaticos, views = False):
    """
    Recibe los frames con dados estaticos en la mesa, detecta los dados y los
    identifica con un bounding box. Para esto toma un frame de referencia, en especifico
    el frame del medio de la lista, y arma los bounding box a partir del mismo para
    todas los demas frames con dados estaticos ya que estaran en el mismo lugar aprox.
    Devuelve todos los dados identificados y la mascara de los bounding box
    """
    # Cargamos los frames y obtenemos un frame de referencia para hacer los bounding box
    dados_identificados = [frame.copy() for frame in frames_dados_estaticos]
    indice_ref = len(frames_dados_estaticos) // 2 #Tomamos el frame del medio para mejor precision
    frame_ref = frames_dados_estaticos[indice_ref]
    frame_ref_c = frame_ref.copy()
    # Calculamos las mascaras a partir de la segmentacion de color verde
    mask, mask_inv = mask_y_mask_inv_segHSV(frame_ref, 65, 85, 100, 255, views)
    # Filtramos por area y relacion de aspecto si hace falta para quedarnos solo con los dados
    mask_dados, cant_dados = filtrar_area(mask_inv, 4300, 6300)
    if cant_dados != 5: # Si no consegui 5 dados filtro por relacion de aspecto
        mask_dados, cant_dados = filtrar_relAsp(mask_dados, 0.7, 1.3)
    frame_ref_gray = cv2.cvtColor(frame_ref, cv2.COLOR_RGB2GRAY)
    mask_box = np.zeros_like(frame_ref_gray)
    # subplot3(frame_ref, mask_dados_ref, True) # Mostrar paso a paso !
    # Calculamos componentes conectadas en la mascara del frame de referencia dado
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dados, connectivity, cv2.CV_32S)
    cant_dados = num_labels - 1
    total = 0
    for st in stats[1:]:
        x1, y1, x2, y2 = st[0], st[1], st[0]+st[2],  st[1]+st[3]
        dado = frame_ref[y1:y2, x1:x2].copy()
        dado_gray = cv2.cvtColor(dado, cv2.COLOR_RGB2GRAY)
        dado_gray_blur = cv2.medianBlur(dado_gray,7)
        _,mask = cv2.threshold(dado_gray_blur, 170, 255,cv2.THRESH_BINARY)
        # subplot4(dado, dado_gray, dado_gray_blur, mask, True) # Mostrar paso a paso !
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S) #Detectamos circulos con este metodo porque no funciono houghcircles probando con diferentes parametros, ademas es mas eficiente
        numero_dado = num_labels - 1
        total += numero_dado
        cv2.rectangle(mask_box, (x1, y1), (x2, y2), 255, thickness=3)
        k1 = 19
        k2 = 5
        pos = (x1 + (x2-x1)//2 - k1, y1-k2)
        text = str(numero_dado)
        cv2.putText(mask_box, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    pos = (frame_ref.shape[1] // 2 - 175, 120)
    text = "Resultado: " + str(total)
    cv2.putText(mask_box, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    frame_ref[mask_box == 255] = [0,0,255]
    if views:
        mask_dados_box = mask_box + mask_dados * 255
        subplot4(frame_ref_c,mask_dados,mask_dados_box,frame_ref, "Frame de referencia", "Mascara dados", "Mascara dados + bounding box", "Resultado")
    # Bounding box a todos los frames
    for frame in dados_identificados:
        frame[mask_box == 255] = [0,0,255]
    if views:
        visualizador_framesRGB(dados_identificados)
    return dados_identificados
    

def grabar_video(frames_out, m_tirada, guardar = False):
    """
    Recibe frames con dados estaticos en la mesa y detecta los dados
    Para la deteccion se usa el frame del medio
    Para la identificacion se usan todos los frames
    """
    cap = cv2.VideoCapture(r'inputs\tirada_' + str(m_tirada) + '.mp4')  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if guardar:
        out = cv2.VideoWriter(r'outputs\output_tirada_' + str(m_tirada) +'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    for frame in frames_out:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
        cv2.imshow("frame",frame_show)
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            cv2.destroyAllWindows()
        if guardar:
            out.write(frame)   
    cap.release()
    if guardar:
        out.release() 
    cv2.destroyAllWindows() 


# 1 # Se cargan todos los frames en una lista

# 1.1 # Seleccionamos el numero de tirada a procesar
n_tirada = 3
path = r'inputs\tirada_' + str(n_tirada) + '.mp4'

# 1.2 # Cargamos los frames y mostramos los detalles del video
frames = carga_frames(path, view = False , detalles = True)

# 1.3 # Creamos una copia de los frames para no alterarlos durante el procesamiento
copy_frames = [frame.copy() for frame in frames]


# 2 # Se obtiene una sublista con frames que tienen dados estaticos, ademas,
# se obtiene:
    # indice_inf : indice inferior de la lista de frames donde los dados se detienen aprox
    # indice_sup : indice superior de la lista de frames donde los dados son levantados aprox
# Los indices nos serviran para armar el video, haciendo slicing de la lista original con la modificada(bounding box)
# La mascara sirve para hacer los bounding box para identificar cada dado

frames_dados_estaticos, indice_inf, indice_sup = filtrar_frames_dados_estaticos(copy_frames, view=False)
print("Indice donde los dados se detienen: ", indice_inf)
print("Indice donde se juntan los dados: ", indice_sup)

# 3 # Identificamos los dados con bounding box en todos los frames para poder grabar el video

dados_identificados = detecta_dados(frames_dados_estaticos, views = False)


# 4 # Agregamos los frames modificado (con los bounding box) a todos los frames 
# en su posicion correspondiente

copy_frames[indice_inf:indice_sup] = dados_identificados #A indice sup no hace falta sumarle 1 ya esta bien asi es el ultimo indice + 1
frames_out = copy_frames
#visualizador_framesRGB(frames_out)

# 5 # Grabamos el video con los frames resultantes
grabar_video(frames_out, n_tirada, guardar=True)
