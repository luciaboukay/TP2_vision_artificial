import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
import re
import open3d as o3d


def generar_puntos_objeto(checkerboard, square_size):
    """
    Genera la grilla de puntos 3D del patrón de checkerboard en el plano Z=0.

    Args:
        checkerboard (tuple[int, int]): Cantidad de esquinas internas (cols, rows) del patrón.
        square_size (float): Tamaño de cada cuadrado del patrón en unidades reales (p. ej., metros).

    Devuelve:
        np.ndarray: Arreglo (N, 3) con las coordenadas 3D de cada esquina en el plano Z=0.
    """
    # crear malla regular de puntos en el plano Z=0
    objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)  # generar coordenadas (x, y)
    objp *= square_size  # escalar por tamaño real del cuadrado
    return objp


def calibrar_stereo_checkerboard(base_dir, checkerboard, square_size):
    """
    Calibra un sistema estéreo a partir de imágenes de un checkerboard y guarda los parámetros obtenidos.

    Args:
        base_dir (str): Directorio base con las imágenes "calib_left*.jpg" y "calib_right*.jpg" y salida del pickle.
        checkerboard (tuple[int, int]): Cantidad de esquinas internas (cols, rows) del patrón.
        square_size (float): Tamaño real del cuadrado del checkerboard.

    Devuelve:
        dict: Diccionario con intrínsecas y coeficientes de distorsión de ambas cámaras, matrices R, T, E, F y el image_size.
    """
    # buscar y ordenar rutas de imágenes left/right
    left_images  = sorted(glob.glob(os.path.join(base_dir, "calib_left*.jpg")))
    right_images = sorted(glob.glob(os.path.join(base_dir, "calib_right*.jpg")))
    # validar cantidad de pares disponibles
    assert len(left_images) == len(right_images) and len(left_images) > 0, \
        "Debe haber el mismo número de imágenes left/right y al menos una pareja."

    # definir criterios de subpíxel y flags de detección
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    # preparar listas de correspondencias 3D-2D
    objp = generar_puntos_objeto(checkerboard, square_size)  # generar puntos 3D del patrón
    objpoints, imgpointsL, imgpointsR = [], [], []  # inicializar acumuladores

    # recorrer parejas y detectar esquinas del patrón
    for lp, rp in zip(left_images, right_images):
        imgL = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)  # leer imagen izquierda
        imgR = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)  # leer imagen derecha
        if imgL is None or imgR is None: 
            continue  # omitir pareja si no se puede leer

        retL, cornersL = cv2.findChessboardCorners(imgL, checkerboard, flags=cb_flags)  # detectar esquinas L
        retR, cornersR = cv2.findChessboardCorners(imgR, checkerboard, flags=cb_flags)  # detectar esquinas R
        if not (retL and retR): 
            continue  # omitir pareja si no se detecta el patrón en ambos

        cornersL = cv2.cornerSubPix(imgL, cornersL, (11,11), (-1,-1), criteria)  # refinar esquinas L
        cornersR = cv2.cornerSubPix(imgR, cornersR, (11,11), (-1,-1), criteria)  # refinar esquinas R

        objpoints.append(objp)          # acumular puntos 3D
        imgpointsL.append(cornersL)     # acumular proyecciones L
        imgpointsR.append(cornersR)     # acumular proyecciones R

    print(f"Pares válidos: {len(objpoints)}")  # informar cantidad de parejas útiles
    assert len(objpoints) > 0, "No se detectó el patrón en ninguna pareja."  # validar tener datos

    image_size = (imgL.shape[1], imgL.shape[0])  # obtener tamaño de imagen (w, h)

    # calibrar cámaras individuales
    rmsL, K1, d1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)  # calibrar izquierda
    rmsR, K2, d2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)  # calibrar derecha

    # calibrar estéreo manteniendo intrínsecas fijas
    flags = cv2.CALIB_FIX_INTRINSIC  # fijar intrínsecas
    criteria_st = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)  # definir criterio estéreo
    rmsS, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(  # ejecutar calibración estéreo
        objpoints, imgpointsL, imgpointsR,
        K1, d1, K2, d2, image_size,
        criteria=criteria_st, flags=flags
    )

    print(f"RMS left: {rmsL:.4f} | RMS right: {rmsR:.4f} | RMS stereo: {rmsS:.4f}")  # reportar errores RMS

    # empaquetar parámetros en diccionario de calibración
    calib = {
        "left_K": K1, "left_dist": d1,
        "right_K": K2, "right_dist": d2,
        "R": R, "T": T, "E": E, "F": F,
        "image_size": image_size
    }
    os.makedirs(base_dir, exist_ok=True)  # asegurar directorio de salida
    with open(os.path.join(base_dir, "stereo_calibration.pkl"), "wb") as f:
        pickle.dump(calib, f)  # guardar calibración en pickle

    print("Guardado: stereo_calibration.pkl")  # informar ruta de guardado
    return calib


def cargar_calibracion_stereo(calibration_path):
    """
    Carga la calibración estéreo desde un archivo pickle.

    Args:
        calibration_path (str): Ruta al archivo de calibración (stereo_calibration.pkl).

    Devuelve:
        dict: Diccionario con las matrices de calibración y parámetros asociados.
    """
    # validar existencia del archivo de calibración
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de calibración en: {calibration_path}. "
            "Ejecutar la función de calibración o ajustar 'base_dir' al directorio correcto."
        )

    # abrir y deserializar pickle
    with open(calibration_path, "rb") as file:
        calibration_data = pickle.load(file)  # cargar diccionario de calibración

    return calibration_data


def graficar_matrices_calibracion(calibracion):
    """
    Grafica las matrices y vectores clave de la calibración estéreo en una grilla 2x4 con valores y barras de color.

    Args:
        calibracion (dict): Diccionario con las matrices de calibración esperadas en las claves:
            'left_K', 'left_dist', 'right_K', 'right_dist', 'R', 'T', 'E', 'F'.

    Devuelve:
        None: Muestra la figura en pantalla sin retornar un valor.
    """
    def plot_matriz(ax, nombre, matriz):
        # renderizar heatmap de la matriz
        im = ax.imshow(matriz, cmap="viridis", interpolation="nearest")
        n_rows, n_cols = matriz.shape  # obtener forma
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j, i, f"{matriz[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if matriz[i, j] < (matriz.max() / 2) else "black",
                    fontsize=8  # fijar tamaño de texto
                )
        ax.set_title(nombre, fontsize=11, weight="bold")  # asignar título
        ax.set_xlabel("Columnas")  # etiquetar eje x
        ax.set_ylabel("Filas")     # etiquetar eje y
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # ocultar ticks
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # agregar barra de color por subgráfico

    # construir diccionario de matrices a graficar
    matrices = {
        "K1": calibracion["left_K"],
        "d1": calibracion["left_dist"],
        "K2": calibracion["right_K"],
        "d2": calibracion["right_dist"],
        "R":  calibracion["R"],
        "T":  calibracion["T"],
        "E":  calibracion["E"],
        "F":  calibracion["F"]
    }

    # crear figura y ejes
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # crear lienzo
    fig.suptitle("Parámetros de Calibración Estéreo", fontsize=14, weight="bold")  # asignar título global

    # iterar matrices y graficar
    for ax, (nombre, matriz) in zip(axes.flat, matrices.items()):
        plot_matriz(ax, nombre, matriz)  # graficar cada matriz

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # ajustar layout
    plt.show()  # mostrar figura



def inferir_image_size(calibracion, carpeta_busqueda, patron_left="calib_left*.jpg"):
    """
    Infere el tamaño de imagen (width, height) a usar en rectificación/calibración.

    Usa la clave 'image_size' del pickle si está disponible; en caso contrario, lo infiere leyendo
    una imagen izquierda que coincida con el patrón indicado.

    Args:
        calibracion (dict): Diccionario de calibración que puede incluir 'image_size'.
        carpeta_busqueda (str): Ruta donde buscar imágenes de la cámara izquierda.
        patron_left (str): Patrón glob para localizar imágenes izquierdas. Por defecto, "calib_left*.jpg".

    Devuelve:
        tuple[int, int]: Par (width, height) en píxeles.
    """
    # usar image_size del pickle si está presente
    if "image_size" in calibracion:
        return tuple(calibracion["image_size"])  # devolver tamaño almacenado
    # buscar una muestra izquierda para inferir tamaño
    left_samples = sorted(glob.glob(os.path.join(carpeta_busqueda, patron_left)))  # listar candidatos
    if not left_samples:
        raise FileNotFoundError("No se encontraron imágenes izquierdas para inferir image_size.")  # reportar ausencia
    img = cv2.imread(left_samples[0], cv2.IMREAD_COLOR)  # leer primera coincidencia
    if img is None:
        raise RuntimeError(f"No se pudo leer: {left_samples[0]}")  # reportar lectura fallida
    return (img.shape[1], img.shape[0])  # devolver (w, h)



def calcular_rectificacion(calibracion, image_size, alpha=0):
    """
    Calcula las matrices de rectificación R1, R2 y las de proyección P1, P2, junto con la matriz Q y ROIs válidas.

    Args:
        calibracion (dict): Diccionario con intrínsecas, distorsiones, R y T del sistema estéreo.
        image_size (tuple[int, int]): Tamaño de imagen (width, height) en píxeles.
        alpha (float): Parámetro de recorte de nuevo FOV (0=cortar máximo, 1=sin recorte). Por defecto, 0.

    Devuelve:
        tuple: Tupla (R1, R2, P1, P2, Q, roi1, roi2) con matrices y regiones de interés.
    """
    # llamar a stereoRectify para obtener rectificación y reproyección
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=calibracion["left_K"],
        distCoeffs1=calibracion["left_dist"],
        cameraMatrix2=calibracion["right_K"],
        distCoeffs2=calibracion["right_dist"],
        imageSize=image_size,
        R=calibracion["R"],
        T=calibracion["T"],
        alpha=alpha,
        newImageSize=image_size
    )
    return R1, R2, P1, P2, Q, roi1, roi2



def generar_mapas_rectificacion(calibracion, R1, R2, P1, P2, image_size):
    """
    Genera los mapas (x, y) para aplicar remapeo de rectificación en ambas cámaras.

    Args:
        calibracion (dict): Diccionario con intrínsecas y distorsiones de ambas cámaras.
        R1, R2 (np.ndarray): Matrices de rotación de rectificación.
        P1, P2 (np.ndarray): Matrices de proyección rectificadas.
        image_size (tuple[int, int]): Tamaño de imagen (width, height) en píxeles.

    Devuelve:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Mapas lmx, lmy, rmx, rmy para remap.
    """
    # inicializar mapas de rectificación para cámara izquierda
    lmx, lmy = cv2.initUndistortRectifyMap(
        calibracion["left_K"], calibracion["left_dist"], R1, P1, image_size, cv2.CV_32FC1
    )
    # inicializar mapas de rectificación para cámara derecha
    rmx, rmy = cv2.initUndistortRectifyMap(
        calibracion["right_K"], calibracion["right_dist"], R2, P2, image_size, cv2.CV_32FC1
    )
    return lmx, lmy, rmx, rmy


def guardar_stereo_maps(destino_pkl, mapas):
    """
    Guarda en disco los mapas y matrices de rectificación en formato pickle.

    Args:
        destino_pkl (str): Ruta completa del archivo de salida (.pkl).
        mapas (dict | tuple): Estructura con mapas y matrices de rectificación a serializar.

    Devuelve:
        None: No retorna valor; persiste el archivo en disco.
    """
    os.makedirs(os.path.dirname(destino_pkl), exist_ok=True)  # crear carpeta destino si no existe
    with open(destino_pkl, "wb") as f:
        pickle.dump(mapas, f)  # serializar estructura en pickle


def visualizar_rectificacion(
    left_path, right_path, lmx, lmy, rmx, rmy,
    step=40, line_color=(0, 255, 0), thickness=2,
    title="Rectificación estéreo"
):
    """
    Rectifica y visualiza un par estéreo, dibujando líneas epipolares horizontales para verificar alineación.

    Args:
        left_path (str): Ruta de la imagen izquierda.
        right_path (str): Ruta de la imagen derecha.
        lmx, lmy, rmx, rmy (np.ndarray): Mapas de rectificación (x, y) para cada cámara.
        step (int): Espaciado vertical entre líneas guía.
        line_color (tuple[int, int, int]): Color BGR de las líneas.
        thickness (int): Grosor de las líneas dibujadas.
        title (str): Título del gráfico.

    Devuelve:
        None: Muestra la figura y no retorna valor.
    """
    # leer imágenes originales
    imgL = cv2.imread(left_path, cv2.IMREAD_COLOR)
    imgR = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if imgL is None or imgR is None:
        raise RuntimeError("No se pudieron leer las imágenes para visualización.")  # reportar fallo de lectura

    # aplicar remapeo para rectificar
    Lr = cv2.remap(imgL, lmx, lmy, cv2.INTER_LINEAR)
    Rr = cv2.remap(imgR, rmx, rmy, cv2.INTER_LINEAR)

    # concatenar vistas lado a lado
    vis = np.hstack((Lr, Rr))

    # dibujar líneas epipolares horizontales
    for y in range(0, vis.shape[0], step):
        cv2.line(vis, (0, y), (vis.shape[1], y), line_color, thickness)  # trazar línea horizontal

    # mostrar resultado en matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))  # convertir a RGB para matplotlib
    plt.axis("off")  # ocultar ejes
    plt.title(title, fontsize=13, weight="bold")  # asignar título
    plt.show()


def cargar_stereo_maps(ruta_maps_pkl):
    """
    Carga desde disco un archivo pickle con mapas y matrices de rectificación estéreo.

    Args:
        ruta_maps_pkl (str): Ruta al archivo pickle previamente guardado.

    Devuelve:
        Any: Objeto deserializado tal como fue guardado (p. ej., diccionario con lmx, lmy, rmx, rmy y matrices).
    """
    # abrir y deserializar pickle de mapas
    with open(ruta_maps_pkl, "rb") as f:
        return pickle.load(f)  # devolver estructura cargada



def emparejar_pares(carpeta_capturas):
    """
    Empareja archivos de imágenes izquierda/derecha dentro de una carpeta y devuelve una lista ordenada de pares.

    Acepta extensiones .jpg y .png. Empareja por coincidencia de nombre y/o índice numérico en el filename.

    Args:
        carpeta_capturas (str): Ruta al directorio que contiene las capturas.

    Devuelve:
        list[tuple[str, str]]: Lista de pares (left_path, right_path) ordenados por clave numérica o nombre.
    """
    # listar rutas candidatas de imágenes
    rutas = sorted(
        glob.glob(os.path.join(carpeta_capturas, "*.jpg")) +
        glob.glob(os.path.join(carpeta_capturas, "*.png"))
    )

    def parse_side_and_key(fname):
        # extraer lado (L/R) y clave de ordenamiento
        name = os.path.basename(fname).lower()  # normalizar nombre
        if "left" in name or re.search(r"(^|[_\-])l([_\-]|\d|\.|$)", name):
            side = "L"  # asignar izquierdo
        elif "right" in name or re.search(r"(^|[_\-])r([_\-]|\d|\.|$)", name):
            side = "R"  # asignar derecho
        else:
            side = None  # marcar desconocido
        m = re.search(r"(\d+)", name)  # buscar índice numérico
        key = m.group(1) if m else name  # elegir clave
        return side, key

    # construir diccionarios por lado
    L_dict, R_dict = {}, {}
    for p in rutas:
        side, key = parse_side_and_key(p)  # parsear lado y clave
        if side == "L":
            L_dict.setdefault(key, p)  # registrar en izquierdas
        elif side == "R":
            R_dict.setdefault(key, p)  # registrar en derechas

    # intersectar claves presentes en ambos lados
    keys = sorted(set(L_dict) & set(R_dict))  # ordenar claves comunes
    if not keys:
        raise RuntimeError(f"No se pudieron emparejar capturas en {carpeta_capturas}. Verificar nombres L/R.")  # reportar falta de pares

    # construir lista de pares ordenados
    pares = [(L_dict[k], R_dict[k]) for k in keys]
    return pares



def interseccion_rois(roi1, roi2):
    """
    Calcula la intersección entre dos regiones de interés (ROI) y devuelve la ROI común.

    Args:
        roi1 (tuple[int, int, int, int]): Primera ROI como (x, y, w, h).
        roi2 (tuple[int, int, int, int]): Segunda ROI como (x, y, w, h).

    Devuelve:
        tuple[int, int, int, int]: ROI intersección (xI, yI, wI, hI).
    """
    # desempaquetar coordenadas de ambas ROIs
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    # calcular intersección en coordenadas
    xI, yI = max(x1, x2), max(y1, y2)  # calcular esquina superior izquierda de la intersección
    xA, yA = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)  # calcular esquina inferior derecha
    if not (xI < xA and yI < yA):
        raise ValueError("Las ROIs no se intersectan; revisar calibración.")  # reportar ausencia de solapamiento
    return xI, yI, (xA - xI), (yA - yI)



def rectificar_y_guardar_pares(
    pares, left_map_x, left_map_y, right_map_x, right_map_y,
    roi_interseccion, carpeta_salida, indice_inicial=1, interpolacion=cv2.INTER_LINEAR
):
    """
    Rectifica y recorta a la ROI común cada par de imágenes a color y guarda los resultados en formato PNG.

    Args:
        pares (list[tuple[str, str]]): Lista de rutas (left_path, right_path) ya emparejadas.
        left_map_x (np.ndarray): Mapa x para rectificación de la izquierda.
        left_map_y (np.ndarray): Mapa y para rectificación de la izquierda.
        right_map_x (np.ndarray): Mapa x para rectificación de la derecha.
        right_map_y (np.ndarray): Mapa y para rectificación de la derecha.
        roi_interseccion (tuple[int, int, int, int]): ROI común (x, y, w, h) para recorte posterior.
        carpeta_salida (str): Directorio donde guardar los PNG rectificados.
        indice_inicial (int): Índice inicial para numerar los archivos de salida.
        interpolacion (int): Flag de interpolación de OpenCV para el remapeo (p. ej., cv2.INTER_LINEAR).

    Devuelve:
        int: Cantidad de pares procesados y guardados.
    """
    os.makedirs(carpeta_salida, exist_ok=True)  # crear carpeta de salida si no existe
    xI, yI, wI, hI = roi_interseccion  # desempaquetar ROI común

    count = 0  # inicializar contador de pares procesados
    for i, (lp, rp) in enumerate(pares, start=indice_inicial):
        Lc = cv2.imread(lp, cv2.IMREAD_COLOR)  # leer imagen izquierda a color
        Rc = cv2.imread(rp, cv2.IMREAD_COLOR)  # leer imagen derecha a color
        if Lc is None or Rc is None:
            print("Error leyendo:", lp, "o", rp)  # informar error de lectura y continuar
            continue  # omitir pareja inválida

        Lr = cv2.remap(Lc, left_map_x,  left_map_y,  interpolacion)  # aplicar rectificación izquierda
        Rr = cv2.remap(Rc, right_map_x, right_map_y, interpolacion)  # aplicar rectificación derecha

        Lr_roi = Lr[yI:yI + hI, xI:xI + wI]  # recortar ROI en izquierda
        Rr_roi = Rr[yI:yI + hI, xI:xI + wI]  # recortar ROI en derecha

        cv2.imwrite(os.path.join(carpeta_salida, f"rect_left_color_{i:04d}.png"),  Lr_roi)  # guardar PNG izquierdo
        cv2.imwrite(os.path.join(carpeta_salida, f"rect_right_color_{i:04d}.png"), Rr_roi)  # guardar PNG derecho
        count += 1  # incrementar contador

    return count  # devolver cantidad de pares procesados


# ----------------------- Carga y utilidades -----------------------

def cargar_calibracion(base_dir):
    """
    Carga la calibración estéreo desde un archivo pickle ubicado en un directorio base.

    Args:
        base_dir (str): Ruta al directorio que contiene el archivo "stereo_calibration.pkl".

    Devuelve:
        dict: Diccionario con matrices intrínsecas, coeficientes de distorsión y parámetros estéreo.
    """
    # construir ruta del archivo de calibración
    with open(os.path.join(base_dir, "stereo_calibration.pkl"), "rb") as f:  # abrir archivo pickle
        calib = pickle.load(f)  # deserializar calibración
    return calib  # devolver estructura de calibración


def cargar_maps(base_dir):
    """
    Carga desde disco los mapas y matrices de rectificación estéreo.

    Args:
        base_dir (str): Ruta al directorio que contiene el archivo "stereo_maps.pkl".

    Devuelve:
        Any: Estructura deserializada con mapas de rectificación (p. ej., lmx, lmy, rmx, rmy) y matrices asociadas.
    """
    # construir ruta del archivo de mapas
    with open(os.path.join(base_dir, "stereo_maps.pkl"), "rb") as f:  # abrir archivo pickle
        maps = pickle.load(f)  # deserializar mapas y matrices
    return maps  # devolver estructura de mapas


def intrinseca_rectificada_y_Q(maps):
    """
    Extrae la intrínseca rectificada (K_rect) y ajusta la matriz Q al origen de la ROI común.

    La función toma P1 y Q (provenientes de `stereoRectify`) y corrige Q restando el desplazamiento
    de la intersección de las ROIs válidas para alinear los reproyectos con la ROI recortada.

    Args:
        maps (dict): Diccionario que contiene al menos las claves 'P1', 'Q', 'validRoi1' y 'validRoi2'.

    Devuelve:
        tuple[np.ndarray, np.ndarray, tuple[int, int]]: Tupla con (K_rect, Q_ajustada, (xI, yI)).
    """
    # extraer matrices y ROIs
    P1 = maps["P1"]; Q = maps["Q"]  # extraer proyección rectificada y matriz Q
    roi1 = maps["validRoi1"]; roi2 = maps["validRoi2"]  # extraer ROIs válidas
    # calcular intersección de ROIs en origen
    x1, y1, w1, h1 = roi1; x2, y2, w2, h2 = roi2  # desempaquetar
    xI, yI = max(x1, x2), max(y1, y2)  # calcular offsets comunes
    # construir intrínseca rectificada a partir de P1
    K_rect = P1[:3, :3].copy()  # copiar submatriz 3x3
    # ajustar Q al recorte (trasladar origen)
    Q_adj = Q.copy()  # copiar Q
    Q_adj[0, 3] -= float(xI)  # restar offset en x
    Q_adj[1, 3] -= float(yI)  # restar offset en y
    return K_rect, Q_adj, (xI, yI)  # devolver intrínseca, Q ajustada y offset


def generar_objpoints_grid(cols, rows, square_size_m):
    """
    Genera una grilla de puntos 3D (plano Z=0) de tamaño `cols x rows` con paso `square_size_m`.

    Args:
        cols (int): Cantidad de columnas de esquinas internas del patrón.
        rows (int): Cantidad de filas de esquinas internas del patrón.
        square_size_m (float): Tamaño del lado de cada celda en unidades reales (p. ej., metros).

    Devuelve:
        np.ndarray: Arreglo (cols*rows, 3) con coordenadas 3D en el plano Z=0.
    """
    # crear arreglo de puntos en Z=0
    objp = np.zeros((cols*rows, 3), np.float32)  # inicializar puntos 3D
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # generar coordenadas (x, y)
    objp *= float(square_size_m)  # escalar por tamaño real de celda
    return objp  # devolver grilla de puntos

# ----------------------- Detectores intercambiables -----------------------

def detect_checkerboard(img_gray, pattern=(10,7), square_size_m=0.0242):
    """
    Detecta esquinas de un checkerboard y devuelve esquinas refinadas y puntos 3D correspondientes.

    Args:
        img_gray (np.ndarray): Imagen en escala de grises donde se busca el patrón.
        pattern (tuple[int, int]): Esquinas internas (cols, rows) del checkerboard (por defecto, (10, 7)).
        square_size_m (float): Tamaño real del cuadrado del patrón para generar objpoints (por defecto, 0.0242).

    Devuelve:
        tuple[bool, np.ndarray|None, np.ndarray|None]: (ok, corners_subpix, objpoints_3d) o (False, None, None).
    """
    # buscar patrón de checkerboard
    ok, corners = cv2.findChessboardCorners(img_gray, pattern, None)  # detectar esquinas
    if not ok:
        return False, None, None  # devolver fallo si no se detecta
    # refinar esquinas a nivel subpíxel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)  # definir criterio
    corners = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)  # refinar esquinas
    # generar puntos 3D asociados al patrón
    objp = generar_objpoints_grid(pattern[0], pattern[1], square_size_m)  # crear objpoints
    return True, corners, objp  # devolver resultado exitoso

# Ejemplo de firma para Charuco (placeholder):
# def detect_charuco(img_gray, board, charuco_dict, cameraMatrix, distCoeffs):
#     # detectar, interpolar y alinear objp/corners
#     # devolver: (ok, corners_2d (N,1,2), objp_3d (N,3))
#     # return ok, corners2d, objp3d

# ----------------------- Nube desde disparidad -----------------------

def nube_desde_disparidad(disp, img_rgb, Q, *,
                          filtrar_dist=True, max_dist=0.8,
                          lim_xy=None):
    """
    Reconstruye una nube de puntos coloreada a partir de una disparidad y la matriz Q.

    La función reprojecta la disparidad a 3D con `cv2.reprojectImageTo3D`, filtra puntos inválidos
    y opcionalmente restringe por distancia y por límites en el plano XY.

    Args:
        disp (np.ndarray): Mapa de disparidad (float) con valores positivos donde hay correspondencia.
        img_rgb (np.ndarray): Imagen RGB alineada a la disparidad para colorear la nube.
        Q (np.ndarray): Matriz de reproyección 4x4 proveniente de `stereoRectify` (posiblemente ajustada).
        filtrar_dist (bool): Indica si aplica filtros por distancia/altura (por defecto, True).
        max_dist (float): Distancia máxima (en metros) para filtrar norma y z (por defecto, 0.8).
        lim_xy (tuple[tuple[float, float], tuple[float, float]] | None): Límites en x e y como ((xmin, xmax), (ymin, ymax)).

    Devuelve:
        o3d.geometry.PointCloud: Nube de puntos coloreada. Si no hay puntos válidos, devuelve una nube vacía.
    """
    # reprojectar disparidad a 3D
    pts3d = cv2.reprojectImageTo3D(disp, Q)  # obtener puntos 3D
    # construir máscara de puntos válidos
    mask = np.isfinite(disp) & (disp > 0)  # validar disparidad
    mask &= np.isfinite(pts3d[:,:,0]) & np.isfinite(pts3d[:,:,1]) & np.isfinite(pts3d[:,:,2])  # validar 3D
    # aplicar filtros de distancia si corresponde
    if filtrar_dist:
        z = pts3d[:,:,2]  # extraer profundidad
        dist = np.linalg.norm(pts3d, axis=2)  # calcular norma
        mask &= (z > 0.05) & (z < max_dist) & (dist < max_dist)  # filtrar por rango
    # aplicar recorte en XY si está definido
    if lim_xy is not None:
        (xmin, xmax), (ymin, ymax) = lim_xy  # desempaquetar límites
        x = pts3d[:,:,0]; y = pts3d[:,:,1]  # extraer ejes
        mask &= (x>xmin)&(x<xmax)&(y>ymin)&(y<ymax)  # aplicar máscara en XY

    # extraer puntos y colores filtrados
    pts = pts3d[mask]; cols = img_rgb[mask]  # seleccionar válidos
    pcd = o3d.geometry.PointCloud()  # crear nube vacía
    if len(pts) == 0:
        return pcd  # devolver nube vacía si no hay puntos
    # asignar puntos y colores a la nube
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))  # setear puntos
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64)/255.0)  # setear colores normalizados
    return pcd  # devolver nube resultante

# ----------------------- Procesamiento por vista -----------------------

def procesar_vista(left_img_path, disp_path, K_rect, Q_adj, detect_fn):
    """
    Procesa una vista: estima la pose cámara↔tablero y proyecta la nube a coordenadas del tablero.

    Lee la imagen izquierda y la disparidad, detecta el patrón (p. ej., checkerboard) con `detect_fn`,
    resuelve PnP para obtener la pose y transforma la nube reconstruida a coordenadas del mundo del tablero.

    Args:
        left_img_path (str): Ruta de la imagen izquierda asociada a la disparidad.
        disp_path (str): Ruta del archivo `.npy` con la disparidad correspondiente.
        K_rect (np.ndarray): Matriz intrínseca rectificada (3x3) de la cámara izquierda.
        Q_adj (np.ndarray): Matriz Q ajustada al origen de la ROI común.
        detect_fn (Callable): Función detectora que devuelve (ok, corners2d, objpoints3d).

    Devuelve:
        tuple[o3d.geometry.PointCloud|None, np.ndarray|None]: (nube_en_mundo, T_world_cam) o (None, None) si falla.
    """
    # leer imagen de entrada
    img_bgr  = cv2.imread(left_img_path, cv2.IMREAD_COLOR)  # leer BGR
    if img_bgr is None:
        return None, None  # devolver fallo si no existe
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # convertir a gris
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # convertir a RGB
    disp = np.load(disp_path).astype(np.float32)  # cargar disparidad

    # detectar patrón con función provista
    ok, corners, objp = detect_fn(img_gray)  # ejecutar detector
    if not ok:
        return None, None  # devolver fallo si no detecta

    # resolver PnP para estimar pose
    ok_pnp, rvec, tvec = cv2.solvePnP(
        objp, corners, K_rect, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok_pnp:
        return None, None  # devolver fallo si PnP no converge

    # convertir rvec a matriz de rotación y armar transformaciones
    R, _ = cv2.Rodrigues(rvec)  # convertir a rotación 3x3
    T = np.eye(4, dtype=np.float64)  # crear T homogénea
    T[:3,:3] = R  # setear rotación
    T[:3, 3] = tvec.flatten()  # setear traslación

    # invertir para obtener cámara->mundo del tablero
    T_world_cam = np.linalg.inv(T)  # invertir transformación

    # reconstruir nube local y transformar a mundo
    pcd_local = nube_desde_disparidad(
        disp, img_rgb, Q_adj,
        filtrar_dist=True, max_dist=0.8,
        lim_xy=((-0.3, 0.3), (-0.3, 0.3))
    )
    if len(pcd_local.points) == 0:
        return None, T_world_cam  # devolver solo pose si no hay puntos

    pcd_world = pcd_local.transform(T_world_cam)  # transformar nube a marco del tablero
    return pcd_world, T_world_cam  # devolver nube y pose

# ----------------------- Post-proceso nube -----------------------

def fusionar_nubes(lista_pcd):
    """
    Fusiona una lista de nubes de puntos en una sola nube acumulada.

    Args:
        lista_pcd (list[o3d.geometry.PointCloud]): Conjunto de nubes a combinar.

    Devuelve:
        o3d.geometry.PointCloud: Nube resultante de concatenar todas las entradas (o vacía si la lista está vacía).
    """
    # manejar caso vacío
    if not lista_pcd:
        return o3d.geometry.PointCloud()  # devolver nube vacía
    # inicializar con la primera nube
    nube = lista_pcd[0]  # tomar base
    for p in lista_pcd[1:]:
        nube += p  # sumar nubes sucesivas
    return nube  # devolver nube fusionada


def filtrar_nube_global(pcd, *, usar_estadistico=True, nb=30, std=1.5, voxel=0.0):
    """
    Filtra una nube de puntos mediante outlier estadístico y/o muestreo voxel.

    Args:
        pcd (o3d.geometry.PointCloud): Nube de puntos a filtrar.
        usar_estadistico (bool): Indica si aplica eliminación estadística de outliers (por defecto, True).
        nb (int): Número de vecinos para el filtro estadístico (por defecto, 30).
        std (float): Umbral en desviaciones estándar para conservar puntos (por defecto, 1.5).
        voxel (float): Tamaño del voxel para `voxel_down_sample` (0 desactiva; por defecto, 0.0).

    Devuelve:
        o3d.geometry.PointCloud: Nube filtrada.
    """
    # aplicar filtro estadístico si corresponde
    if usar_estadistico and len(pcd.points) > 100:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)  # calcular índices válidos
        pcd = pcd.select_by_index(ind)  # seleccionar inliers
    # aplicar muestreo por voxel si está habilitado
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)  # reducir densidad
    return pcd  # devolver nube filtrada


def guardar_ply(pcd, path):
    """
    Guarda una nube de puntos en formato PLY.

    Args:
        pcd (o3d.geometry.PointCloud): Nube a serializar en disco.
        path (str): Ruta de salida, típicamente con extensión ".ply".

    Devuelve:
        str: Ruta de salida escrita.
    """
    o3d.io.write_point_cloud(path, pcd)  # escribir nube en disco
    return path  # devolver ruta escrita


def visualizar_vistas(pcd, poses, *, frame_size=0.1, cam_size=0.05, title="Reconstrucción"):
    """
    Visualiza la nube de puntos junto con marcos de cámara estimados y un marco de referencia.

    Args:
        pcd (o3d.geometry.PointCloud): Nube a visualizar.
        poses (list[np.ndarray]): Lista de transformaciones 4x4 (cámara en coordenadas de mundo del tablero).
        frame_size (float): Tamaño del marco de referencia global (por defecto, 0.1).
        cam_size (float): Tamaño de los ejes que representan cada cámara (por defecto, 0.05).
        title (str): Título de la ventana (por defecto, "Reconstrucción").

    Devuelve:
        None: Muestra una ventana interactiva con geometrías.
    """
    ref = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)  # crear marco global
    cams = []  # inicializar lista de marcos de cámara
    for T in poses:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cam_size)  # crear marco de cámara
        cf.transform(T)  # transformar a la pose
        cams.append(cf)  # acumular marco
    o3d.visualization.draw_geometries([pcd, ref] + cams, window_name=title, width=1280, height=720)  # visualizar


def bbox_dimensiones(pcd, oriented=True):
    """
    Calcula dimensiones (dx, dy, dz) en metros y retorna también la caja (AABB u OBB).

    Args:
        pcd (o3d.geometry.PointCloud): Nube de puntos de interés.
        oriented (bool): Indica si usa OBB (True) o AABB (False). Por defecto, True.

    Devuelve:
        tuple[np.ndarray|None, o3d.geometry.AxisAlignedBoundingBox|o3d.geometry.OrientedBoundingBox|None]:
            Par (dims, box), o (None, None) si la nube está vacía.
    """
    # manejar nube vacía
    if len(pcd.points) == 0:
        return None, None  # devolver vacío
    # elegir tipo de caja
    if oriented:
        box = pcd.get_oriented_bounding_box()  # calcular OBB
    else:
        box = pcd.get_axis_aligned_bounding_box()  # calcular AABB
    dims = np.abs(box.extent)  # obtener dimensiones
    return dims, box  # devolver resultados


def rango_eje(pcd, axis='z'):
    """
    Obtiene el rango (mínimo, máximo y amplitud) de un eje cartesiano de la nube.

    Args:
        pcd (o3d.geometry.PointCloud): Nube de puntos de interés.
        axis (str): Eje a evaluar ('x', 'y' o 'z'). Por defecto, 'z'.

    Devuelve:
        tuple[float, float, float] | None: (min, max, max-min) del eje solicitado, o None si la nube está vacía.
    """
    # manejar nube vacía
    if len(pcd.points) == 0:
        return None  # devolver vacío
    # mapear eje a índice
    idx = {'x': 0, 'y': 1, 'z': 2}[axis]  # seleccionar índice de columna
    pts = np.asarray(pcd.points)  # convertir a ndarray
    vmin = float(np.min(pts[:, idx]))  # calcular mínimo
    vmax = float(np.max(pts[:, idx]))  # calcular máximo
    return vmin, vmax, (vmax - vmin)  # devolver rango


def altura_sobre_tablero(pcd):
    """
    Calcula la altura (rango en Z) de la nube respecto del tablero, asumiendo que Z=0 pertenece al plano del patrón.

    Args:
        pcd (o3d.geometry.PointCloud): Nube de puntos cuyo rango en Z se desea medir.

    Devuelve:
        float | None: Altura (z_max - z_min) en metros, o None si la nube está vacía.
    """
    # obtener rango en eje Z
    r = rango_eje(pcd, 'z')  # calcular (min, max, delta)
    if r is None:
        return None  # devolver vacío si no hay puntos
    zmin, zmax, h = r  # desempaquetar resultados
    return h  # devolver altura