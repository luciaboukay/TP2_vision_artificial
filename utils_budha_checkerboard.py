import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
import re
import open3d as o3d
from pathlib import Path
from stereodemo.method_cre_stereo import CREStereo
from stereodemo.method_opencv_bm import StereoBM, StereoSGBM
from stereodemo.methods import InputPair, Config
from typing import Tuple


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


def calibrar_stereo_checkerboard(base_dir, img_paths, checkerboard, square_size):
    """
    Calibra un sistema estéreo a partir de imágenes de un checkerboard y guarda los parámetros obtenidos.

    Args:
        base_dir (str): Directorio base con las imágenes y la salida del pickle.
        img_paths (list[str]): Lista de rutas de imágenes para calibrar.
        checkerboard (tuple[int, int]): Cantidad de esquinas internas (cols, rows) del patrón.
        square_size (float): Tamaño real del cuadrado del checkerboard.

    Devuelve:
        dict: Diccionario con intrínsecas y coeficientes de distorsión de ambas cámaras, matrices R, T, E, F y el image_size.
    """
    # buscar y ordenar rutas de imágenes left/right
    left_images  = sorted(glob.glob(os.path.join(base_dir, img_paths[0])))
    right_images = sorted(glob.glob(os.path.join(base_dir, img_paths[1])))
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

def visualizar_imagenes_rectificadas(path_rectificadas, step=40, n_mostrar=2):
    """
    Muestra pares de imágenes rectificadas lado a lado con líneas epipolares horizontales
    para verificar visualmente la alineación estéreo.

    Args:
        path_rectificadas (str): Ruta a la carpeta que contiene las imágenes rectificadas.
            Debe incluir archivos con nombres del tipo 'rect_left_*.jpg' y 'rect_right_*.jpg'.
        step (int): Espaciado vertical en píxeles entre líneas epipolares. Por defecto, 40.
        n_mostrar (int): Cantidad de pares de imágenes a visualizar. Por defecto, 2.

    Devuelve:
        None: Muestra las figuras en pantalla sin retornar un valor.
    """
    # Buscar y ordenar pares de imágenes rectificadas
    left_imgs  = sorted(glob.glob(os.path.join(path_rectificadas, "rect_left_*.jpg")))
    right_imgs = sorted(glob.glob(os.path.join(path_rectificadas, "rect_right_*.jpg")))

    # Validar existencia y correspondencia de pares
    if len(left_imgs) == 0 or len(left_imgs) != len(right_imgs):
        print("No se encontraron pares válidos en:", path_rectificadas)
        print(f"   Left: {len(left_imgs)} | Right: {len(right_imgs)}")
        return

    n_pairs = min(n_mostrar, len(left_imgs))

    # Iterar sobre los primeros n pares
    for i in range(n_pairs):
        imgL = cv2.imread(left_imgs[i], cv2.IMREAD_COLOR)
        imgR = cv2.imread(right_imgs[i], cv2.IMREAD_COLOR)

        if imgL is None or imgR is None:
            print(f"Error leyendo: {os.path.basename(left_imgs[i])} o {os.path.basename(right_imgs[i])}")
            continue

        # Convertir BGR → RGB para visualización correcta en matplotlib
        imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

        # Concatenar horizontalmente ambas vistas
        vis = np.hstack((imgL_rgb, imgR_rgb))

        # Dibujar líneas epipolares horizontales
        vis_lines = vis.copy()
        for y in range(0, vis_lines.shape[0], step):
            cv2.line(vis_lines, (0, y), (vis_lines.shape[1], y), (0, 255, 0), 1)

        # Mostrar resultado
        plt.figure(figsize=(12, 5))
        plt.imshow(vis_lines)
        plt.title(
            f"Par {i+1} — {os.path.basename(left_imgs[i])} / {os.path.basename(right_imgs[i])}",
            fontsize=12, weight="bold"
        )
        plt.axis("off")
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

def calcular_disparidad_stereodemo(path_imgs, out_dir, metodo="cre"):
    """
    Procesa imágenes ya rectificadas para generar mapas de disparidad utilizando distintos métodos estéreo.

    Guarda en disco:
        - Mapas de disparidad crudos (.npy)
        - Mapas normalizados en escala de grises (.png)
        - Mapas colorizados (.png)

    Args:
        path_imgs (str): Ruta al directorio que contiene las imágenes rectificadas 
            (archivos con nombres tipo 'rect_left_*.jpg' y 'rect_right_*.jpg').
        out_dir (str): Carpeta base donde se guardarán los resultados.
        metodo (str): Método de correspondencia estéreo a emplear.
            Valores válidos:
                - "cre": usa CREStereo (modelo profundo)
                - "bm":  usa StereoBM (bloque clásico)
                - "sgbm": usa StereoSGBM (semi-global)
            Por defecto: "cre".

    Devuelve:
        None: Muestra mensajes de progreso y guarda los resultados en disco.
    """
    # Ruta base del modelo CREStereo
    models_path = Path.home() / ".cache" / "stereodemo" / "models"

    # Inicializar configuración y seleccionar método
    config = Config(models_path=models_path)
    if metodo == "cre":
        method = CREStereo(config)
    elif metodo == "bm":
        method = StereoBM(config)
    elif metodo == "sgbm":
        method = StereoSGBM(config)
    else:
        raise ValueError(f"Método no reconocido: {metodo}. Usar 'cre', 'bm' o 'sgbm'.")

    # Crear carpeta de salida específica para el método
    out_dir = os.path.join(out_dir, f"disparidad_{metodo}")
    os.makedirs(out_dir, exist_ok=True)

    # Buscar imágenes rectificadas
    left_imgs  = sorted(glob.glob(os.path.join(path_imgs, "rect_left_*.jpg")))
    right_imgs = sorted(glob.glob(os.path.join(path_imgs, "rect_right_*.jpg")))

    if len(left_imgs) == 0 or len(left_imgs) != len(right_imgs):
        print("No se encontraron pares válidos de imágenes rectificadas.")
        return

    print(f"Imágenes encontradas: {len(left_imgs)} pares")
    print(f"Método seleccionado: {metodo.upper()}")

    # Iterar sobre los pares
    for i, (left_path, right_path) in enumerate(zip(left_imgs, right_imgs), 1):

        # Cargar imágenes
        imgL = cv2.imread(left_path, cv2.IMREAD_COLOR)
        imgR = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if imgL is None or imgR is None:
            print(f"Error leyendo imágenes: {left_path} o {right_path}")
            continue

        # Construir InputPair (las imágenes ya están rectificadas, sin calibración adicional)
        pair = InputPair(
            left_image=imgL,
            right_image=imgR,
            calibration=None,
            status=os.path.basename(left_path)
        )

        # Calcular disparidad
        disparity = method.compute_disparity(pair)
        d = disparity.disparity_pixels.astype(np.float32)

        # Generar nombre base
        base = os.path.basename(left_path).replace("rect_left_", "").replace(".jpg", "")

        # Guardar disparidad cruda
        np.save(os.path.join(out_dir, f"disp_raw_{base}.npy"), d)

        # Normalizar disparidad a [0, 255] para visualización
        d_min, d_max = np.nanmin(d), np.nanmax(d)
        if np.isfinite(d_min) and np.isfinite(d_max) and d_max > d_min:
            dvis = 255 * (d - d_min) / (d_max - d_min)
        else:
            dvis = np.zeros_like(d)
        dvis = np.clip(dvis, 0, 255).astype(np.uint8)

        # Aplicar colorización (colormap JET)
        dvis_color = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)

        # Guardar resultados visuales
        cv2.imwrite(os.path.join(out_dir, f"disp_gray_{base}.png"), dvis)
        cv2.imwrite(os.path.join(out_dir, f"disp_color_{base}.png"), dvis_color)

    cv2.destroyAllWindows()
    print(f"Resultados guardados en: {out_dir}")

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


def detect_charuco_markers(image, board, detector_params=None):
    """
    Detecta los marcadores ArUco presentes en un tablero ChArUco dentro de una imagen.

    Args:
        image (np.ndarray): Imagen de entrada en formato BGR o en escala de grises.
        board (cv2.aruco.CharucoBoard): Tablero ChArUco utilizado para la detección.
        detector_params (cv2.aruco.DetectorParameters | None): Parámetros opcionales del detector.
            Si no se especifican, se crean con valores más permisivos.

    Devuelve:
        dict | None: Diccionario con las claves:
            - 'corners' (list[np.ndarray]): Lista de esquinas detectadas para cada marcador.
            - 'ids' (np.ndarray): Arreglo de IDs asociados a los marcadores detectados.
            - 'rejected' (list[np.ndarray]): Marcadores rechazados durante la detección.
        Si no se detectan marcadores válidos, retorna None.
    """
    # Convertir a escala de grises si es necesario
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Inicializar parámetros del detector si no se proveen
    if detector_params is None:
        detector_params = cv2.aruco.DetectorParameters()
        # configurar parámetros más permisivos para robustecer detección
        detector_params.adaptiveThreshWinSizeMin = 3
        detector_params.adaptiveThreshWinSizeMax = 23
        detector_params.adaptiveThreshWinSizeStep = 10
        detector_params.minMarkerPerimeterRate = 0.03
        detector_params.maxMarkerPerimeterRate = 4.0

    # Crear detector ArUco asociado al diccionario del tablero
    detector = cv2.aruco.ArucoDetector(board.getDictionary(), detector_params)

    # Detectar marcadores en la imagen
    corners, ids, rejected = detector.detectMarkers(gray)

    # Validar detección
    if ids is None or len(ids) == 0:
        return None

    # Ordenar por ID ascendente
    ids = ids.flatten()
    order = np.argsort(ids)
    ids = ids[order]
    corners = [corners[i] for i in order]

    # Empaquetar resultados en diccionario
    result = {
        "corners": corners,
        "ids": ids,
        "rejected": rejected
    }

    return result

# ----------------------- Estimación de pose mediante homografía -----------------------

def estimate_camera_pose_with_homography(image, board, detection, calibration, undistort=False):
    """
    Estima la pose de la cámara con respecto a un tablero ChArUco utilizando una homografía
    inicial seguida de una optimización por PnP con el método IPPE (ideal para planos).

    La función utiliza las correspondencias 2D–3D obtenidas de los marcadores detectados
    para estimar primero una homografía entre el plano del tablero y la imagen, y luego
    refina la pose de la cámara resolviendo PnP con las esquinas del tablero.

    Args:
        image (np.ndarray): Imagen de entrada (BGR o en escala de grises) donde se detectó el tablero.
        board (cv2.aruco.CharucoBoard): Tablero ChArUco con la geometría conocida del patrón.
        detection (dict): Diccionario devuelto por `detect_charuco_markers` con claves:
            - 'corners': lista de esquinas detectadas.
            - 'ids': arreglo de IDs de marcadores detectados.
        calibration (tuple[np.ndarray, np.ndarray]): Tupla (K, dist) con la matriz intrínseca y
            coeficientes de distorsión de la cámara.
        undistort (bool): Si es True, se corrigen los puntos antes y después de estimar la homografía.
            Por defecto, False.

    Devuelve:
        tuple[bool, np.ndarray, np.ndarray] | None:
            (success, rvec, tvec) si la estimación fue exitosa, o None en caso de fallo.
    """
    # Extraer datos de entrada
    corners = detection["corners"]
    ids = detection["ids"]
    K, dist = calibration

    # Validar detección mínima
    if len(ids) < 4:
        return None

    # Construir correspondencias 2D (imagen) ↔ 2D (plano del tablero)
    image_points = []
    board_points = []
    board_ids = board.getIds()
    points3d = board.getObjPoints()

    for corner, id in zip(corners, ids):
        id = int(id)
        if id not in board_ids:
            continue
        point3d = points3d[id]           # (4, 3)
        image_points.extend(corner[0])   # (4, 2)
        board_points.extend(point3d)     # (4, 3)

    image_points = np.array(image_points, dtype=np.float32)
    board_points = np.array(board_points, dtype=np.float32)

    # Validar cantidad de correspondencias suficientes
    if len(image_points) < 4:
        return None

    # Corregir distorsión si se solicita
    if undistort:
        use_image_points = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2), K, dist, P=K
        ).reshape(-1, 2)
    else:
        use_image_points = image_points

    # Estimar homografía (2D→2D) entre plano del tablero y la imagen
    H, inliers = cv2.findHomography(
        board_points[:, :2],
        use_image_points,
        method=cv2.LMEDS
    )
    if H is None:
        return None

    # Obtener todas las esquinas del tablero ChArUco en el plano del patrón
    charuco_obj_points = board.getChessboardCorners()  # (N, 3)
    charuco_board_points = np.array(
        [p[:2] for p in charuco_obj_points],
        dtype=np.float32
    ).reshape(-1, 1, 2)

    # Proyectar puntos 2D del tablero a la imagen mediante la homografía
    projected_charuco_corners = cv2.perspectiveTransform(charuco_board_points, H)
    projected_charuco_corners = projected_charuco_corners.reshape(-1, 2)
    obj_pts = np.array(charuco_obj_points, dtype=np.float32)

    # Aplicar corrección de distorsión a los puntos proyectados si corresponde
    if undistort:
        use_image_points = cv2.undistortPoints(
            projected_charuco_corners.reshape(-1, 1, 2), K, dist, P=K
        ).reshape(-1, 2)
    else:
        use_image_points = projected_charuco_corners

    # Resolver la pose mediante PnP con el método IPPE (óptimo para planos)
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        use_image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE
    )
    if not success:
        return None

    return success, rvec, tvec

# ----------------------- Estimación de pose de tablero ChArUco -----------------------

def detect_charuco_pose(image, camera_matrix, dist_coeffs, board, verbose=False):
    """
    Detecta un tablero ChArUco en una imagen y estima la pose de la cámara respecto del mismo,
    utilizando homografía inicial seguida de resolución por PnP (IPPE).

    Esta función combina la detección de marcadores ArUco y la estimación de pose plana del tablero,
    retornando tanto los vectores de rotación y traslación como las esquinas e IDs detectados.

    Args:
        image (np.ndarray): Imagen de entrada (BGR o en escala de grises) donde se busca el tablero.
        camera_matrix (np.ndarray): Matriz intrínseca de la cámara (3x3).
        dist_coeffs (np.ndarray): Coeficientes de distorsión de la cámara.
        board (cv2.aruco.CharucoBoard): Tablero ChArUco utilizado para la detección y estimación de pose.
        verbose (bool): Si es True, imprime información detallada del proceso. Por defecto, False.

    Devuelve:
        tuple[bool, np.ndarray|None, np.ndarray|None, list[np.ndarray]|None, np.ndarray|None]:
            (success, rvec, tvec, corners, ids), donde:
                - success (bool): True si se estimó la pose correctamente.
                - rvec (np.ndarray): Vector de rotación (Rodrigues).
                - tvec (np.ndarray): Vector de traslación (posición cámara→tablero).
                - corners (list[np.ndarray]): Lista de esquinas detectadas de los marcadores.
                - ids (np.ndarray): IDs correspondientes a los marcadores detectados.
            En caso de fallo, devuelve (False, None, None, None, None).
    """
    # detectar marcadores ArUco en el tablero
    detection = detect_charuco_markers(image, board)
    if detection is None:
        if verbose:
            print("No se detectaron marcadores ArUco.")
        return False, None, None, None, None

    if verbose:
        print(f"Marcadores ArUco detectados: {len(detection['ids'])}")

    # estimar la pose usando homografía + PnP
    result = estimate_camera_pose_with_homography(
        image,
        board,
        detection,
        (camera_matrix, dist_coeffs),
        undistort=False
    )

    if result is None:
        if verbose:
            print("No se pudo estimar la pose.")
        return False, None, None, None, None

    success, rvec, tvec = result

    if verbose:
        print("Pose estimada correctamente.")
        print(f"     Traslación (t): {tvec.flatten()}")

    return success, rvec, tvec, detection["corners"], detection["ids"]

# ----------------------- Dibujo y visualización de tableros ChArUco -----------------------

def draw_charuco_markers(image, detection, draw_rejected=False):
    """
    Dibuja sobre una imagen los resultados de la detección de un tablero ChArUco.

    Args:
        image (np.ndarray): Imagen de entrada (BGR o RGB).
        detection (dict | None): Resultado devuelto por `detect_charuco_markers`, con claves:
            - 'corners': lista de esquinas detectadas.
            - 'ids': IDs de los marcadores válidos.
            - 'rejected': marcadores descartados (opcional).
        draw_rejected (bool): Si True, también dibuja los marcadores rechazados en rojo.

    Devuelve:
        np.ndarray: Imagen de salida con los marcadores dibujados.
    """
    output_image = image.copy()
    if detection is None:
        return output_image

    corners = detection["corners"]
    ids = detection["ids"]
    rejected = detection["rejected"]

    # Dibujar marcadores aceptados (bordes verdes)
    if ids is not None and len(ids) > 0:
        output_image = cv2.aruco.drawDetectedMarkers(output_image, corners, ids)

    # Dibujar marcadores rechazados (bordes rojos)
    if draw_rejected and rejected is not None and len(rejected) > 0:
        cv2.aruco.drawDetectedMarkers(output_image, rejected, borderColor=(0, 0, 255))

    return output_image


def visualizar_deteccion_charuco(image_path, camera_matrix, dist_coeffs, board, square_length):
    """
    Visualiza la detección de un tablero ChArUco en una imagen y estima su pose.

    Args:
        image_path (str): Ruta a la imagen donde se evaluará la detección.
        camera_matrix (np.ndarray): Matriz intrínseca de la cámara.
        dist_coeffs (np.ndarray): Coeficientes de distorsión.
        board (cv2.aruco.CharucoBoard): Tablero ChArUco utilizado.
        square_length (float): Longitud del cuadrado del tablero (en mm).

    Devuelve:
        None: Muestra la imagen con la detección visualizada y mensajes de estado.
    """
    # Leer imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return

    # Detectar marcadores
    detection = detect_charuco_markers(img, board)

    # Dibujar resultados sobre la imagen
    img_markers = draw_charuco_markers(img, detection, draw_rejected=True)

    if detection is None:
        print(f"No se detectaron marcadores ArUco en {os.path.basename(image_path)}")
    else:
        print(f"Detectados {len(detection['ids'])} marcadores ArUco")

        # Estimar pose a partir de la detección
        result = estimate_camera_pose_with_homography(
            img, board, detection, (camera_matrix, dist_coeffs), undistort=False
        )

        if result is not None:
            success, rvec, tvec = result
            if success:
                # Dibujar ejes 3D del tablero
                axis_length = square_length / 1000.0 * 2  # 2 cuadrados de longitud
                cv2.drawFrameAxes(
                    img_markers, camera_matrix, dist_coeffs,
                    rvec, tvec, axis_length, thickness=3
                )
                print(f"Pose estimada correctamente")
                print(f"   Traslación (x, y, z): "
                      f"({tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}) m")
            else:
                print("No se pudo estimar la pose")
        else:
            print("Insuficientes marcadores para estimar pose")

    # Mostrar resultado con Matplotlib
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_markers, cv2.COLOR_BGR2RGB))
    title = f"Detección ChArUco - {os.path.basename(image_path)}"
    if detection is not None and result is not None:
        title += " Detección exitosa"
    else:
        title += " Fallo en la detección"
    plt.title(title, fontsize=14, weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------- Análisis de calidad de detecciones ChArUco -----------------------

def analizar_calidad_detecciones(
    visualizar=True,
    charuco_board=None,
    square_length=52.6,
    max_visualizaciones=21
):
    """
    Analiza todas las imágenes del dataset y reporta la calidad de detección
    del tablero ChArUco. Además, muestra una figura única con todas las poses
    estimadas si 'visualizar' es True.

    Args:
        visualizar (bool): Si True, muestra todas las detecciones con ejes y marcadores.
        charuco_board (cv2.aruco.CharucoBoard): Tablero ChArUco utilizado para la detección.
        square_length (float): Longitud del cuadrado del tablero (en mm).
        max_visualizaciones (int): Máximo de imágenes a visualizar.

    Devuelve:
        list[dict]: Lista con resultados de detección (imagen, éxito, marcadores, tvec).
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS DE CALIDAD DE DETECCIÓN DE CHARUCO")
    print("=" * 80)

    base_dir = "tp2_reconstruccion_3d/datasets/stereo_budha_charuco"
    calib_path = os.path.join(base_dir, "calib/stereo_calibration.pkl")

    # Cargar calibración estéreo
    with open(calib_path, "rb") as f:
        calib = pickle.load(f)

    left_K = calib["left_K"]
    left_dist = calib["left_dist"]

    captures_dir = os.path.join(base_dir, "captures")
    left_imgs = sorted(glob.glob(os.path.join(captures_dir, "left_*.jpg")))

    resultados = []
    imagenes_vis = []  # para almacenar visualizaciones

    for i, img_path in enumerate(left_imgs):
        img = cv2.imread(img_path)
        if img is None:
            continue

        success, rvec, tvec, corners, ids = detect_charuco_pose(
            img, left_K, left_dist, charuco_board, verbose=False
        )

        num_markers = len(ids) if ids is not None else 0
        resultados.append({
            "imagen": os.path.basename(img_path),
            "exito": success,
            "num_marcadores": num_markers,
            "tvec": tvec.flatten() if tvec is not None else None
        })

        # Si hay que visualizar, dibujamos sobre la imagen
        if visualizar and i < max_visualizaciones:
            detection = {"corners": corners, "ids": ids, "rejected": None}
            img_draw = draw_charuco_markers(img, detection, draw_rejected=True)

            if success:
                axis_length = square_length / 1000.0 * 2  # dos cuadrados de longitud
                cv2.drawFrameAxes(img_draw, left_K, left_dist, rvec, tvec, axis_length, thickness=2)
                label = f"{os.path.basename(img_path)}"
            else:
                label = f"{os.path.basename(img_path)} (Fallo)"

            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            imagenes_vis.append((img_rgb, label))

    # ----------------------------------------------------------------------
    # Tabla resumen
    # ----------------------------------------------------------------------
    print(f"\n{'Imagen':<15} {'Estado':<10} {'Marcadores':<12} {'Posición (x, y, z)'}")
    print("-" * 80)
    exitos = 0
    for r in resultados:
        estado = "OK" if r["exito"] else "FALLO"
        if r["exito"]:
            exitos += 1
            pos = f"({r['tvec'][0]:.3f}, {r['tvec'][1]:.3f}, {r['tvec'][2]:.3f})"
        else:
            pos = "N/A"
        print(f"{r['imagen']:<15} {estado:<10} {r['num_marcadores']:<12} {pos}")

    print("-" * 80)
    print(f"\nRESUMEN GLOBAL:")
    print(f"   Total de imágenes: {len(resultados)}")
    print(f"   Éxitos: {exitos} | Fallos: {len(resultados) - exitos}")
    print(f"   Tasa de éxito: {100 * exitos / len(resultados):.1f}%")

    # ----------------------------------------------------------------------
    # Visualización combinada
    # ----------------------------------------------------------------------
    if visualizar and imagenes_vis:
        n = len(imagenes_vis)
        cols = 7
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(3.2 * cols, 3.5 * rows))
        for i, (img_rgb, label) in enumerate(imagenes_vis, 1):
            plt.subplot(rows, cols, i)
            plt.imshow(img_rgb)
            plt.title(label, fontsize=9)
            plt.axis("off")

        plt.suptitle("Detección y Pose de Tableros ChArUco", fontsize=14, weight="bold")
        plt.tight_layout()
        plt.show()

    return resultados


# ----------------------- Reconstruccion 3D charuco -----------------------
def reconstruccion_3d_charuco(
    max_distance=0.8,
    voxel_size=0.002,
    max_images=None,
    min_markers=6,
    filtrar_planos=True,
    charuco_board=None
):
    """
    Pipeline de reconstrucción 3D con sistema de referencia FIJO (tablero ChArUco).

    Integra detección de pose, reproyección 3D y combinación de nubes, usando funciones internas del módulo.

    Args:
        max_distance (float): Distancia máxima válida en metros.
        voxel_size (float): Tamaño de voxel para el muestreo.
        max_images (int | None): Límite de imágenes a procesar (None = todas).
        min_markers (int): Mínimo de marcadores ArUco requeridos.
        filtrar_planos (bool): Si True, elimina planos grandes (fondo/mesa).

    Devuelve:
        o3d.geometry.PointCloud | None: Nube 3D combinada (en coordenadas del tablero) o None si falla.
    """
    # Directorios base
    base_dir = "tp2_reconstruccion_3d/datasets/stereo_budha_charuco"
    calib_path = os.path.join(base_dir, "calib")
    captures_dir = os.path.join(base_dir, "captures")
    disparity_dir = os.path.join(captures_dir, "disparidad_cre")

    # Cargar calibración y mapas
    calib = cargar_calibracion(calib_path)
    maps = cargar_maps(calib_path)

    left_K = calib["left_K"]
    left_dist = calib["left_dist"]
    Q = maps["Q"]

    # Listar imágenes disponibles
    left_imgs = sorted(glob.glob(os.path.join(captures_dir, "left_*.jpg")))
    if max_images is not None:
        left_imgs = left_imgs[:max_images]

    if len(left_imgs) == 0:
        print("No se encontraron imágenes para reconstrucción.")
        return None

    all_point_clouds = []
    successful = 0

    print(f"\nIniciando reconstrucción 3D con {len(left_imgs)} imágenes...\n")

    for i, left_path in enumerate(left_imgs, start=1):
        img = cv2.imread(left_path)
        if img is None:
            print(f"[{i}] Error cargando imagen: {left_path}")
            continue

        # Detección de tablero ChArUco
        success, rvec, tvec, corners, ids = detect_charuco_pose(
            img, left_K, left_dist, charuco_board, verbose=False
        )

        if not success or ids is None or len(ids) < min_markers:
            print(f"[{i}] Marcadores insuficientes ({len(ids) if ids is not None else 0}/{min_markers})")
            continue

        # Validar distancia física
        dist_to_cam = np.linalg.norm(tvec)
        if dist_to_cam > max_distance * 1.5 or dist_to_cam < 0.1:
            continue

        # Construir transformaciones
        T_board_to_cam = create_transform_matrix(rvec, tvec)
        T_cam_to_board = np.linalg.inv(T_board_to_cam)

        # Cargar disparidad
        base_name = os.path.basename(left_path).replace("left_", "").replace(".jpg", "")
        disp_path = os.path.join(disparity_dir, f"disp_raw_{base_name}.npy")

        if not os.path.exists(disp_path):
            print(f"[{i}] No existe disparidad asociada ({disp_path})")
            continue

        disparity = np.load(disp_path).astype(np.float32)
        valid_ratio = np.sum(disparity > 0) / disparity.size
        if valid_ratio < 0.1:
            continue

        # Leer color y alinear tamaño
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != disparity.shape:
            img_rgb = cv2.resize(img_rgb, (disparity.shape[1], disparity.shape[0]))

        # Reconstruir nube local
        pcd_local = nube_desde_disparidad(
            disparity,
            img_rgb,
            Q,
            filtrar_dist=True,
            max_dist=max_distance,
            lim_xy=((-0.3, 0.3), (-0.3, 0.3))
        )

        if len(pcd_local.points) == 0:
            continue

        # Transformar a coordenadas del tablero
        pcd_local.transform(T_cam_to_board)

        # Filtrado estadístico y por radio
        pcd_local = filtrar_nube_global(
            pcd_local,
            usar_estadistico=True,
            nb=30,
            std=1.5,
            voxel=voxel_size
        )

        if len(pcd_local.points) == 0:
            continue

        all_point_clouds.append(pcd_local)
        successful += 1

    # Verificar resultados
    if successful == 0:
        print("\nNo se generaron nubes válidas.")
        return None

    # Combinar nubes
    pcd_combined = fusionar_nubes(all_point_clouds)

    # Filtrado final opcional de planos
    if filtrar_planos and len(pcd_combined.points) > 1000:
        plane_model, inliers = pcd_combined.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )
        pcd_combined = pcd_combined.select_by_index(inliers, invert=True)

    # Visualizar resultado
    ref = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(
        [pcd_combined, ref],
        window_name=f"Reconstrucción 3D ({successful} imágenes)",
        width=1280, height=720
    )

    # Guardar PLY
    out_path = os.path.join(base_dir, f"reconstruccion_alineada_{successful}imgs.ply")
    guardar_ply(pcd_combined, out_path)
    print(f"\nReconstrucción completada: {successful} imágenes válidas.")
    print(f"Guardado: {out_path}")

    return pcd_combined


# ----------------------- Utilidades de transformación -----------------------

def create_transform_matrix(rvec, tvec):
    """
    Crea una matriz de transformación homogénea 4x4 a partir de un vector de rotación y un vector de traslación.

    Args:
        rvec (np.ndarray): Vector de rotación (Rodrigues) de forma (3, 1) o (1, 3).
        tvec (np.ndarray): Vector de traslación de forma (3, 1) o (1, 3).

    Devuelve:
        np.ndarray: Matriz de transformación homogénea 4x4 (pose de cámara en coordenadas del mundo).
    """
    R, _ = cv2.Rodrigues(rvec)  # convertir a matriz de rotación 3x3
    T = np.eye(4, dtype=np.float64)  # inicializar matriz homogénea
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


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

def filtrar_buda_bbox(
    nube: o3d.geometry.PointCloud,
    tablero_cols: int = 11,
    tablero_rows: int = 8,
    square_size_m: float = 0.0242,
    bbox_range_z: float = 0.600,  
    margin: float = 0.080, 
    auto_centrar: bool = True,
    verbose: bool = True
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.AxisAlignedBoundingBox]:
    """
    Filtra la nube de puntos para extraer el Buda mediante una bounding box alineada a ejes.

    Centra la caja en el tablero (por dimensiones conocidas) y opcionalmente auto-centra en XY
    según el rango actual de la nube. Limita en Z con un rango simétrico alrededor del plano del tablero.

    Args:
        nube (o3d.geometry.PointCloud): Nube de puntos completa (tablero + Buda + fondo).
        tablero_cols (int): Número de columnas de esquinas internas del tablero (p. ej., 11).
        tablero_rows (int): Número de filas de esquinas internas del tablero (p. ej., 8).
        square_size_m (float): Tamaño del lado del cuadrado del tablero en metros (p. ej., 0.0242).
        bbox_range_z (float): Rango simétrico en Z (±) desde el plano del tablero en metros (p. ej., 0.600).
        margin (float): Margen adicional en X e Y alrededor del tablero en metros (p. ej., 0.080).
        auto_centrar (bool): Indica si centra la bbox en el punto medio de la nube en XY.
        verbose (bool): Indica si imprime información diagnóstica del filtrado.

    Devuelve:
        Tuple[o3d.geometry.PointCloud, o3d.geometry.AxisAlignedBoundingBox]:
            Nube filtrada y bounding box utilizada.
    """
    # validar nube no vacía
    if not nube.has_points():
        raise ValueError("La nube de puntos está vacía")

    # obtener puntos como ndarray
    puntos = np.asarray(nube.points)

    # calcular dimensiones del tablero en el mundo
    width_x = (tablero_cols - 1) * square_size_m
    height_y = (tablero_rows - 1) * square_size_m

    # imprimir información si corresponde
    if verbose:
        print(f"Dimensiones del tablero: {width_x*1000:.1f} x {height_y*1000:.1f} mm")
        print(f"Rango Z: ±{bbox_range_z*1000:.1f} mm")
        print(f"Margen XY: {margin*1000:.1f} mm")
        print(f"\nRango actual de la nube:")
        print(f"  X: [{puntos[:, 0].min()*1000:.1f}, {puntos[:, 0].max()*1000:.1f}] mm")
        print(f"  Y: [{puntos[:, 1].min()*1000:.1f}, {puntos[:, 1].max()*1000:.1f}] mm")
        print(f"  Z: [{puntos[:, 2].min()*1000:.1f}, {puntos[:, 2].max()*1000:.1f}] mm")

    # definir límites de la bbox
    if auto_centrar:
        # calcular centro de la nube en XY
        centro_x = (puntos[:, 0].min() + puntos[:, 0].max()) / 2
        centro_y = (puntos[:, 1].min() + puntos[:, 1].max()) / 2
        z_min = puntos[:, 2].min()

        # construir límites centrados en el punto medio de la nube
        min_bound = np.array([
            centro_x - width_x/2 - margin,
            centro_y - height_y/2 - margin,
            z_min - bbox_range_z
        ])
        max_bound = np.array([
            centro_x + width_x/2 + margin,
            centro_y + height_y/2 + margin,
            z_min + bbox_range_z
        ])

        # informar centro detectado
        if verbose:
            print(f"\nCentro detectado: [{centro_x*1000:.1f}, {centro_y*1000:.1f}] mm")
    else:
        # usar bbox centrada en el origen (comportamiento original)
        min_bound = np.array([-margin, -margin, -bbox_range_z])
        max_bound = np.array([width_x + margin, height_y + margin, bbox_range_z])

    # informar límites de la bbox
    if verbose:
        print(f"\nBounding Box:")
        print(f"  Min: [{min_bound[0]*1000:.1f}, {min_bound[1]*1000:.1f}, {min_bound[2]*1000:.1f}] mm")
        print(f"  Max: [{max_bound[0]*1000:.1f}, {max_bound[1]*1000:.1f}, {max_bound[2]*1000:.1f}] mm")

    # crear bounding box alineada a ejes
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = (1, 0, 0)  # asignar color rojo

    # recortar nube con la bbox
    nube_filtrada = nube.crop(bbox)

    return nube_filtrada, bbox
