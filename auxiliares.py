import cv2
import os
import glob
import pickle
import numpy as np
from pathlib import Path

try:
    from stereodemo.config import models_path
except Exception:
    models_path = Path(__file__).parent / "models"
    models_path.mkdir(exist_ok=True)
    
from stereodemo.method_cre_stereo import CREStereo
from stereodemo.method_opencv_bm import StereoBM, StereoSGBM
from stereodemo.methods import InputPair, Config


# _______________________________ RECTIFICACIÓN _______________________________

def rectificar_imagenes(path_imgs, path_maps):
    """
    Rectificar todos los pares estéreo usando mapas precomputados y recortar al ROI común.

    Args:
        path_imgs (str): Carpeta con imágenes `left_*.jpg` y `right_*.jpg`.
        path_maps (str): Ruta a pickle con `left_map_x/y`, `right_map_x/y` y opcionalmente `validRoi1/2`.

    Devuelve:
        None
    """
    output_dir = os.path.join(path_imgs, "rectificadas")
    os.makedirs(output_dir, exist_ok=True)

    # Cargar mapas y ROIs válidos
    with open(path_maps, "rb") as f:
        maps = pickle.load(f)
    left_map_x, left_map_y   = maps["left_map_x"],  maps["left_map_y"]
    right_map_x, right_map_y = maps["right_map_x"], maps["right_map_y"]
    validRoi1 = maps.get("validRoi1", (0, 0, left_map_x.shape[1], left_map_x.shape[0]))
    validRoi2 = maps.get("validRoi2", (0, 0, right_map_x.shape[1], right_map_x.shape[0]))

    # Calcular intersección de ROIs
    def _intersect_roi(r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        xa, ya = max(x1, x2), max(y1, y2)
        xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if xb <= xa or yb <= ya:
            return (0, 0, left_map_x.shape[1], left_map_x.shape[0])
        return (xa, ya, xb - xa, yb - ya)

    roiX, roiY, roiW, roiH = _intersect_roi(validRoi1, validRoi2)

    # Buscar pares
    left_imgs  = sorted(glob.glob(os.path.join(path_imgs, "left_*.jpg")))
    right_imgs = sorted(glob.glob(os.path.join(path_imgs, "right_*.jpg")))

    if len(left_imgs) != len(right_imgs):
        print("Cantidad distinta de imágenes left/right")
        print(f"Left: {len(left_imgs)}, Right: {len(right_imgs)}")
        return

    for left_path, right_path in zip(left_imgs, right_imgs):
        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)

        # Remapear
        rectL = cv2.remap(imgL, left_map_x,  left_map_y,  cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, right_map_x, right_map_y, cv2.INTER_LINEAR)

        # Recortar a ROI común
        rectL = rectL[roiY:roiY+roiH, roiX:roiX+roiW]
        rectR = rectR[roiY:roiY+roiH, roiX:roiX+roiW]

        # Construir nombres de salida
        baseL = os.path.basename(left_path)
        baseR = os.path.basename(right_path)
        outL  = os.path.join(output_dir, f"rect_{baseL}")
        outR  = os.path.join(output_dir, f"rect_{baseR}")

        # Guardar resultados
        cv2.imwrite(outL, rectL)
        cv2.imwrite(outR, rectR)


# ___________________________ CÁLCULO DE DISPARIDAD ___________________________

def calcular_mapa_disparidad(imgL, imgR, build_matcher_fn, **kwargs):
    """
    Calcular mapa de disparidad con el matcher especificado conservando tamaño original.

    Args:
        imgL (np.ndarray): Imagen izquierda en escala de grises.
        imgR (np.ndarray): Imagen derecha en escala de grises.
        build_matcher_fn (Callable): Fábrica que devuelve un matcher (BM o SGBM).
        **kwargs: Parámetros para la fábrica del matcher.

    Devuelve:
        tuple[np.ndarray, np.ndarray]: (disparity_raw int16, disparity_vis uint8 normalizada)
    """
    stereo = build_matcher_fn(**kwargs)
    disparity_raw = stereo.compute(imgL, imgR)
    disparity_vis = cv2.normalize(disparity_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_raw, disparity_vis


def procesar_dataset_completo(path_rectificadas, output_path):
    """
    Procesar todos los pares rectificados, calcular disparidad y guardar crudo/visual.

    Args:
        path_rectificadas (str): Carpeta con `rect_left_*.jpg` y `rect_right_*.jpg`.
        output_path (str): Carpeta base de salida.

    Devuelve:
        None
    """
    path_crudo = os.path.join(output_path, 'crudo')
    path_visual = os.path.join(output_path, 'visual')
    os.makedirs(path_crudo, exist_ok=True)
    os.makedirs(path_visual, exist_ok=True)

    left_images = sorted(glob.glob(os.path.join(path_rectificadas, 'rect_left_*.jpg')))
    right_images = sorted(glob.glob(os.path.join(path_rectificadas, 'rect_right_*.jpg')))

    print(f"Se encontraron {len(left_images)} pares de imágenes.")

    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        print(f"Procesando par {i+1}/{len(left_images)}...")
        imgL = cv2.imread(left_path, 0)
        imgR = cv2.imread(right_path, 0)

        disparity_raw = calcular_mapa_disparidad(imgL, imgR)

        file_basename = os.path.basename(left_path).replace('rect_left_', '').replace('.jpg', '')
        
        np.save(os.path.join(path_crudo, f'disp_raw_{file_basename}.npy'), disparity_raw)

        disparity_visual = cv2.normalize(disparity_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity_color = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_PLASMA)
        cv2.imwrite(os.path.join(path_visual, f'disp_visual_{file_basename}.jpg'), disparity_color)

    print(f"Datos crudos: {path_crudo}")
    print(f"Imágenes visuales: {path_visual}")


def make_bm(num_disp=16*12, block=15, *,
            preFilterType=cv2.STEREO_BM_PREFILTER_XSOBEL,
            preFilterSize=9, preFilterCap=31,
            textureThreshold=10, uniquenessRatio=5,
            speckleRange=32, speckleWindowSize=100):
    """
    Crear matcher StereoBM de OpenCV.

    Args:
        num_disp (int): Rango de disparidad (múltiplo de 16).
        block (int): Tamaño de ventana impar en [5..51].
        preFilterType (int): Tipo de prefiltrado.
        preFilterSize (int): Tamaño del prefiltrado.
        preFilterCap (int): Límite del prefiltrado.
        textureThreshold (int): Umbral de textura.
        uniquenessRatio (int): Ratio de unicidad.
        speckleRange (int): Rango de speckle.
        speckleWindowSize (int): Ventana de speckle.

    Devuelve:
        cv2.StereoBM: Matcher configurado.
    """
    assert num_disp % 16 == 0, "num_disp debe ser múltiplo de 16"
    assert block % 2 == 1 and 5 <= block <= 51, "block debe ser impar en [5..51]"

    bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
    bm.setPreFilterType(preFilterType)
    bm.setPreFilterSize(preFilterSize)
    bm.setPreFilterCap(preFilterCap)
    bm.setTextureThreshold(textureThreshold)
    bm.setUniquenessRatio(uniquenessRatio)
    bm.setSpeckleRange(speckleRange)
    bm.setSpeckleWindowSize(speckleWindowSize)
    return bm


def make_sgbm(
    min_disp=0,
    num_disp=16*12,
    block=5,
    P1=None,
    P2=None,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
):
    """
    Crear matcher StereoSGBM configurable.

    Args:
        min_disp (int): Disparidad mínima.
        num_disp (int): Rango de disparidad.
        block (int): Tamaño de bloque (impar).
        P1 (int | None): Penalización P1 (auto si None).
        P2 (int | None): Penalización P2 (auto si None).
        disp12MaxDiff (int): Diferencia máx. disp12.
        uniquenessRatio (int): Ratio de unicidad.
        speckleWindowSize (int): Ventana de speckle.
        speckleRange (int): Rango de speckle.
        preFilterCap (int): Límite de prefiltro.
        mode (int): Modo de SGBM.

    Devuelve:
        cv2.StereoSGBM: Matcher configurado.
    """
    if P1 is None:
        P1 = 8 * 3 * block**2
    if P2 is None:
        P2 = 32 * 3 * block**2

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        preFilterCap=preFilterCap,
        mode=mode
    )
    return stereo


def sgbm_P1P2(block, channels=1):
    """
    Calcular penalizaciones P1/P2 estándar para SGBM.

    Args:
        block (int): Tamaño de bloque.
        channels (int): Cantidad de canales de imagen.

    Devuelve:
        tuple[int, int]: (P1, P2)
    """
    P1 = 8  * channels * (block ** 2)
    P2 = 32 * channels * (block ** 2)
    return P1, P2


def procesar_imagenes_rectificadas(path_imgs, out_dir, metodo="cre"):
    """
    Procesar imágenes rectificadas con stereodemo y guardar disparidades.

    Args:
        path_imgs (str): Carpeta con `rect_left_*.jpg` y `rect_right_*.jpg`.
        out_dir (str): Carpeta base de salida.
        metodo (str): 'cre' | 'bm' | 'sgbm'.

    Devuelve:
        None
    """
    config = Config(models_path=models_path)
    
    # Seleccionar método
    if metodo == "cre":
        method = CREStereo(config)
    elif metodo == "bm":
        method = StereoBM(config)
    else:
        method = StereoSGBM(config)

    # Preparar carpeta de salida
    out_dir = os.path.join(out_dir, f"disparidad_{metodo}")
    os.makedirs(out_dir, exist_ok=True)

    # Buscar pares rectificados
    left_imgs = sorted(glob.glob(os.path.join(path_imgs, "rect_left_*.jpg")))
    right_imgs = sorted(glob.glob(os.path.join(path_imgs, "rect_right_*.jpg")))

    print(f"Encontradas {len(left_imgs)} imágenes izquierdas")
    print(f"Encontradas {len(right_imgs)} imágenes derechas\n")

    for i, (left_path, right_path) in enumerate(zip(left_imgs, right_imgs)):
        imgL = cv2.imread(left_path)
        imgR = cv2.imread(right_path)
        
        pair = InputPair(
            left_image=imgL, 
            right_image=imgR, 
            calibration=None,  
            status=os.path.basename(left_path)
        )
        
        disparity = method.compute_disparity(pair)
        d = disparity.disparity_pixels.astype(np.float32)

        base = os.path.basename(left_path).replace("rect_left_", "").replace(".jpg", "")
        np.save(os.path.join(out_dir, f"disp_raw_{base}.npy"), d)

        d_min, d_max = np.nanmin(d), np.nanmax(d)
        if np.isfinite(d_min) and np.isfinite(d_max) and d_max > d_min:
            dvis = 255 * (d - d_min) / (d_max - d_min)
        else:
            dvis = np.zeros_like(d)
        dvis = np.clip(dvis, 0, 255).astype("uint8")

        dvis_color = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(out_dir, f"disp_gray_{base}.png"), dvis)
        cv2.imwrite(os.path.join(out_dir, f"disp_color_{base}.png"), dvis_color)

    cv2.destroyAllWindows()


def seleccionar_pares(path, n=3,
                      pattern_left="rect_left_color_*.png",
                      pattern_right="rect_right_color_*.png",
                      indices=None):
    """
    Seleccionar pares (L/R) por índice o tomar los primeros n.

    Args:
        path (str): Carpeta base donde buscar imágenes.
        n (int): Cantidad de pares por defecto si `indices` es None.
        pattern_left (str): Patrón glob para izquierda.
        pattern_right (str): Patrón glob para derecha.
        indices (list[int] | None): Índices específicos a seleccionar.

    Devuelve:
        tuple[list[int], list[tuple[str, str]], list[str], list[str]]:
            (pares_indices, pares_imgs, lefts, rights)
    """
    L = sorted(glob.glob(os.path.join(path, pattern_left)))
    R = sorted(glob.glob(os.path.join(path, pattern_right)))
    assert len(L) == len(R) and len(L) > 0, f"No hay pares válidos en {path}"

    if indices is None:
        k = min(n, len(L))
        pares_indices = list(range(k))
    else:
        pares_indices = [i for i in indices if 0 <= i < len(L)]
        assert pares_indices, "Los índices no son válidos."

    pares_imgs = [(L[i], R[i]) for i in pares_indices]
    lefts  = [l for l, _ in pares_imgs]
    rights = [r for _, r in pares_imgs]
    return pares_indices, pares_imgs, lefts, rights
