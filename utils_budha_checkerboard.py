import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
import re
import open3d as o3d


def generar_puntos_objeto(checkerboard, square_size):
    objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def calibrar_stereo_checkerboard(base_dir, checkerboard, square_size):
    left_images  = sorted(glob.glob(os.path.join(base_dir, "calib_left*.jpg")))
    right_images = sorted(glob.glob(os.path.join(base_dir, "calib_right*.jpg")))
    assert len(left_images) == len(right_images) and len(left_images) > 0, \
        "Debe haber el mismo número de imágenes left/right y al menos una pareja."

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    objp = generar_puntos_objeto(checkerboard, square_size)
    objpoints, imgpointsL, imgpointsR = [], [], []

    for lp, rp in zip(left_images, right_images):
        imgL = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None: continue

        retL, cornersL = cv2.findChessboardCorners(imgL, checkerboard, flags=cb_flags)
        retR, cornersR = cv2.findChessboardCorners(imgR, checkerboard, flags=cb_flags)
        if not (retL and retR): continue

        cornersL = cv2.cornerSubPix(imgL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv2.cornerSubPix(imgR, cornersR, (11,11), (-1,-1), criteria)

        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

    print(f"Pares válidos: {len(objpoints)}")
    assert len(objpoints) > 0, "No se detectó el patrón en ninguna pareja."

    image_size = (imgL.shape[1], imgL.shape[0])  # (w, h)

    # Calibración individual
    rmsL, K1, d1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
    rmsR, K2, d2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)

    # Calibración estéreo (manteniendo intrínsecas)
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_st = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rmsS, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR,
        K1, d1, K2, d2, image_size,
        criteria=criteria_st, flags=flags
    )

    print(f"RMS left: {rmsL:.4f} | RMS right: {rmsR:.4f} | RMS stereo: {rmsS:.4f}")

    calib = {
        "left_K": K1, "left_dist": d1,
        "right_K": K2, "right_dist": d2,
        "R": R, "T": T, "E": E, "F": F,
        "image_size": image_size
    }
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "stereo_calibration.pkl"), "wb") as f:
        pickle.dump(calib, f)

    print("Guardado: stereo_calibration.pkl")
    return calib

def cargar_calibracion_stereo(calibration_path):
    """
    Carga la calibración estéreo desde un archivo pickle.

    Args:
        calibration_path (str): Ruta al archivo de calibración (stereo_calibration.pkl).

    Returns:
        dict: Diccionario con las matrices de calibración y parámetros asociados.
    """
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de calibración en: {calibration_path}. "
            "Ejecutar la función de calibración o ajustar 'base_dir' al directorio correcto."
        )

    with open(calibration_path, "rb") as file:
        calibration_data = pickle.load(file)

    return calibration_data

def graficar_matrices_calibracion(calibracion):
    """
    Visualiza las matrices principales de la calibración estéreo en una grilla 2x4,
    mostrando valores numéricos y barras de color individuales.

    Args:
        calibracion (dict): Diccionario con las matrices de calibración.
            Se esperan las claves: 'left_K', 'left_dist', 'right_K', 'right_dist', 
            'R', 'T', 'E', 'F'.
    """
    def plot_matriz(ax, nombre, matriz):
        im = ax.imshow(matriz, cmap="viridis", interpolation="nearest")
        n_rows, n_cols = matriz.shape
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j, i, f"{matriz[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if matriz[i, j] < (matriz.max() / 2) else "black",
                    fontsize=8
                )
        ax.set_title(nombre, fontsize=11, weight="bold")
        ax.set_xlabel("Columnas")
        ax.set_ylabel("Filas")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Barra de color individual ajustada a la matriz
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Parámetros de Calibración Estéreo", fontsize=14, weight="bold")

    for ax, (nombre, matriz) in zip(axes.flat, matrices.items()):
        plot_matriz(ax, nombre, matriz)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def inferir_image_size(calibracion, carpeta_busqueda, patron_left="calib_left*.jpg"):
    """
    Devuelve (width, height). Usa 'image_size' si está en el pickle; en caso contrario,
    infiere el tamaño a partir de una imagen izquierda.
    """
    if "image_size" in calibracion:
        return tuple(calibracion["image_size"])
    left_samples = sorted(glob.glob(os.path.join(carpeta_busqueda, patron_left)))
    if not left_samples:
        raise FileNotFoundError("No se encontraron imágenes izquierdas para inferir image_size.")
    img = cv2.imread(left_samples[0], cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"No se pudo leer: {left_samples[0]}")
    return (img.shape[1], img.shape[0])  # (w, h)


def calcular_rectificacion(calibracion, image_size, alpha=0):
    """
    Calcula R1, R2, P1, P2, Q y ROIs de rectificación a partir de la calibración.
    """
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
    Genera mapas (x,y) para remap de rectificación en ambas cámaras.
    """
    lmx, lmy = cv2.initUndistortRectifyMap(
        calibracion["left_K"], calibracion["left_dist"], R1, P1, image_size, cv2.CV_32FC1
    )
    rmx, rmy = cv2.initUndistortRectifyMap(
        calibracion["right_K"], calibracion["right_dist"], R2, P2, image_size, cv2.CV_32FC1
    )
    return lmx, lmy, rmx, rmy


def guardar_stereo_maps(destino_pkl, mapas):
    """
    Guarda en pickle los mapas y matrices de rectificación.
    """
    os.makedirs(os.path.dirname(destino_pkl), exist_ok=True)
    with open(destino_pkl, "wb") as f:
        pickle.dump(mapas, f)

def visualizar_rectificacion(
    left_path, right_path, lmx, lmy, rmx, rmy,
    step=40, line_color=(0, 255, 0), thickness=2,
    title="Rectificación estéreo"
):
    """
    Rectifica un par de imágenes y dibuja líneas epipolares claramente visibles.

    Args:
        left_path (str): Ruta de la imagen izquierda.
        right_path (str): Ruta de la imagen derecha.
        lmx, lmy, rmx, rmy (np.ndarray): Mapas de rectificación (x, y) para cada cámara.
        step (int): Espaciado vertical entre líneas epipolares.
        line_color (tuple): Color BGR de las líneas.
        thickness (int): Grosor de las líneas.
        title (str): Título del gráfico.
    """
    # Leer imágenes
    imgL = cv2.imread(left_path, cv2.IMREAD_COLOR)
    imgR = cv2.imread(right_path, cv2.IMREAD_COLOR)
    if imgL is None or imgR is None:
        raise RuntimeError("No se pudieron leer las imágenes para visualización.")

    # Aplicar rectificación
    Lr = cv2.remap(imgL, lmx, lmy, cv2.INTER_LINEAR)
    Rr = cv2.remap(imgR, rmx, rmy, cv2.INTER_LINEAR)

    # Unir imágenes lado a lado
    vis = np.hstack((Lr, Rr))

    # Dibujar líneas epipolares con sombra para mejorar visibilidad
    for y in range(0, vis.shape[0], step):
        # línea principal
        cv2.line(vis, (0, y), (vis.shape[1], y), line_color, thickness)

    # Mostrar resultado
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title, fontsize=13, weight="bold")
    plt.show()

def cargar_stereo_maps(ruta_maps_pkl):
    with open(ruta_maps_pkl, "rb") as f:
        return pickle.load(f)


def emparejar_pares(carpeta_capturas):
    """
    Devuelve una lista ordenada de pares (left_path, right_path) emparejados por índice numérico
    o por coincidencia de nombre. Acepta .jpg y .png.
    """
    rutas = sorted(
        glob.glob(os.path.join(carpeta_capturas, "*.jpg")) +
        glob.glob(os.path.join(carpeta_capturas, "*.png"))
    )

    def parse_side_and_key(fname):
        name = os.path.basename(fname).lower()
        if "left" in name or re.search(r"(^|[_\-])l([_\-]|\d|\.|$)", name):
            side = "L"
        elif "right" in name or re.search(r"(^|[_\-])r([_\-]|\d|\.|$)", name):
            side = "R"
        else:
            side = None
        m = re.search(r"(\d+)", name)
        key = m.group(1) if m else name
        return side, key

    L_dict, R_dict = {}, {}
    for p in rutas:
        side, key = parse_side_and_key(p)
        if side == "L":
            L_dict.setdefault(key, p)
        elif side == "R":
            R_dict.setdefault(key, p)

    keys = sorted(set(L_dict) & set(R_dict))
    if not keys:
        raise RuntimeError(f"No se pudieron emparejar capturas en {carpeta_capturas}. Verificar nombres L/R.")

    pares = [(L_dict[k], R_dict[k]) for k in keys]
    return pares


def interseccion_rois(roi1, roi2):
    """
    Calcula la intersección de dos ROIs (x, y, w, h). Devuelve (xI, yI, wI, hI).
    """
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    xI, yI = max(x1, x2), max(y1, y2)
    xA, yA = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if not (xI < xA and yI < yA):
        raise ValueError("Las ROIs no se intersectan; revisar calibración.")
    return xI, yI, (xA - xI), (yA - yI)


def rectificar_y_guardar_pares(
    pares, left_map_x, left_map_y, right_map_x, right_map_y,
    roi_interseccion, carpeta_salida, indice_inicial=1, interpolacion=cv2.INTER_LINEAR
):
    """
    Rectifica y recorta a la ROI común cada par de imágenes a color, guardando PNGs.
    Retorna la cantidad de pares procesados.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    xI, yI, wI, hI = roi_interseccion

    count = 0
    for i, (lp, rp) in enumerate(pares, start=indice_inicial):
        Lc = cv2.imread(lp, cv2.IMREAD_COLOR)
        Rc = cv2.imread(rp, cv2.IMREAD_COLOR)
        if Lc is None or Rc is None:
            print("Error leyendo:", lp, "o", rp)
            continue

        Lr = cv2.remap(Lc, left_map_x,  left_map_y,  interpolacion)
        Rr = cv2.remap(Rc, right_map_x, right_map_y, interpolacion)

        Lr_roi = Lr[yI:yI + hI, xI:xI + wI]
        Rr_roi = Rr[yI:yI + hI, xI:xI + wI]

        cv2.imwrite(os.path.join(carpeta_salida, f"rect_left_color_{i:04d}.png"),  Lr_roi)
        cv2.imwrite(os.path.join(carpeta_salida, f"rect_right_color_{i:04d}.png"), Rr_roi)
        count += 1

    return count

# ----------------------- Carga y utilidades -----------------------

def cargar_calibracion(base_dir):
    with open(os.path.join(base_dir, "stereo_calibration.pkl"), "rb") as f:
        calib = pickle.load(f)
    return calib

def cargar_maps(base_dir):
    with open(os.path.join(base_dir, "stereo_maps.pkl"), "rb") as f:
        maps = pickle.load(f)
    return maps

def intrinseca_rectificada_y_Q(maps):
    P1 = maps["P1"]; Q = maps["Q"]
    roi1 = maps["validRoi1"]; roi2 = maps["validRoi2"]
    x1, y1, w1, h1 = roi1; x2, y2, w2, h2 = roi2
    xI, yI = max(x1, x2), max(y1, y2)
    K_rect = P1[:3, :3].copy()
    Q_adj = Q.copy()
    Q_adj[0, 3] -= float(xI)
    Q_adj[1, 3] -= float(yI)
    return K_rect, Q_adj, (xI, yI)

def generar_objpoints_grid(cols, rows, square_size_m):
    objp = np.zeros((cols*rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp

# ----------------------- Detectores intercambiables -----------------------

def detect_checkerboard(img_gray, pattern=(10,7), square_size_m=0.0242):
    ok, corners = cv2.findChessboardCorners(img_gray, pattern, None)
    if not ok:
        return False, None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
    objp = generar_objpoints_grid(pattern[0], pattern[1], square_size_m)
    return True, corners, objp

# Ejemplo de firma para Charuco (placeholder):
# def detect_charuco(img_gray, board, charuco_dict, cameraMatrix, distCoeffs):
#     # Deberías implementar: detectar, interpolar y alinear objp/corners
#     # Debe retornar: (ok, corners_2d (N,1,2), objp_3d (N,3))
#     return ok, corners2d, objp3d

# ----------------------- Nube desde disparidad -----------------------

def nube_desde_disparidad(disp, img_rgb, Q, *,
                          filtrar_dist=True, max_dist=0.8,
                          lim_xy=None):
    pts3d = cv2.reprojectImageTo3D(disp, Q)
    mask = np.isfinite(disp) & (disp > 0)
    mask &= np.isfinite(pts3d[:,:,0]) & np.isfinite(pts3d[:,:,1]) & np.isfinite(pts3d[:,:,2])
    if filtrar_dist:
        z = pts3d[:,:,2]
        dist = np.linalg.norm(pts3d, axis=2)
        mask &= (z > 0.05) & (z < max_dist) & (dist < max_dist)
    if lim_xy is not None:
        (xmin, xmax), (ymin, ymax) = lim_xy
        x = pts3d[:,:,0]; y = pts3d[:,:,1]
        mask &= (x>xmin)&(x<xmax)&(y>ymin)&(y<ymax)

    pts = pts3d[mask]; cols = img_rgb[mask]
    pcd = o3d.geometry.PointCloud()
    if len(pts) == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64)/255.0)
    return pcd

# ----------------------- Procesamiento por vista -----------------------

def procesar_vista(left_img_path, disp_path, K_rect, Q_adj, detect_fn):
    img_bgr  = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    disp = np.load(disp_path).astype(np.float32)

    ok, corners, objp = detect_fn(img_gray)
    if not ok:
        return None, None

    ok_pnp, rvec, tvec = cv2.solvePnP(
        objp, corners, K_rect, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok_pnp:
        return None, None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = tvec.flatten()

    # objeto->cámara (T), necesitamos cámara->objeto
    T_world_cam = np.linalg.inv(T)

    pcd_local = nube_desde_disparidad(
        disp, img_rgb, Q_adj,
        filtrar_dist=True, max_dist=0.8,
        lim_xy=((-0.3, 0.3), (-0.3, 0.3))
    )
    if len(pcd_local.points) == 0:
        return None, T_world_cam

    pcd_world = pcd_local.transform(T_world_cam)
    return pcd_world, T_world_cam

# ----------------------- Post-proceso nube -----------------------

def fusionar_nubes(lista_pcd):
    if not lista_pcd:
        return o3d.geometry.PointCloud()
    nube = lista_pcd[0]
    for p in lista_pcd[1:]:
        nube += p
    return nube

def filtrar_nube_global(pcd, *, usar_estadistico=True, nb=30, std=1.5, voxel=0.0):
    if usar_estadistico and len(pcd.points) > 100:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
        pcd = pcd.select_by_index(ind)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)
    return pcd

def guardar_ply(pcd, path):
    o3d.io.write_point_cloud(path, pcd)
    return path

def visualizar_vistas(pcd, poses, *, frame_size=0.1, cam_size=0.05, title="Reconstrucción"):
    ref = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cams = []
    for T in poses:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cam_size)
        cf.transform(T)
        cams.append(cf)
    o3d.visualization.draw_geometries([pcd, ref] + cams, window_name=title, width=1280, height=720)