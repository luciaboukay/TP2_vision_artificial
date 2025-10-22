import cv2
import numpy as np
import pickle
import open3d as o3d
import os
import glob

DICTIONARY_TYPE = cv2.aruco.DICT_6X6_250
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 52.6
MARKER_LENGTH = 31.3

aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY_TYPE)
charuco_board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH / 1000.0,
    MARKER_LENGTH / 1000.0,
    aruco_dict
)

def detect_charuco_markers(image, board, detector_params=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if not detector_params:
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.adaptiveThreshWinSizeMin = 3
        detector_params.adaptiveThreshWinSizeMax = 23
        detector_params.adaptiveThreshWinSizeStep = 10
        detector_params.minMarkerPerimeterRate = 0.03
        detector_params.maxMarkerPerimeterRate = 4.0

    detector = cv2.aruco.ArucoDetector(board.getDictionary(), detector_params)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    ids = ids.flatten()
    order = np.argsort(ids)
    ids = ids[order]
    corners = [corners[i] for i in order]

    return {'corners': corners, 'ids': ids, 'rejected': rejected}

def estimate_camera_pose_with_homography(image, board, detection, calibration, undistort=False):
    corners = detection['corners']
    ids = detection['ids']
    K, dist = calibration

    if len(ids) < 4:
        return None

    image_points = []
    board_points = []

    board_ids = board.getIds()
    points3d = board.getObjPoints()
    
    for corner, id in zip(corners, ids):
        id = int(id)
        if id not in board_ids:
            continue
        point3d = points3d[id]
        image_points.extend(corner[0])
        board_points.extend(point3d)

    image_points = np.array(image_points, dtype=np.float32)
    board_points = np.array(board_points, dtype=np.float32)

    if len(image_points) < 4:
        return None

    if undistort:
        use_image_points = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2), K, dist, P=K
        ).reshape(-1, 2)
    else:
        use_image_points = image_points

    H, inliers = cv2.findHomography(board_points[:, :2], use_image_points, method=cv2.LMEDS)

    if H is None:
        return None

    charuco_obj_points = []
    chess = board.getChessboardCorners()
    for i in range(chess.shape[0]):
        charuco_obj_points.append(chess[i])

    charuco_board_points = np.array(
        [p[:2] for p in charuco_obj_points], dtype=np.float32
    ).reshape(-1, 1, 2)

    projected_charuco_corners = cv2.perspectiveTransform(charuco_board_points, H)
    projected_charuco_corners = projected_charuco_corners.reshape(-1, 2)

    obj_pts = np.array(charuco_obj_points, dtype=np.float32)

    if undistort:
        use_image_points = cv2.undistortPoints(
            projected_charuco_corners.reshape(-1, 1, 2), K, dist, P=K
        ).reshape(-1, 2)
    else:
        use_image_points = projected_charuco_corners

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, use_image_points, K, dist, flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None

    return success, rvec, tvec

def detect_charuco_pose(image, camera_matrix, dist_coeffs, board, verbose=False):
    detection = detect_charuco_markers(image, board)
    
    if detection is None:
        if verbose:
            print("  ❌", end="")
        return False, None, None, None, None
    
    if verbose:
        print(f"  ✅{len(detection['ids'])}m", end="")
    
    result = estimate_camera_pose_with_homography(
        image, board, detection, (camera_matrix, dist_coeffs), undistort=False
    )
    
    if result is None:
        if verbose:
            print("❌", end="")
        return False, None, None, None, None
    
    success, rvec, tvec = result
    tvec_mm = tvec * 1000.0
    
    return success, rvec, tvec_mm, detection['corners'], detection['ids']


def create_transform_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec.flatten()
    return T


def filtrar_puntos_3d_mejorado(points_3d, disparity, max_distance_mm=1500.0, min_disparity=10.0):
    """
    ✅ FILTRADO AGRESIVO DE RUIDO
    """
    mask = np.ones(disparity.shape, dtype=bool)
    
    # 1. Disparidad válida
    mask &= (disparity > min_disparity) & (disparity < 200.0)
    
    # 2. Valores finitos
    mask &= np.isfinite(points_3d[:, :, 0])
    mask &= np.isfinite(points_3d[:, :, 1])
    mask &= np.isfinite(points_3d[:, :, 2])
    
    # 3. Distancia
    distances = np.linalg.norm(points_3d, axis=2)
    mask &= (distances < max_distance_mm) & (distances > 200.0)
    
    # 4. Z positivo y razonable
    mask &= (points_3d[:, :, 2] > 0) & (points_3d[:, :, 2] < max_distance_mm)
    
    # 5. X e Y razonables
    mask &= (np.abs(points_3d[:, :, 0]) < max_distance_mm * 0.7)
    mask &= (np.abs(points_3d[:, :, 1]) < max_distance_mm * 0.7)
    
    return mask


def crear_bounding_box_filtro(center, extent):
    bbox = o3d.geometry.OrientedBoundingBox(center, np.eye(3), extent)
    bbox.color = [1.0, 0.0, 0.0]
    return bbox


def filtrar_por_bounding_box(point_cloud, center, extent):
    bbox = crear_bounding_box_filtro(center, extent)
    idx = bbox.get_point_indices_within_bounding_box(point_cloud.points)
    idx = np.asarray(idx, dtype=np.int64)

    filtered_pc = o3d.geometry.PointCloud()
    pts = np.asarray(point_cloud.points)
    
    if idx.size == 0:
        return filtered_pc, bbox

    filtered_pts = pts[idx]
    filtered_pc.points = o3d.utility.Vector3dVector(filtered_pts)

    if point_cloud.has_colors():
        cols = np.asarray(point_cloud.colors)
        if cols.shape[0] == pts.shape[0]:
            filtered_cols = cols[idx]
            filtered_pc.colors = o3d.utility.Vector3dVector(filtered_cols)
        else:
            filtered_pc.paint_uniform_color([0.6, 0.6, 0.6])
    else:
        filtered_pc.paint_uniform_color([0.6, 0.6, 0.6])

    return filtered_pc, bbox


def reconstruccion_3d_muneca(max_distance_mm=1400.0, voxel_size_mm=1.0, 
                              outlier_nb_neighbors=35, outlier_std_ratio=1.2):
    """
    ✅ TU CÓDIGO ORIGINAL - SOLO QUITANDO CONVERSIÓN A GRISES
    """
    base_dir = "imgs"
    calib_path = os.path.join(base_dir, "calibracion/stereo_calibration.pkl")
    maps_path = os.path.join(base_dir, "calibracion/stereo_maps.pkl")
    
    with open(calib_path, "rb") as f:
        calib = pickle.load(f)
    with open(maps_path, "rb") as f:
        maps = pickle.load(f)
    
    left_K = calib["left_K"]
    left_dist = calib["left_dist"]
    Q = maps["Q"]
    
    objeto_dir = os.path.join(base_dir, "objeto")
    disparity_dir = os.path.join(objeto_dir, "disparidad_cre")
    left_imgs_original = sorted(glob.glob(os.path.join(objeto_dir, "left_*.jpg")))
    
    all_point_clouds = []
    successful_reconstructions = 0
    
    print("🔹 Reconstrucción 3D EN COLOR")
    print(f"   max={max_distance_mm/1000:.1f}m, voxel={voxel_size_mm}mm")
    print(f"   outliers: n={outlier_nb_neighbors}, σ={outlier_std_ratio}\n")
    
    for i, left_img_path in enumerate(left_imgs_original):
        print(f"[{i+1:2d}/{len(left_imgs_original)}] {os.path.basename(left_img_path)[:12]}", end=" ")
        
        left_img_original = cv2.imread(left_img_path)
        if left_img_original is None:
            print("❌")
            continue
        
        success, rvec, tvec_mm, corners, ids = detect_charuco_pose(
            left_img_original, left_K, left_dist, charuco_board, verbose=True
        )
        
        if not success:
            print()
            continue
        
        C_T_O = create_transform_matrix(rvec, tvec_mm)
        O_T_C = np.linalg.inv(C_T_O)
        
        R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        T_flip = np.eye(4)
        T_flip[:3, :3] = R_flip
        O_T_C_corrected = T_flip @ O_T_C
        
        base_name = os.path.basename(left_img_path).replace("left_", "").replace(".jpg", "")
        disp_raw_path = os.path.join(disparity_dir, f"disp_raw_{base_name}.npy")
        
        if not os.path.exists(disp_raw_path):
            print(" ❌")
            continue
        
        disparity = np.load(disp_raw_path)
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        mask = filtrar_puntos_3d_mejorado(points_3d, disparity, max_distance_mm, min_disparity=10.0)
        valid_points = points_3d[mask]
        
        if len(valid_points) < 1000:
            print(f" ❌")
            continue
        
        # ✅ ÚNICA DIFERENCIA: NO CONVERTIR A GRISES
        img_left_color = cv2.imread(left_img_path)
        img_left_color = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB)  # ← COLOR, no grises

        if img_left_color.shape[:2] != disparity.shape:
            img_left_color = cv2.resize(img_left_color, 
                                       (disparity.shape[1], disparity.shape[0]), 
                                       interpolation=cv2.INTER_LINEAR)

        colors = img_left_color.reshape(-1, 3) / 255.0
        valid_colors = colors[mask.flatten()]
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(valid_points)
        point_cloud.colors = o3d.utility.Vector3dVector(valid_colors)  # ← COLOR
        
        points_camera = np.array(point_cloud.points)
        points_camera_homogeneous = np.column_stack([points_camera, np.ones(len(points_camera))])
        
        points_world_homogeneous = (O_T_C_corrected @ points_camera_homogeneous.T).T
        points_world = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3:4]
        
        point_cloud.points = o3d.utility.Vector3dVector(points_world)
        
        cl, ind = point_cloud.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors, 
            std_ratio=outlier_std_ratio
        )
        point_cloud = point_cloud.select_by_index(ind)
        
        point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size_mm)
        
        all_point_clouds.append(point_cloud)
        successful_reconstructions += 1
        
        print(f" → {len(point_cloud.points):,}pts")
    
    if len(all_point_clouds) == 0:
        print("\n❌ Sin puntos")
        return None
    
    print(f"\n🔗 Combinando {len(all_point_clouds)} nubes...")
    point_cloud_combined = o3d.geometry.PointCloud()

    all_points = []
    all_colors = []

    for pc in all_point_clouds:
        pts = np.asarray(pc.points)
        all_points.append(pts)

        if pc.has_colors() and len(pc.colors) == len(pc.points):
            all_colors.append(np.asarray(pc.colors))
        else:
            all_colors.append(np.full((pts.shape[0], 3), 0.7, dtype=np.float64))

    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)

    point_cloud_combined.points = o3d.utility.Vector3dVector(combined_points)
    point_cloud_combined.colors = o3d.utility.Vector3dVector(combined_colors)

    print(f"   Total: {len(combined_points):,} pts")
    
    print(f"🧹 Limpieza global 1...")
    cl, ind = point_cloud_combined.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    point_cloud_combined = point_cloud_combined.select_by_index(ind)
    print(f"   Después: {len(point_cloud_combined.points):,} pts")
    
    print(f"🧹 Limpieza global 2 (agresiva)...")
    cl, ind = point_cloud_combined.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    point_cloud_combined = point_cloud_combined.select_by_index(ind)
    print(f"   Final: {len(point_cloud_combined.points):,} pts")
    
    print(f"\n🎉 Éxito: {successful_reconstructions}/{len(left_imgs_original)}\n")
    
    # ✅ NO convertir a grises
    # point_cloud_gray = convertir_a_escala_grises(point_cloud_combined)  # ← COMENTADO
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [point_cloud_combined, coordinate_frame],  # ← EN COLOR
        window_name="Reconstrucción 3D EN COLOR",
        width=1280,
        height=720,
        point_show_normal=False
    )
    
    output_path = os.path.join(base_dir, "objeto/reconstruccion_3d/reconstruccion_muneca_color.ply")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, point_cloud_combined, write_ascii=False)
    print(f"💾 Guardado\n")
    
    return point_cloud_combined