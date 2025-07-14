import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
import logging
import sys
import os
ROOT_DIR =os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup
import open3d as o3d
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)
from groundingdino.util.inference import Model
from PIL import Image
# -------------------- 模型加载 --------------------
CONFIG_PATH = "/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/config/groundingdino_swint_ogc.py"
DINO_CHECKPOINT = "/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/weights/groundingdino_swint_ogc.pth"
# 初始化Grounding DINO
grounding_dino_model = Model(
    model_config_path=CONFIG_PATH,
    model_checkpoint_path=DINO_CHECKPOINT,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)


# def choose_model():
#     """Initialize SAM predictor with proper parameters"""
#     model_weight = 'sam_b.pt'
#     overrides = dict(
#         task='segment',
#         mode='predict',
#         #imgsz=1024,
#         model=model_weight,
#         conf=0.01,
#         save=False
#     )
#     return SAMPredictor(overrides=overrides)

def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/weights/sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.1,
        save=False
    )
    return SAMPredictor(overrides=overrides)


def set_classes(model, target_class):
    """Set YOLO-World model to detect specific class"""
    model.set_classes([target_class])

def detect_objects_dinoground(image, text_prompt: str, plot_bbox: bool = True):
    detections, labels = grounding_dino_model.predict_with_caption(
        image=image,
        caption=text_prompt,
        box_threshold=0.25,
        text_threshold=0.25
    )
    if len(detections) == 0:
        print("⚠️ 没有检测到目标")
        return image, None
    valid_boxes = []
    # 提取检测结果
    for detctetion in detections:
        valid_boxes.append(
            {"xyxy": detctetion[0],
             "conf": detctetion[2],
             "cls": labels[0]})
    draw_image = image.copy()
    # 在原图上绘制bbox
    if plot_bbox:
        for box in valid_boxes:
            x1, y1, x2, y2 = map(int, box['xyxy'])

            # 画框
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 写标签
            label_text = f"{box['cls']}: {box['conf']:.2f}"
            cv2.putText(draw_image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(filename='/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/doc/image/detection.jpg', img=draw_image)
        cv2.imshow("DINO", draw_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    return valid_boxes, draw_image


def detect_objects(image_or_path, target_class=None):
    """
    Detect objects with YOLO-World
    image_or_path: can be a file path (str) or a numpy array (image).
    Returns: (list of bboxes in xyxy format, detected classes list, visualization image)
    """
    model = YOLO("yolov8s-world.pt")
    if target_class:
        set_classes(model, target_class)
    # YOLOv8 的 predict 可同时处理 文件路径(str) 或 图像数组(np.ndarray)
    results = model.predict(image_or_path)
    boxes = results[0].boxes
    vis_img = results[0].plot()  # Get visualized detection results
    # Extract valid detections
    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.25:  # Confidence threshold 0.25
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": results[0].names[box.cls.item()]
            })

    return valid_boxes, vis_img


def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def segment_image(image_path, output_mask='mask1.png'):
    """
    image_path: can be either a file path (str) or a numpy array (BGR image).
    output_mask: output mask file name.
    1) 用户可决定是否检测特定类别
    2) 调用 detect_objects 做初步检测
    3) 若 detections 存在，自动选最高分；否则让用户点击选择
    4) 用 SAM 分割并保存结果掩码
    5) 返回分割后的 mask (np.ndarray or None)
    """
    # 1) 用户输入 - 是否检测特定类别
    # use_target_class = input("Detect specific class? (yes/no): ").lower() == 'yes'
    # target_class = input("Enter class name: ").strip() if use_target_class else None
    # 用户直接输入类别
    target_class = input("\n"
                         "===============\n"
                         "Enter class name: ").strip()

    # 2) 初步检测 - YOLO
    detections, vis_img = detect_objects(image_path, target_class)

    # 保存检测可视化结果
    cv2.imwrite('detection_visualization.jpg', vis_img)

    # 3) 准备给 SAM 的图像 (RGB 格式)
    if isinstance(image_path, str):
        # 如果是字符串，说明是图像路径
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        # 否则假设 image_path 就是一个 BGR 的 numpy 数组
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

    # 4) 初始化 SAM predictor
    predictor = choose_model()
    predictor.set_image(image_rgb)

    # 5) 判断是否有目标检测结果
    if detections:
        # 自动选最高置信度
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        # 手动点击
        cv2.imshow('Select Object', vis_img)
        # 初始化全局变量
        point = []
        clicked = False

        def click_handler(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                clicked = True  # 标记点击完成

        cv2.setMouseCallback('Select Object', click_handler)
        # 循环等待点击或ESC键
        while not clicked:
            key = cv2.waitKey(10)  # 10ms延迟，减少CPU占用
            if key == 27:  # ESC键退出
                raise ValueError("User cancelled selection")
        cv2.destroyAllWindows()  # 安全关闭窗口

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    # 6) 保存 mask
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    else:
        print("[WARNING] Could not generate mask")

    return mask

# 相机内参（使用深度相机参数生成点云，根据实际相机修改）
# DEPTH_INTR = {
#     "ppx": 319.304,  # cx
#     "ppy": 236.915,  # cy
#     "fx": 387.897,  # fx
#     "fy": 387.897  # fy
# }
# DEPTH_FACTOR = 1000.0  # 深度因子，根据实际数据调整

def get_and_process_data(color_path, depth_path, mask_path,DEPTH_INTR,DEPTH_FACTOR):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
    NUM_POINT = 400
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color = color[..., [2, 1, 0]]
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")
    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    print("\n=== 尺寸验证 ===")
    print("深度图尺寸:", depth.shape[::-1])
    print("颜色图尺寸:", color.shape[:2][::-1])
    print("工作空间尺寸:", workspace_mask.shape[::-1])
    print("相机参数预设尺寸:", (DEPTH_INTR['width'], DEPTH_INTR['height']))
    camera = CameraInfo(
        width=DEPTH_INTR['width'],
        height=DEPTH_INTR['height'],
        fx=DEPTH_INTR['fx'],
        fy=DEPTH_INTR['fy'],
        cx=DEPTH_INTR['ppx'],
        cy=DEPTH_INTR['ppy'],
        scale=DEPTH_FACTOR
    )
    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': cloud_sampled}
    return end_points, cloud_o3d
# ==================== 碰撞检测 ====================
def collision_detection(gg, cloud_points,COLLISION_THRESH=0.02,VOXEL_SIZE=0.01):
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]

if __name__ == '__main__':
    seg_mask = segment_image('color_img_path.jpg')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
