import os
from multiprocessing import Pool, get_start_method, set_start_method
from typing import Tuple, Optional

# Globals inside worker processes
MODEL = None
DEVICE = 'cpu'
NAMES = [
   '印痕', '光圈线',
        '划痕', '坑点',
        '有线', '气泡',
        '坑痕', '羽毛纹',
        '脱膜', '瑕疵点',
        '黑点'
]


_POOL: Optional[Pool] = None


def _init_worker(model_path: str):
    global MODEL, DEVICE
    
    print(f"[worker] 开始初始化，模型路径: {model_path}")
    
    # 验证模型文件在工作进程中也存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[worker] 模型文件不存在: {model_path}")
    
    # Lazy import in worker to avoid importing ultralytics in parent before fork/spawn
    from ultralytics import YOLO  # noqa: WPS433
    import torch  # noqa: WPS433

    # Backend optimizations
    try:
        torch.backends.cudnn.benchmark = True  # noqa: WPS437
    except Exception:
        pass

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[worker] 使用设备: {DEVICE}")

    try:
        MODEL = YOLO(model_path)
        print(f"[worker] 模型加载成功: {model_path}")
    except Exception as e:
        print(f"[worker] 模型加载失败: {e}")
        raise

    # Fuse Conv+BN for faster inference
    try:
        MODEL.fuse()
    except Exception:
        pass

    # channels_last may help on some GPUs
    try:
        core = getattr(MODEL, 'model', None)
        if core is not None:
            core.to(memory_format=torch.channels_last)
    except Exception:
        pass

    # torch.compile for extra speed (PyTorch 2+)
    try:
        core = getattr(MODEL, 'model', None)
        if core is not None and hasattr(torch, 'compile'):
            compiled = torch.compile(core, mode='max-autotune')  # type: ignore[attr-defined]
            setattr(MODEL, 'model', compiled)
    except Exception:
        # Fallback silently if compile not supported
        pass


def _worker_infer(image_path: str, tmp_dir: str, result_dir: str) -> Tuple[str, float, Optional[object]]:
    from PIL import Image  # noqa: WPS433

    if MODEL is None:
        raise RuntimeError("MODEL not initialized in worker")

    # Ensure output dirs exist (tmp_dir unused now but kept for signature compatibility)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Single-pass inference on the original image
    Image.open(image_path).convert("RGB")  # ensure readable; no save needed
    # Use GPU and FP16 when available for faster inference without accuracy drop
    use_half = DEVICE.startswith('cuda')
    results = MODEL(
        source=image_path,
        save=False,
        show=False,
        save_conf=True,
        conf=0.1,
        device=0 if use_half else 'cpu',
        half=use_half
    )

    best_confidence = -1.0
    best_class = None

    result = results[0]
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence_val = float(box.conf.item())
        if confidence_val > best_confidence:
            best_confidence = confidence_val
            best_class = NAMES[class_id]

    # 生成可视化图片数据，但不保存
    vis_img = None
    try:
        vis_img = result.plot()
    except Exception:
        # 忽略可视化错误，不影响推理
        pass

    if best_class is None:
        best_class = "No detection"
        best_confidence = 0.0

    return best_class, float(best_confidence), vis_img


def initialize_model_pool(model_path: str, processes: int = 1):
    global _POOL
    
    # 验证模型文件存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"[model_pool] 正在初始化，模型路径: {model_path}")
    
    # Ensure spawn on Windows to avoid issues
    try:
        if get_start_method(allow_none=True) != 'spawn':
            set_start_method('spawn', force=True)
            print("[model_pool] 已设置multiprocessing启动方法为spawn")
    except RuntimeError as e:
        # Start method already set elsewhere; ignore
        print(f"[model_pool] 启动方法设置警告: {e}")
        pass

    if _POOL is None:
        try:
            _POOL = Pool(processes=processes, initializer=_init_worker, initargs=(model_path,))
            print(f"[model_pool] 进程池创建成功，进程数: {processes}")
        except Exception as e:
            print(f"[model_pool] 进程池创建失败: {e}")
            raise


def infer_image(image_path: str, tmp_dir: str, result_dir: str) -> Tuple[str, float, Optional[object]]:
    if _POOL is None:
        raise RuntimeError("Model pool is not initialized. Call initialize_model_pool() first.")
    # Synchronous call; switch to apply_async if needed
    return _POOL.apply(_worker_infer, (image_path, tmp_dir, result_dir))


def shutdown_model_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None


