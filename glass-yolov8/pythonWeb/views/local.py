from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils import db
from utils.model_pool import infer_image  # type: ignore
from datetime import datetime
from PIL import Image, ExifTags
import os
import uuid
# OSS上传功能已移至batch_processing.py处理

#蓝图对象
lo = Blueprint( "local", __name__ )

@lo.route("/local")
def home():
    

    return render_template("local.html")


# 上传并识别接口：支持多文件
@lo.route('/api/local/upload', methods=['POST'])
def upload_and_detect():
    # 目录（与 result.py 对齐，使用已预加载的进程池模型）
    upload_dir = r"D:\zjzjzj\glass-yolov8\pythonWeb\static\data\local_cache\images"
    tmp_dir = r"D:\zjzjzj\glass-yolov8\pythonWeb\static\data\tmp"
    result_dir = r"D:\zjzjzj\glass-yolov8\pythonWeb\static\data\local_cache\results"

    os.makedirs(upload_dir, exist_ok=True)

    # 获取上传的文件，支持单文件和多文件
    files = request.files.getlist('files')
    if not files:
        # 如果没有files字段，尝试获取单个file字段
        single_file = request.files.get('file')
        files = [single_file] if single_file and single_file.filename else []

    if not files or all(not f.filename for f in files):
        return jsonify({"code": 400, "msg": "未收到图片文件"}), 400

    responses = []

    # EXIF 标签映射
    exif_dt_tags = {tag for tag, name in ExifTags.TAGS.items() if name in ("DateTimeOriginal", "DateTime")}

    for f in files:
        if not f or not f.filename:
            continue

        # 生成唯一文件名：uuid + off_line + 日期
        original_filename = secure_filename(f.filename)
        if not original_filename:
            continue

        # 获取文件扩展名
        _, ext = os.path.splitext(original_filename)
        if not ext:
            ext = '.jpg'  # 默认扩展名

        # 生成新的文件名：uuid_off_line_日期.扩展名
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # 取UUID的前8位
        filename = f"{unique_id}_off_line_{date_str}{ext}"

        print(f"原始文件名: {original_filename} -> 新文件名: {filename}")

        save_path = os.path.join(upload_dir, filename)

        # 直接保存到本地，不进行OSS上传
        # OSS上传和数据库写入统一由batch_processing.py处理
        try:
            f.save(save_path)
            print(f"文件已保存到本地: {filename}")
        except Exception as e:
            print(f"保存文件失败: {filename}, 错误: {e}")
            continue

        # 拍摄时间（优先 EXIF）
        capture_time = None
        try:
            with Image.open(save_path) as im:
                exif = getattr(im, '_getexif', lambda: None)()
                if exif:
                    for tag_id, value in exif.items():
                        if tag_id in exif_dt_tags and isinstance(value, str):
                            # EXIF 时间格式如 2023:09:10 12:34:56
                            capture_time = value.replace(':', '-', 2)
                            break
        except Exception:
            pass
        if not capture_time:
            capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 推理并保存带框图片到 static/data/result（使用预加载模型池）
        best_class = "推理失败"  # 默认值
        best_confidence = 0.0   # 默认值
        print(f"开始推理图片: {filename}")

        try:
            # 使用本地文件进行推理
            result = infer_image(save_path, tmp_dir, result_dir)
            print(f"推理结果: {result}")
            if result and len(result) >= 2:
                best_class, best_confidence, vis_img = result
                print(f"设置 best_class: {best_class}, best_confidence: {best_confidence}")
                
                # 保存可视化图片到result目录
                if vis_img is not None:
                    stem, _ = os.path.splitext(filename)
                    annotated_path = os.path.join(result_dir, f"{stem}_pred.jpg")
                    
                    try:
                        import cv2
                        cv2.imwrite(annotated_path, vis_img)
                        print(f"结果图片已保存到本地local: {stem}_pred.jpg")
                    except Exception as e:
                        print(f"保存结果图片失败: {e}")
        except Exception as e:
            print(f"YOLO推理失败: {filename}, 错误: {e}")
            # 保持默认值：推理失败

        # 确保 best_class 不为空且不为None
        print(f"验证前的 best_class: '{best_class}' (类型: {type(best_class)})")
        if best_class is None or not str(best_class).strip() or str(best_class).strip() == "No detection":
            best_class = "未识别"
            print(f"best_class 被设置为: '{best_class}'")
        else:
            best_class = str(best_class).strip()
            print(f"best_class 保持为: '{best_class}'")

        # 确保 best_confidence 是有效数值
        try:
            best_confidence = float(best_confidence) if best_confidence is not None else 0.0
        except (ValueError, TypeError):
            best_confidence = 0.0

        # 构造已保存的可视化结果路径：result_dir/<stem>_pred.jpg
        stem, _ = os.path.splitext(filename)
        annotated_filename = f"{stem}_pred.jpg"
        annotated_path = os.path.join(result_dir, annotated_filename)
        annotated_rel = None

        if os.path.exists(annotated_path):
            # 结果图片保留在本地
            annotated_rel = f"static/data/result/{annotated_filename}"
            print(f"结果图片已保存到本地: {annotated_filename}")

        confidence = round(float(best_confidence), 2)

        # 本地处理完成，直接写入数据库
        try:
            now = datetime.now()
            create_time = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # 使用本地路径作为数据库存储路径
            db_image_path = f"/static/data/result/{annotated_filename}"
            
            print(f"[数据库] 准备写入历史记录:")
            print(f"  - 时间: {create_time}")
            print(f"  - 类别: {best_class}")
            print(f"  - 用户: admin")
            print(f"  - 图片路径: {db_image_path}")
            
            # 插入历史记录
            sql_history = 'insert into history (create_time, result_class, user, image) values (%s, %s, %s, %s)'
            print(f"[数据库] 执行SQL: {sql_history}")
            
            db.insert(sql_history, (create_time, best_class, 'admin', db_image_path))
            print(f"[数据库] ✅ history表插入成功")
            
        except Exception as e:
            print(f"[数据库] 写入失败: {e}")
            import traceback
            traceback.print_exc()

        responses.append({
            "filename": filename,
            "capture_time": capture_time,
            "result_class": best_class,
            "confidence": confidence,
            "annotated_image": annotated_rel,  # 本地结果图片路径
            "local_path": save_path,  # 本地原始图片路径
            "result_path": annotated_path  # 本地结果图片完整路径
        })

    return jsonify({"code": 200, "data": responses})

def save_to_local_backup(image_path, filename, result_image_path=None, result_filename=None):
    """
    云端上传失败后的本地存储函数
    供batch_processing.py调用
    """
    try:
        # 定义目录路径
        uploads_dir = r"E:\DengHuiXiong\python\flash\pythonWeb\pythonWeb\static\data\uploads"
        result_dir = r"E:\DengHuiXiong\python\flash\pythonWeb\pythonWeb\static\data\result"
        
        # 保存原始图片到uploads目录
        os.makedirs(uploads_dir, exist_ok=True)
        uploads_image_path = os.path.join(uploads_dir, filename)
        
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            cv2.imwrite(uploads_image_path, img)
            print(f"原始图片保存到本地uploads目录: {filename}")
        else:
            print(f"无法读取图片: {image_path}")
            return False
        
        # 保存结果图片到result目录（如果有）
        if result_image_path and result_filename and os.path.exists(result_image_path):
            os.makedirs(result_dir, exist_ok=True)
            result_backup_path = os.path.join(result_dir, result_filename)
            
            import shutil
            shutil.copy2(result_image_path, result_backup_path)
            print(f"结果图片保存到本地result目录: {result_filename}")
        
        return True
        
    except Exception as e:
        print(f"本地存储失败: {e}")
        return False


# 获取置信度统计数据的API接口
@lo.route('/api/local/confidence_stats', methods=['GET'])
def get_confidence_stats():
    """
    获取各类别的平均置信度统计数据
    返回格式: {
        "success": true,
        "data": {
            "categories": ["类别1", "类别2", ...],
            "avg_confidences": [0.95, 0.87, ...],
            "stats_by_class": {
                "类别1": {"count": 10, "avg": 0.95, "max": 0.98, "min": 0.90},
                ...
            },
            "overall_avg": 0.91,
            "total_results": 25
        }
    }
    """
    try:
        # 从batch_processing缓存中读取结果
        from views.batch_processing import batch_manager
        
        results = batch_manager.get_all_results()
        
        if not results or len(results) == 0:
            return jsonify({
                "success": True,
                "data": {
                    "categories": [],
                    "avg_confidences": [],
                    "stats_by_class": {},
                    "overall_avg": 0,
                    "total_results": 0
                }
            })
        
        # 按类别统计
        stats_by_class = {}
        total_confidence = 0
        
        for result in results:
            result_class = result.get('result_class', '未知')
            confidence = result.get('confidence', 0)
            
            if result_class not in stats_by_class:
                stats_by_class[result_class] = {
                    'confidences': [],
                    'count': 0,
                    'total': 0
                }
            
            stats_by_class[result_class]['confidences'].append(confidence)
            stats_by_class[result_class]['count'] += 1
            stats_by_class[result_class]['total'] += confidence
            total_confidence += confidence
        
        # 计算平均值、最大值、最小值
        categories = []
        avg_confidences = []
        final_stats = {}
        
        for class_name, stats in stats_by_class.items():
            avg = stats['total'] / stats['count'] if stats['count'] > 0 else 0
            categories.append(class_name)
            avg_confidences.append(round(avg, 4))
            
            final_stats[class_name] = {
                'count': stats['count'],
                'avg': round(avg, 4),
                'max': round(max(stats['confidences']), 4) if stats['confidences'] else 0,
                'min': round(min(stats['confidences']), 4) if stats['confidences'] else 0
            }
        
        overall_avg = total_confidence / len(results) if len(results) > 0 else 0
        
        return jsonify({
            "success": True,
            "data": {
                "categories": categories,
                "avg_confidences": avg_confidences,
                "stats_by_class": final_stats,
                "overall_avg": round(overall_avg, 4),
                "total_results": len(results)
            }
        })
        
    except Exception as e:
        print(f"获取置信度统计失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "data": {
                "categories": [],
                "avg_confidences": [],
                "stats_by_class": {},
                "overall_avg": 0,
                "total_results": 0
            }
        })