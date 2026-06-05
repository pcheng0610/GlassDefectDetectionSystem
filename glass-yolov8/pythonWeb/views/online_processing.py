"""
在线处理API接口
专门处理在线功能的图片传输和缓存管理
与离线功能解耦，独立管理online_cache
"""
from flask import Blueprint, request, jsonify
from utils.local_cache import LocalCacheManager
from utils.model_pool import infer_image
from utils import db
from datetime import datetime
import os
import cv2
import uuid
import shutil
from typing import List, Dict

try:
    from utils.oss_client import upload_file_to_oss, get_image_url, upload_captured_image, upload_result_image
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False

# 蓝图对象
online_bp = Blueprint("online_processing", __name__)

# 初始化在线缓存管理器
online_cache_manager = LocalCacheManager(
    cache_dir=os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'online_cache'),
    images_dir=os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'online_cache', 'on_images'),
    results_dir=os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'online_cache', 'on_results'),
    cache_metadata_file='on_cache_metadata.json',
    results_metadata_file='on_results_metadata.json'
)

@online_bp.route('/api/online/add_images', methods=['POST'])
def add_online_images():
    """添加图片到在线缓存"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({"success": False, "error": "未收到图片文件"}), 400
        
        images_data = []
        filenames = []
        
        for file in files:
            if file and file.filename:
                file_data = file.read()
                images_data.append(file_data)
                filenames.append(file.filename)
        
        if not images_data:
            return jsonify({"success": False, "error": "没有有效的图片文件"}), 400
        
        # 添加到在线缓存
        results = online_cache_manager.add_images_batch(images_data, filenames, "upload")
        
        return jsonify({
            "success": True,
            "data": results,
            "message": f"成功添加 {len(results)} 张图片到在线缓存"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# 注意: /api/online/cache_status 和 /api/online/results_status 已移至 recognition.py
# 避免路由重复注册

@online_bp.route('/api/online/recognize', methods=['POST'])
def online_batch_recognize():
    """在线批量识别缓存中的图片"""
    try:
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])
        
        if not image_ids:
            return jsonify({"success": False, "error": "请指定要识别的图片ID"}), 400
        
        # 获取要识别的图片
        images = [online_cache_manager.get_image(img_id) for img_id in image_ids]
        images = [img for img in images if img is not None]
        
        if not images:
            return jsonify({"success": False, "error": "未找到要识别的图片"}), 400
        
        # 设置临时目录
        tmp_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        
        results = []
        
        for img_meta in images:
            try:
                # 更新状态为处理中
                online_cache_manager.update_image_status(img_meta["id"], "processing")
                
                # 进行识别（保存到tmp目录）
                best_class, best_confidence, vis_img = infer_image(img_meta["path"], tmp_dir, tmp_dir)
                confidence = round(float(best_confidence), 2)
                
                # 检查是否有结果图片
                stem, _ = os.path.splitext(img_meta["cached_filename"])
                result_image_name = f"{stem}_pred.jpg"
                result_image_path = os.path.join(tmp_dir, result_image_name)
                
                if os.path.exists(result_image_path):
                    # 确保结果目录存在
                    os.makedirs(online_cache_manager.results_dir, exist_ok=True)
                    
                    # 生成结果文件名
                    result_filename = f"result_{img_meta['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    result_cached_path = os.path.join(online_cache_manager.results_dir, result_filename)
                    
                    # 复制结果图片到在线缓存
                    shutil.copy2(result_image_path, result_cached_path)
                    
                    # 添加到结果缓存
                    result_meta = online_cache_manager.add_result(
                        image_id=img_meta["id"],
                        result_class=best_class,
                        confidence=confidence,
                        annotated_image_path=result_image_path,
                        result_filename=result_filename,
                        result_path=result_cached_path
                    )
                    
                    annotated_rel = f"/api/online/result/{result_filename}"
                else:
                    annotated_rel = None
                
                # 更新图片状态
                result_data = {
                    "result_class": best_class,
                    "confidence": confidence,
                    "annotated_image": annotated_rel
                }
                online_cache_manager.update_image_status(img_meta["id"], "completed", result_data)
                
                results.append({
                    "id": img_meta["id"],
                    "filename": img_meta["filename"],
                    "result_class": best_class,
                    "confidence": confidence,
                    "annotated_image": annotated_rel,
                    "status": "completed"
                })
                
            except Exception as e:
                print(f"识别图片失败 {img_meta['filename']}: {e}")
                online_cache_manager.update_image_status(img_meta["id"], "error")
                results.append({
                    "id": img_meta["id"],
                    "filename": img_meta["filename"],
                    "error": str(e),
                    "status": "error"
                })
        
        return jsonify({
            "success": True,
            "data": results,
            "message": f"在线识别完成，成功 {len([r for r in results if r.get('status') == 'completed'])} 张"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@online_bp.route('/api/online/upload_images', methods=['POST'])
def upload_online_images():
    """上传在线捕获图片到云端"""
    try:
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])
        
        if not image_ids:
            return jsonify({"success": False, "error": "请指定要上传的图片ID"}), 400
        
        # 获取要上传的图片
        images = [online_cache_manager.get_image(img_id) for img_id in image_ids]
        images = [img for img in images if img is not None]
        
        if not images:
            return jsonify({"success": False, "error": "未找到要上传的图片"}), 400
        
        upload_results = []
        
        for img_meta in images:
            try:
                # 读取图片数据
                img = cv2.imread(img_meta["path"])
                if img is None:
                    raise Exception("无法读取图片数据")
                
                # 上传到OSS云端 root/ 文件夹
                if OSS_AVAILABLE:
                    try:
                        print(f"[批量上传] 开始上传图片到云端: {img_meta['filename']}")
                        image_url = upload_captured_image(img, img_meta["filename"])
                        print(f"[批量上传] ✅ 云端上传成功! URL: {image_url}")
                        
                        # ⚠️ 注意：此 API 只上传捕获图（原图）到云端备份，不写入数据库
                        # 数据库写入应该在上传结果图时进行（使用 /api/online/upload_results）
                        # 或者使用前端的 /recognition/upload_result
                        
                        upload_results.append({
                            "id": img_meta["id"],
                            "filename": img_meta["filename"],
                            "image_url": image_url,
                            "uploaded_to_oss": True,
                            "status": "uploaded"
                        })
                        
                    except Exception as e:
                        print(f"[批量上传] ❌ OSS上传失败: {img_meta['filename']}, 错误: {e}")
                        
                        # ⚠️ 上传失败时，保存到root目录作为备用，但不写入数据库
                        root_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'root')
                        os.makedirs(root_dir, exist_ok=True)
                        filename_path = os.path.join(root_dir, img_meta["filename"])
                        cv2.imwrite(filename_path, img)
                        print(f"[批量上传] 图片已保存到root备用目录，未写入数据库: {img_meta['filename']}")
                        
                        upload_results.append({
                            "id": img_meta["id"],
                            "filename": img_meta["filename"],
                            "uploaded_to_oss": False,
                            "backup_path": filename_path,
                            "status": "upload_failed_backup",
                            "error": f"云端上传失败: {str(e)}"
                        })
                else:
                    # ⚠️ OSS不可用时，保存到root目录作为备用，但不写入数据库
                    print(f"[批量上传] ❌ OSS不可用")
                    root_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'root')
                    os.makedirs(root_dir, exist_ok=True)
                    filename_path = os.path.join(root_dir, img_meta["filename"])
                    cv2.imwrite(filename_path, img)
                    print(f"[批量上传] 图片已保存到root备用目录，未写入数据库: {img_meta['filename']}")
                    
                    upload_results.append({
                        "id": img_meta["id"],
                        "filename": img_meta["filename"],
                        "uploaded_to_oss": False,
                        "backup_path": filename_path,
                        "status": "oss_unavailable",
                        "error": "OSS云存储服务不可用"
                    })
                
            except Exception as e:
                print(f"上传图片失败 {img_meta['filename']}: {e}")
                upload_results.append({
                    "id": img_meta["id"],
                    "filename": img_meta["filename"],
                    "error": str(e),
                    "status": "upload_failed"
                })
        
        return jsonify({
            "success": True,
            "data": upload_results,
            "message": f"在线图片上传完成，成功 {len([r for r in upload_results if r.get('status') == 'uploaded'])} 张"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@online_bp.route('/api/online/upload_results', methods=['POST'])
def upload_online_results():
    """上传在线识别结果图片到云端"""
    try:
        data = request.get_json() or {}
        result_ids = data.get('result_ids', [])
        
        if not result_ids:
            return jsonify({"success": False, "error": "请指定要上传的结果ID"}), 400
        
        # 获取要上传的结果
        results = [online_cache_manager.get_result(result_id) for result_id in result_ids]
        results = [result for result in results if result is not None]
        
        if not results:
            return jsonify({"success": False, "error": "未找到要上传的结果"}), 400
        
        upload_results = []
        
        for result_meta in results:
            try:
                # 读取结果图片数据
                img = cv2.imread(result_meta["result_path"])
                if img is None:
                    raise Exception("无法读取结果图片数据")
                
                # 上传到OSS云端 result/ 文件夹
                if OSS_AVAILABLE:
                    try:
                        image_url = upload_result_image(img, result_meta["result_filename"])
                        print(f"结果图片已上传到云端OSS result/: {result_meta['result_filename']}")
                        
                        upload_results.append({
                            "id": result_meta["id"],
                            "result_filename": result_meta["result_filename"],
                            "image_url": image_url,
                            "uploaded_to_oss": True,
                            "status": "uploaded"
                        })
                        
                    except Exception as e:
                        print(f"上传到OSS失败: {result_meta['result_filename']}, 错误: {e}")
                        
                        # 上传失败时，保存到result目录作为备用
                        result_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'result')
                        os.makedirs(result_dir, exist_ok=True)
                        result_backup_path = os.path.join(result_dir, result_meta["result_filename"])
                        cv2.imwrite(result_backup_path, img)
                        print(f"上传失败，结果图片已保存到result备用目录: {result_meta['result_filename']}")
                        
                        upload_results.append({
                            "id": result_meta["id"],
                            "result_filename": result_meta["result_filename"],
                            "uploaded_to_oss": False,
                            "backup_path": result_backup_path,
                            "status": "upload_failed_backup"
                        })
                else:
                    # OSS不可用时，保存到result目录作为备用
                    result_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'result')
                    os.makedirs(result_dir, exist_ok=True)
                    result_backup_path = os.path.join(result_dir, result_meta["result_filename"])
                    cv2.imwrite(result_backup_path, img)
                    print(f"OSS不可用，结果图片已保存到result备用目录: {result_meta['result_filename']}")
                    
                    upload_results.append({
                        "id": result_meta["id"],
                        "result_filename": result_meta["result_filename"],
                        "uploaded_to_oss": False,
                        "backup_path": result_backup_path,
                        "status": "oss_unavailable_backup"
                    })
                
            except Exception as e:
                print(f"上传结果图片失败 {result_meta['result_filename']}: {e}")
                upload_results.append({
                    "id": result_meta["id"],
                    "result_filename": result_meta["result_filename"],
                    "error": str(e),
                    "status": "upload_failed"
                })
        
        return jsonify({
            "success": True,
            "data": upload_results,
            "message": f"在线结果图片上传完成，成功 {len([r for r in upload_results if r.get('status') == 'uploaded'])} 张"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# 注意: /api/online/clear_images 和 /api/online/clear_results 已移至 recognition.py
# 这里保留仅供 online_processing.py 内部使用，但不暴露为 API 端点
# 避免路由重复注册

@online_bp.route('/api/online/remove_image', methods=['POST'])
def remove_online_image():
    """删除指定在线图片"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        
        if not image_id:
            return jsonify({"success": False, "error": "请指定图片ID"}), 400
        
        success = online_cache_manager.remove_image(image_id)
        
        if success:
            return jsonify({"success": True, "message": "在线图片已删除"})
        else:
            return jsonify({"success": False, "error": "图片不存在"}), 404
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@online_bp.route('/api/online/remove_result', methods=['POST'])
def remove_online_result():
    """删除指定在线结果"""
    try:
        data = request.get_json()
        result_id = data.get('result_id')
        
        if not result_id:
            return jsonify({"success": False, "error": "请指定结果ID"}), 400
        
        success = online_cache_manager.remove_result(result_id)
        
        if success:
            return jsonify({"success": True, "message": "在线结果已删除"})
        else:
            return jsonify({"success": False, "error": "结果不存在"}), 404
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@online_bp.route('/api/online/save_to_database', methods=['POST'])
def save_online_to_database():
    """将在线识别结果保存到数据库"""
    try:
        data = request.get_json() or {}
        result_ids = data.get('result_ids', [])
        
        if not result_ids:
            return jsonify({"success": False, "error": "请指定要保存的结果ID"}), 400
        
        # 获取要保存的结果
        results = [online_cache_manager.get_result(result_id) for result_id in result_ids]
        results = [result for result in results if result is not None]
        
        if not results:
            return jsonify({"success": False, "error": "未找到要保存的结果"}), 400
        
        saved_count = 0
        
        for result_meta in results:
            try:
                # 获取对应的图片信息
                image_meta = online_cache_manager.get_image(result_meta["image_id"])
                if not image_meta:
                    continue
                
                # 保存到数据库
                now = datetime.now()
                create_time = now.strftime("%Y-%m-%d %H:%M:%S")
                
                # 使用短文件名存储到数据库，避免字段长度问题
                short_filename = image_meta["filename"]
                if len(short_filename) > 100:
                    short_filename = short_filename[:100]
                
                # 使用事务确保数据完整性
                try:
                    # 开始事务
                    db.connection.begin()
                    
                    # 插入在线检测结果
                    sql_online = 'insert into on_line (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                    # 插入历史记录
                    sql_history = 'insert into history (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                    
                    db.connection.insert(sql_online, (create_time, result_meta["result_class"], 'admin', short_filename))
                    db.connection.insert(sql_history, (create_time, result_meta["result_class"], 'admin', short_filename))
                    
                    # 提交事务
                    db.connection.commit()
                    print(f"在线结果保存成功，已写入数据库: {short_filename}")
                    
                except Exception as e:
                    # 回滚事务
                    db.connection.rollback()
                    print(f"数据库事务失败，已回滚: {e}")
                    raise e
                
                saved_count += 1
                
            except Exception as e:
                print(f"保存结果到数据库失败 {result_meta['id']}: {e}")
                continue
        
        return jsonify({
            "success": True,
            "message": f"成功保存 {saved_count} 条在线识别结果到数据库"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
