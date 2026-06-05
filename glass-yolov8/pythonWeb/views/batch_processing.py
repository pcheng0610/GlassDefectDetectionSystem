"""
批量处理API接口
支持本地缓存的批量识别和上传
"""
from flask import Blueprint, request, jsonify
from utils.local_cache import cache_manager
from utils.model_pool import infer_image
from utils import db
from datetime import datetime
import os
from .local import save_to_local_backup
import cv2
import uuid
from typing import List, Dict



try:
    from utils.oss_client import upload_file_to_oss, get_image_url
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False

# 蓝图对象
batch_bp = Blueprint("batch_processing", __name__)

@batch_bp.route('/api/batch/add_images', methods=['POST'])
def add_images_to_cache():
    """添加图片到本地缓存"""
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
        
        # 添加到缓存
        results = cache_manager.add_images_batch(images_data, filenames, "upload")
        
        return jsonify({
            "success": True,
            "data": results,
            "message": f"成功添加 {len(results)} 张图片到本地缓存"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@batch_bp.route('/api/batch/add_online_images', methods=['POST'])
def add_online_images_to_cache():
    """添加图片到在线缓存"""
    try:
        from utils.local_cache import LocalCacheManager
        
        # 初始化在线缓存管理器
        online_cache_manager = LocalCacheManager(
            cache_dir=os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'online_cache'),
            images_dir=os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'online_cache', 'on_images'),
            results_dir=os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'online_cache', 'on_results'),
            cache_metadata_file='on_cache_metadata.json',
            results_metadata_file='on_results_metadata.json'
        )
        
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

@batch_bp.route('/api/batch/cache_status', methods=['GET'])
def get_cache_status():
    """获取缓存状态"""
    try:
        status = request.args.get('status')
        images = cache_manager.get_all_images(status)
        stats = cache_manager.get_cache_stats()
        
        return jsonify({
            "success": True,
            "data": {
                "images": images,
                "stats": stats
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@batch_bp.route('/api/batch/recognize', methods=['POST'])
def batch_recognize():
    """批量识别缓存中的图片"""
    try:
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])
        batch_id = data.get('batch_id')
        
        if not image_ids and not batch_id:
            return jsonify({"success": False, "error": "请指定要识别的图片ID或批次ID"}), 400
        
        # 获取要识别的图片
        if batch_id:
            images = cache_manager.get_all_images()
        else:
            images = [cache_manager.get_image(img_id) for img_id in image_ids]
            images = [img for img in images if img is not None]
        
        if not images:
            return jsonify({"success": False, "error": "未找到要识别的图片"}), 400
        
        # 设置临时目录
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data')
        tmp_dir = os.path.join(base_dir, 'tmp')
        result_dir = os.path.join(base_dir, 'result')
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        
        results = []
        
        for img_meta in images:
            try:
                # 更新状态为处理中
                cache_manager.update_image_status(img_meta["id"], "processing")
                
                # 进行识别
                best_class, best_confidence, vis_img = infer_image(img_meta["path"], tmp_dir, result_dir)
                confidence = round(float(best_confidence), 2)
                
                # 保存可视化图片到result目录
                stem, _ = os.path.splitext(img_meta["cached_filename"])
                annotated_filename = f"{stem}_pred.jpg"
                annotated_path = os.path.join(result_dir, annotated_filename)
                
                if vis_img is not None:
                    try:
                        import cv2
                        cv2.imwrite(annotated_path, vis_img)
                        print(f"结果图片已保存到本地: {annotated_filename}")
                    except Exception as e:
                        print(f"保存结果图片失败: {e}")
                
                if os.path.exists(annotated_path):
                    # 添加结果到results缓存
                    result_meta = cache_manager.add_result(
                        img_meta["id"], 
                        best_class, 
                        confidence, 
                        annotated_path
                    )
                    annotated_rel = f"pythonWeb/static/data/local_cache/results/{result_meta['result_filename']}"
                else:
                    annotated_rel = None
                
                # 更新图片状态
                result_data = {
                    "result_class": best_class,
                    "confidence": confidence,
                    "annotated_image": annotated_rel
                }
                cache_manager.update_image_status(img_meta["id"], "completed", result_data)
                
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
                cache_manager.update_image_status(img_meta["id"], "error")
                results.append({
                    "id": img_meta["id"],
                    "filename": img_meta["filename"],
                    "error": str(e),
                    "status": "error"
                })
        
        return jsonify({
            "success": True,
            "data": results,
            "message": f"批量识别完成，成功 {len([r for r in results if r.get('status') == 'completed'])} 张"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@batch_bp.route('/api/batch/upload', methods=['POST'])
def batch_upload():
    """批量上传识别结果到服务器"""
    try:
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])
        batch_id = data.get('batch_id')
        
        if not image_ids and not batch_id:
            return jsonify({"success": False, "error": "请指定要上传的图片ID或批次ID"}), 400
        
        # 获取要上传的图片
        if batch_id:
            images = cache_manager.get_batch_images(batch_id)
        else:
            images = [cache_manager.get_image(img_id) for img_id in image_ids]
            images = [img for img in images if img is not None]
        
        # 只上传已完成的图片
        completed_images = [img for img in images if img.get("status") == "completed"]
        
        if not completed_images:
            return jsonify({"success": False, "error": "没有已完成的识别结果可上传"}), 400
        
        upload_results = []
        
        for img_meta in completed_images:
            try:
                # 检查是否已经上传过，避免重复上传
                if img_meta.get("uploaded_to_oss", False):
                    print(f"图片 {img_meta['cached_filename']} 已经上传过，跳过重复上传")
                    continue
                
                # 先尝试上传到OSS
                uploaded_to_oss = False
                result_uploaded_to_oss = False
                db_image_path = None
                
                # 定义本地目录路径
                base_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'data')
                uploads_dir = os.path.join(base_dir, 'uploads')
                result_dir = os.path.join(base_dir, 'result')
                os.makedirs(uploads_dir, exist_ok=True)
                os.makedirs(result_dir, exist_ok=True)
                
                if OSS_AVAILABLE:
                    try:
                        # 1. 先尝试上传原始图片到OSS的uploads目录
                        upload_file_to_oss(img_meta["path"], "uploads", img_meta["cached_filename"])
                        print(f"原始图片上传成功到云端uploads目录: {img_meta['cached_filename']}")
                        uploaded_to_oss = True
                        
                        # 上传成功后，删除本地uploads目录中的原始图片
                        local_uploads_path = os.path.join(uploads_dir, img_meta["cached_filename"])
                        if os.path.exists(local_uploads_path):
                            os.remove(local_uploads_path)
                            print(f"云端上传成功，已删除本地原始图片: {img_meta['cached_filename']}")
                        
                        # 2. 尝试上传结果图片到OSS的result目录
                        if img_meta.get("annotated_image"):
                            # 检查缓存中的结果图片路径
                            cache_result_path = os.path.join(cache_manager.cache_dir, "results", 
                                                           os.path.basename(img_meta["annotated_image"]))
                            
                            # 检查本地result目录中的结果图片路径
                            stem, _ = os.path.splitext(img_meta["cached_filename"])
                            local_result_filename = f"{stem}_pred.jpg"
                            local_result_path = os.path.join(result_dir, local_result_filename)
                            
                            # 优先使用缓存中的结果图片，如果不存在则使用本地result目录中的
                            result_path_to_upload = cache_result_path if os.path.exists(cache_result_path) else local_result_path
                            result_filename_for_oss = os.path.basename(img_meta["annotated_image"]) if os.path.exists(cache_result_path) else local_result_filename
                            
                            if os.path.exists(result_path_to_upload):
                                upload_file_to_oss(result_path_to_upload, "result", result_filename_for_oss)
                                print(f"结果图片上传成功到云端result目录: {result_filename_for_oss}")
                                result_uploaded_to_oss = True
                                
                                # 上传成功后删除对应的本地文件
                                # 删除缓存中的结果图片
                                if os.path.exists(cache_result_path):
                                    os.remove(cache_result_path)
                                    print(f"云端上传成功，已删除缓存中的结果图片: {os.path.basename(img_meta['annotated_image'])}")
                                
                                # 删除本地result目录中的结果图片
                                if os.path.exists(local_result_path):
                                    os.remove(local_result_path)
                                    print(f"云端上传成功，已删除本地result目录中的结果图片: {local_result_filename}")
                        
                        # 3. 只有结果图片上传成功才执行数据库插入
                        if result_uploaded_to_oss:
                            # 如果有结果图片且上传成功，使用结果图片的云端地址
                            db_image_path = get_image_url("result", result_filename_for_oss)
                            print(f"数据库存储云端结果图片地址: {db_image_path}")
                        else:
                            # 如果没有结果图片，使用原始图片的云端地址
                            db_image_path = get_image_url("uploads", img_meta["cached_filename"])
                            print(f"数据库存储云端原始图片地址: {db_image_path}")
                        
                    except Exception as e:
                        print(f"上传到OSS失败: {e}")
                        # 上传失败，需要保存到本地作为备用
                        uploaded_to_oss = False
                        result_uploaded_to_oss = False
                else:
                    print("OSS不可用，需要保存到本地")
                    uploaded_to_oss = False
                    result_uploaded_to_oss = False
                
                # 如果上传失败或OSS不可用，调用local.py进行本地存储
                if not uploaded_to_oss or not result_uploaded_to_oss:

                    
                    # 准备结果图片路径
                    result_image_path = None
                    result_filename = None
                    if img_meta.get("annotated_image"):
                        result_image_path = os.path.join(cache_manager.cache_dir, "results", 
                                                       os.path.basename(img_meta["annotated_image"]))
                        result_filename = os.path.basename(img_meta["annotated_image"])
                    
                    # 调用local.py的本地存储函数
                    success = save_to_local_backup(
                        image_path=img_meta["path"],
                        filename=img_meta["cached_filename"],
                        result_image_path=result_image_path,
                        result_filename=result_filename
                    )
                    
                    if success:
                        print(f"云端上传失败，已通过local.py保存到本地: {img_meta['cached_filename']}")
                    else:
                        print(f"本地存储也失败: {img_meta['cached_filename']}")
                
                # 确定数据库存储路径
                if not db_image_path:
                    # 生成统一的结果图片文件名
                    stem, _ = os.path.splitext(img_meta["cached_filename"])
                    local_result_filename = f"{stem}_pred.jpg"
                    local_result_path = os.path.join(result_dir, local_result_filename)
                    
                    if os.path.exists(local_result_path):
                        # 如果本地result目录有结果图片，优先使用结果图片地址
                        db_image_path = f"static/data/result/{local_result_filename}"
                    elif img_meta.get("annotated_image") and os.path.exists(os.path.join(cache_manager.cache_dir, "results", os.path.basename(img_meta["annotated_image"]))):
                        # 如果缓存中有结果图片，使用缓存中的结果图片地址
                        db_image_path = f"static/data/result/{os.path.basename(img_meta['annotated_image'])}"
                    else:
                        # 否则使用原始图片地址
                        db_image_path = f"static/data/uploads/{img_meta['cached_filename']}"
                    print(f"使用本地备用地址: {db_image_path}")
                
                # 获取result_class用于后续使用
                result_class = img_meta.get("result_class") or img_meta.get("result") or "未识别"
                
                # 只有结果图片上传成功才执行数据库插入
                if result_uploaded_to_oss:
                    # 标记为已上传，避免重复上传
                    img_meta["uploaded_to_oss"] = True
                    
                    # 写入数据库
                    now = datetime.now()
                    create_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 使用短文件名存储到数据库，避免字段长度问题
                    short_filename = img_meta["cached_filename"]
                    if len(short_filename) > 100:  # 如果文件名太长，截取前100个字符
                        short_filename = short_filename[:100]
                    
                    # 使用事务确保数据完整性
                    try:
                        # 开始事务
                        db.connection.begin()
                        
                        # 插入离线检测结果
                        sql_offline = 'insert into off_line (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                        # 插入历史记录
                        sql_history = 'insert into history (create_time, result_class, user, image) values (%s, %s, %s, %s)'
                        
                        print(f"批量上传 - 图片: {img_meta['filename']}, result_class: '{result_class}'")
                        
                        db.connection.insert(sql_offline, (create_time, result_class, 'admin', db_image_path))
                        db.connection.insert(sql_history, (create_time, result_class, 'admin', db_image_path))
                        
                        # 提交事务
                        db.connection.commit()
                        print(f"批量上传成功，已写入数据库: {short_filename}")
                        
                    except Exception as e:
                        # 回滚事务
                        db.connection.rollback()
                        print(f"数据库事务失败，已回滚: {e}")
                        raise e
                else:
                    print(f"结果图片上传失败，跳过数据库插入: {img_meta['cached_filename']}")
                
                upload_results.append({
                    "id": img_meta["id"],
                    "filename": img_meta["filename"],
                    "result_class": result_class,
                    "confidence": img_meta["confidence"],
                    "uploaded_to_oss": uploaded_to_oss,
                    "db_path": db_image_path,
                    "status": "uploaded"
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
            "message": f"批量上传完成，成功 {len([r for r in upload_results if r.get('status') == 'uploaded'])} 张"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@batch_bp.route('/api/batch/clear_cache', methods=['POST'])
def clear_cache():
    """清空原始图片缓存（只清空images目录和cache_metadata.json，不影响results）"""
    try:
        print("收到清空images缓存请求")
        
        # 安全地获取JSON数据，避免解析错误
        try:
            data = request.get_json() or {}
        except Exception as json_error:
            print(f"JSON解析失败，使用默认值: {json_error}")
            data = {}
        
        status = data.get('status')  # 可选：只清空指定状态的图片
        print(f"清空images缓存参数: status={status}")
        
        cache_manager.clear_images_cache(status)
        
        return jsonify({
            "success": True,
            "message": f"原始图片缓存已清空" + (f"（状态: {status}）" if status else "") + "，结果图片已保留"
        })
    
    except Exception as e:
        print(f"清空images缓存API错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@batch_bp.route('/api/batch/clear_results', methods=['POST'])
def clear_results():
    """清空results缓存"""
    try:
        print("收到清空results缓存请求")
        
        cache_manager.clear_results_cache()
        
        return jsonify({
            "success": True,
            "message": "results缓存已清空"
        })
    
    except Exception as e:
        print(f"清空results缓存API错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@batch_bp.route('/api/batch/results_status', methods=['GET'])
def get_results_status():
    """获取results缓存状态"""
    try:
        results = cache_manager.get_all_results()
        stats = cache_manager.get_cache_stats()
        
        return jsonify({
            "success": True,
            "data": {
                "results": results,
                "stats": stats
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@batch_bp.route('/api/batch/remove_image', methods=['POST'])
def remove_image():
    """删除指定图片"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        
        if not image_id:
            return jsonify({"success": False, "error": "请指定图片ID"}), 400
        
        success = cache_manager.remove_image(image_id)
        
        if success:
            return jsonify({"success": True, "message": "图片已删除"})
        else:
            return jsonify({"success": False, "error": "图片不存在"}), 404
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@batch_bp.route('/api/batch/clean_orphaned_results', methods=['POST'])
def clean_orphaned_results():
    """清理result目录中的孤立文件（没有对应缓存记录的文件）"""
    try:
        result_dir = r"E:\DengHuiXiong\python\flash\pythonWeb\pythonWeb\static\data\result"
        
        if not os.path.exists(result_dir):
            return jsonify({"success": True, "message": "result目录不存在，无需清理"})
        
        # 获取result目录中的所有文件
        result_files = [f for f in os.listdir(result_dir) if f.endswith('_pred.jpg')]
        
        # 获取缓存中的所有结果记录
        cached_results = cache_manager.get_all_results()
        cached_filenames = {os.path.basename(result.get("result_path", "")) for result in cached_results}
        
        # 获取缓存中的所有图片记录，生成对应的结果文件名
        cached_images = cache_manager.get_all_images()
        expected_result_filenames = set()
        for img in cached_images:
            if img.get("status") == "completed":
                stem, _ = os.path.splitext(img["cached_filename"])
                expected_result_filenames.add(f"{stem}_pred.jpg")
        
        # 合并所有应该保留的文件名
        files_to_keep = cached_filenames.union(expected_result_filenames)
        
        # 找出孤立文件
        orphaned_files = [f for f in result_files if f not in files_to_keep]
        
        cleaned_count = 0
        errors = []
        
        for filename in orphaned_files:
            file_path = os.path.join(result_dir, filename)
            try:
                os.remove(file_path)
                print(f"已删除孤立的结果图片: {filename}")
                cleaned_count += 1
            except Exception as e:
                error_msg = f"删除文件失败 {filename}: {e}"
                print(error_msg)
                errors.append(error_msg)
        
        message = f"清理完成，删除了 {cleaned_count} 个孤立文件"
        if errors:
            message += f"，{len(errors)} 个文件删除失败"
        
        return jsonify({
            "success": True,
            "message": message,
            "details": {
                "cleaned_count": cleaned_count,
                "total_orphaned": len(orphaned_files),
                "errors": errors
            }
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
