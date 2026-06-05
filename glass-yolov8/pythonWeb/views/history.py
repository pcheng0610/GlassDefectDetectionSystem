from flask import Blueprint, render_template, request, jsonify, session
import os
import json
from datetime import datetime

hi = Blueprint("history", __name__)

# 本地缓存目录
LOCAL_CACHE_DIR = r"D:\zjzjzj\glass-yolov8\pythonWeb\static\data\local_cache"
IMAGES_DIR = os.path.join(LOCAL_CACHE_DIR, "images")
RESULTS_DIR = os.path.join(LOCAL_CACHE_DIR, "results")
RESULTS_METADATA_FILE = os.path.join(LOCAL_CACHE_DIR, "results_metadata.json")

def get_local_cache_images():
    """从本地缓存读取图片信息"""
    images = []
    
    if not os.path.exists(IMAGES_DIR):
        return images
    
    # 读取图片文件
    for filename in os.listdir(IMAGES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            filepath = os.path.join(IMAGES_DIR, filename)
            images.append({
                'filename': filename,
                'filepath': filepath,
                'create_time': datetime.fromtimestamp(os.path.getctime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
            })
    
    return images

def get_local_cache_results():
    """从本地缓存元数据读取结果信息"""
    results = []
    
    if not os.path.exists(RESULTS_METADATA_FILE):
        return results
    
    try:
        with open(RESULTS_METADATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                for result in data["results"]:
                    results.append({
                        'id': result.get('id'),
                        'image_id': result.get('image_id'),
                        'result_class': result.get('result_class', '未识别'),
                        'confidence': result.get('confidence', 0.0),
                        'result_filename': result.get('result_filename'),
                        'result_path': result.get('result_path'),
                        'created_time': result.get('created_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    })
    except Exception as e:
        print(f"读取 results_metadata.json 失败: {e}")
    
    return results

@hi.route("/history")
def history_page():
    return render_template("history.html")

@hi.route("/api/history/list", methods=["GET"])
def get_history_list():
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 10))
    offset = (page - 1) * limit
    
    # 从本地缓存读取图片和结果
    images = get_local_cache_images()
    results = get_local_cache_results()
    
    # 创建 image_id 到结果的映射
    results_map = {}
    for result in results:
        image_id = result['image_id']
        if image_id not in results_map:
            results_map[image_id] = []
        results_map[image_id].append(result)
    
    # 合并图片和结果
    records = []
    for img in images:
        # 从图片文件名中提取 image_id（UUID）
        filename = img['filename']
        if len(filename) >= 36:
            image_id = filename[:36]
        else:
            continue
        
        # 查找对应的结果
        result_list = results_map.get(image_id, [])
        
        if result_list:
            result = result_list[0]
            record = {
                'id': image_id,
                'create_time': img['create_time'],
                'result_class': result['result_class'],
                'user': 'admin',
                'image_url': f"/static/data/local_cache/results/{result['result_filename']}",
                'original_image_url': f"/static/data/local_cache/images/{img['filename']}",
                'confidence': result['confidence']
            }
        else:
            record = {
                'id': image_id,
                'create_time': img['create_time'],
                'result_class': '未识别',
                'user': 'admin',
                'image_url': None,
                'original_image_url': f"/static/data/local_cache/images/{img['filename']}",
                'confidence': 0.0
            }
        records.append(record)
    
    # 按时间排序
    records.sort(key=lambda x: x['create_time'], reverse=True)
    
    total = len(records)
    start = offset
    end = offset + limit
    records = records[start:end]
    
    return jsonify({
        "code": 200,
        "data": records,
        "total": total,
        "page": page,
        "limit": limit
    })

@hi.route("/api/history/delete", methods=["POST"])
def delete_history():
    data = request.get_json()
    record_id = data.get('id')
    
    if not record_id:
        return jsonify({"code": 400, "message": "缺少记录ID"})
    
    try:
        # 删除对应的图片文件
        image_filename = record_id + ".bmp"
        image_path = os.path.join(IMAGES_DIR, image_filename)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # 删除对应的结果文件
        results_metadata = get_local_cache_results()
        for result in results_metadata:
            if result['image_id'] == record_id:
                result_filename = result['result_filename']
                result_path = os.path.join(RESULTS_DIR, result_filename)
                if os.path.exists(result_path):
                    os.remove(result_path)
        
        # 更新 results_metadata.json
        results_metadata = [r for r in results_metadata if r['image_id'] != record_id]
        with open(RESULTS_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({"results": results_metadata, "batch_id": 0}, f, ensure_ascii=False, indent=2)
        
        return jsonify({"code": 200, "message": "删除成功"})
    except Exception as e:
        return jsonify({"code": 500, "message": f"删除失败: {str(e)}"})

@hi.route("/api/history/clear", methods=["POST"])
def clear_history():
    try:
        # 删除所有图片文件
        for filename in os.listdir(IMAGES_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                os.remove(os.path.join(IMAGES_DIR, filename))
        
        # 删除所有结果文件
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                os.remove(os.path.join(RESULTS_DIR, filename))
        
        # 清空 results_metadata.json
        with open(RESULTS_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({"results": [], "batch_id": 0}, f, ensure_ascii=False, indent=2)
        
        return jsonify({"code": 200, "message": "清空成功"})
    except Exception as e:
        return jsonify({"code": 500, "message": f"清空失败: {str(e)}"})

@hi.route("/api/history/delete_batch", methods=["POST"])
def delete_batch():
    data = request.get_json()
    ids = data.get('ids', [])
    
    if not ids:
        return jsonify({"code": 400, "message": "请选择要删除的记录"})
    
    try:
        for record_id in ids:
            # 删除对应的图片文件
            image_filename = record_id + ".bmp"
            image_path = os.path.join(IMAGES_DIR, image_filename)
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # 删除对应的结果文件
            results_metadata = get_local_cache_results()
            for result in results_metadata:
                if result['image_id'] == record_id:
                    result_filename = result['result_filename']
                    result_path = os.path.join(RESULTS_DIR, result_filename)
                    if os.path.exists(result_path):
                        os.remove(result_path)
        
        # 更新 results_metadata.json
        results_metadata = get_local_cache_results()
        results_metadata = [r for r in results_metadata if r['image_id'] not in ids]
        with open(RESULTS_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({"results": results_metadata, "batch_id": 0}, f, ensure_ascii=False, indent=2)
        
        return jsonify({"code": 200, "message": f"成功删除 {len(ids)} 条记录"})
    except Exception as e:
        return jsonify({"code": 500, "message": f"批量删除失败: {str(e)}"})
