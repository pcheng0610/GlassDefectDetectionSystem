from flask import Blueprint, render_template, request, jsonify
import os
import json
from datetime import datetime

vi = Blueprint("visualization", __name__)

# 本地缓存目录
LOCAL_CACHE_DIR = r"D:\zjzjzj\glass-yolov8\pythonWeb\static\data\local_cache"
RESULTS_METADATA_FILE = os.path.join(LOCAL_CACHE_DIR, "results_metadata.json")

@vi.route("/visualization")
def visualization_page():
    return render_template("visualization.html")

@vi.route("/api/visualization/data", methods=["GET"])
def get_visualization_data():
    try:
        # 读取结果元数据
        results = []
        if os.path.exists(RESULTS_METADATA_FILE):
            try:
                with open(RESULTS_METADATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "results" in data:
                        results = data["results"]
            except Exception as e:
                print(f"读取 results_metadata.json 失败: {e}")
        
        # 计算统计信息
        total_count = len(results)
        # 瑕疵数量：排除「No detection」和「未识别」（表示未检测到瑕疵）
        no_defect_classes = ['No detection', '未识别', 'no detection', 'No Detection']
        defect_count = sum(1 for r in results if r.get('result_class') and r.get('result_class') not in no_defect_classes)
        
        # 类别分布
        category_distribution = {}
        for r in results:
            category = r.get('result_class', '未识别')
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        category_list = [
            {"category": cat, "count": count}
            for cat, count in sorted(category_distribution.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # 时间趋势（按小时统计）
        trend_data = {}
        for r in results:
            created_time = r.get('created_time', '')
            if created_time:
                # 提取日期和小时
                try:
                    dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                    hour_key = dt.strftime("%Y-%m-%d %H:00")
                    trend_data[hour_key] = trend_data.get(hour_key, 0) + 1
                except:
                    # 简单解析
                    if len(created_time) >= 13:
                        hour_key = created_time[:13] + ":00"
                        trend_data[hour_key] = trend_data.get(hour_key, 0) + 1
        
        trend_list = [
            {"time": time, "count": count}
            for time, count in sorted(trend_data.items(), key=lambda x: x[0])
        ]
        
        # 置信度分布
        confidence_distribution = [0, 0, 0, 0, 0]
        for r in results:
            confidence = r.get('confidence', 0)
            if confidence < 0.5:
                confidence_distribution[0] += 1
            elif confidence < 0.7:
                confidence_distribution[1] += 1
            elif confidence < 0.85:
                confidence_distribution[2] += 1
            elif confidence < 0.95:
                confidence_distribution[3] += 1
            else:
                confidence_distribution[4] += 1
        
        # 正常数量（未检测到瑕疵）
        normal_count = total_count - defect_count
        
        return jsonify({
            "code": 200,
            "data": {
                "total_count": total_count,
                "defect_count": defect_count,
                "normal_count": normal_count,
                "category_distribution": category_list,
                "trend": trend_list,
                "confidence_distribution": confidence_distribution
            }
        })
    except Exception as e:
        return jsonify({
            "code": 500,
            "message": f"获取数据失败: {str(e)}"
        })
