"""
多模态暗场检测 - Flask API接口

提供RESTful API用于：
1. 多模态图像采集
2. 多模态缺陷检测
3. 检测报告生成
4. 可视化结果查看

作者：AI助手
日期：2025-10-22
"""

from flask import Blueprint, request, jsonify, send_file, session
import cv2
import numpy as np
import os
import json
from datetime import datetime
import uuid
from typing import Dict, List

# 导入多模态检测模块
from .multimodal_darkfield import (
    MultiModalDarkfieldDetector,
    create_multimodal_detector
)

# 导入相机管理器
from .recognition import camera_thread_manager

# 导入在线缓存管理器
from .online_processing import online_cache_manager


# 创建蓝图
multimodal_bp = Blueprint('multimodal', __name__, url_prefix='/api/multimodal')


# 全局检测器实例
MULTIMODAL_DETECTOR = None


def get_detector():
    """获取或创建检测器实例"""
    global MULTIMODAL_DETECTOR
    
    if MULTIMODAL_DETECTOR is None:
        # 获取YOLO模型路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(
            base_dir, "runs", "detect", "yolov8m_glass_detection", 
            "weights", "best.pt"
        )
        
        if os.path.exists(model_path):
            MULTIMODAL_DETECTOR = create_multimodal_detector(model_path)
            print(f"[多模态API] 检测器初始化成功: {model_path}")
        else:
            MULTIMODAL_DETECTOR = create_multimodal_detector()
            print(f"[多模态API] 检测器初始化（无YOLO模型）")
    
    return MULTIMODAL_DETECTOR


@multimodal_bp.route('/status', methods=['GET'])
def get_status():
    """获取多模态检测系统状态"""
    try:
        detector = get_detector()
        
        # 获取相机实例
        CAM = camera_thread_manager.get_camera()
        
        status = {
            'system_ready': detector is not None,
            'yolo_model_loaded': detector.yolo_model is not None,
            'fusion_model_loaded': detector.fusion_model is not None,
            'supported_modalities': [
                'darkfield', 'brightfield', 'sidelight', 'coaxial'
            ],
            'supported_defects': detector.defect_names,
            'camera_connected': CAM.device_status if CAM else False,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multimodal_bp.route('/capture', methods=['POST'])
def capture_multimodal():
    """
    采集多模态图像
    
    Request Body:
    {
        "modalities": ["darkfield", "brightfield", "sidelight", "coaxial"],
        "save_to_cache": true
    }
    
    Response:
    {
        "success": true,
        "images": {
            "darkfield": "/api/multimodal/image/xxx_darkfield.jpg",
            "brightfield": "/api/multimodal/image/xxx_brightfield.jpg",
            ...
        },
        "session_id": "xxx"
    }
    """
    try:
        data = request.get_json() or {}
        requested_modalities = data.get('modalities', [
            'darkfield', 'brightfield', 'sidelight', 'coaxial'
        ])
        save_to_cache = data.get('save_to_cache', True)
        
        # 获取相机实例
        CAM = camera_thread_manager.get_camera()
        
        # 检查相机状态
        if not CAM or not CAM.device_status:
            return jsonify({
                'success': False,
                'error': '相机未启动或未就绪，请先启动相机'
            }), 400
        
        # 获取检测器
        detector = get_detector()
        
        # 采集多模态图像
        print(f"[多模态API] 开始采集图像，模式：{requested_modalities}")
        modality_images = detector.capture_multimodal_images(CAM)
        
        # 筛选请求的模态
        filtered_images = {
            k: v for k, v in modality_images.items() 
            if k in requested_modalities
        }
        
        if not filtered_images:
            return jsonify({
                'success': False,
                'error': '未能采集到任何图像'
            }), 500
        
        # 生成会话ID
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存图像
        image_urls = {}
        saved_paths = {}
        
        multimodal_dir = os.path.join(
            os.path.dirname(__file__), '..', 'static', 'data', 
            'multimodal_cache'
        )
        os.makedirs(multimodal_dir, exist_ok=True)
        
        for modality, image in filtered_images.items():
            filename = f"{session_id}_{timestamp}_{modality}.jpg"
            filepath = os.path.join(multimodal_dir, filename)
            
            cv2.imwrite(filepath, image)
            
            image_urls[modality] = f"/api/multimodal/image/{filename}"
            saved_paths[modality] = filepath
        
        # 存储到session
        session['multimodal_session_id'] = session_id
        session['multimodal_images'] = saved_paths
        
        print(f"[多模态API] 图像采集完成，会话ID：{session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'images': image_urls,
            'timestamp': timestamp,
            'modalities_captured': list(filtered_images.keys())
        })
        
    except Exception as e:
        print(f"[多模态API] 采集失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multimodal_bp.route('/detect', methods=['POST'])
def detect_defects():
    """
    执行多模态缺陷检测
    
    Request Body:
    {
        "session_id": "xxx",  // 可选，不提供则使用当前session
        "image_paths": {...},  // 可选，直接提供图像路径
        "generate_visualization": true
    }
    
    Response:
    {
        "success": true,
        "defects": [...],
        "report": {...},
        "visualization": "/api/multimodal/result/xxx_visualization.jpg"
    }
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', session.get('multimodal_session_id'))
        generate_vis = data.get('generate_visualization', True)
        
        # 获取图像路径
        if 'image_paths' in data:
            image_paths = data['image_paths']
        else:
            image_paths = session.get('multimodal_images')
        
        if not image_paths:
            return jsonify({
                'success': False,
                'error': '未找到待检测的图像，请先采集图像'
            }), 400
        
        # 加载图像
        modality_images = {}
        for modality, path in image_paths.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    modality_images[modality] = img
        
        if not modality_images:
            return jsonify({
                'success': False,
                'error': '无法加载图像数据'
            }), 500
        
        # 获取检测器并执行检测
        detector = get_detector()
        print(f"[多模态API] 开始检测，模态数量：{len(modality_images)}")
        
        defects, vis_image, report = detector.detect(modality_images)
        
        print(f"[多模态API] 检测完成，发现{len(defects)}个缺陷")
        
        # 保存可视化结果
        vis_url = None
        if generate_vis and vis_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_filename = f"{session_id}_{timestamp}_visualization.jpg"
            
            multimodal_dir = os.path.join(
                os.path.dirname(__file__), '..', 'static', 'data', 
                'multimodal_cache'
            )
            os.makedirs(multimodal_dir, exist_ok=True)
            
            vis_path = os.path.join(multimodal_dir, vis_filename)
            cv2.imwrite(vis_path, vis_image)
            
            vis_url = f"/api/multimodal/result/{vis_filename}"
            
            # 存储到session
            session['multimodal_visualization'] = vis_path
        
        # 保存报告
        report_filename = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_report.json"
        report_path = os.path.join(
            os.path.dirname(__file__), '..', 'static', 'data', 
            'multimodal_cache', report_filename
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        session['multimodal_report'] = report_path
        
        return jsonify({
            'success': True,
            'defects': report['defects'],
            'report': report,
            'visualization': vis_url,
            'report_file': f"/api/multimodal/report/{report_filename}"
        })
        
    except Exception as e:
        print(f"[多模态API] 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multimodal_bp.route('/image/<path:filename>', methods=['GET'])
def serve_image(filename):
    """提供多模态图像"""
    try:
        multimodal_dir = os.path.join(
            os.path.dirname(__file__), '..', 'static', 'data', 
            'multimodal_cache'
        )
        
        filepath = os.path.join(multimodal_dir, filename)
        
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        else:
            return jsonify({
                'success': False,
                'error': '图像不存在'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multimodal_bp.route('/result/<path:filename>', methods=['GET'])
def serve_result(filename):
    """提供检测结果可视化图像"""
    return serve_image(filename)


@multimodal_bp.route('/report/<path:filename>', methods=['GET'])
def serve_report(filename):
    """提供检测报告JSON文件"""
    try:
        multimodal_dir = os.path.join(
            os.path.dirname(__file__), '..', 'static', 'data', 
            'multimodal_cache'
        )
        
        filepath = os.path.join(multimodal_dir, filename)
        
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='application/json')
        else:
            return jsonify({
                'success': False,
                'error': '报告不存在'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multimodal_bp.route('/analyze', methods=['POST'])
def analyze_single_image():
    """
    对单张图像进行多模态分析（模拟多光源）
    
    用于没有多光源硬件的情况，通过图像处理模拟不同光源效果
    
    Request: multipart/form-data with 'image' file
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': '未提供图像文件'
            }), 400
        
        file = request.files['image']
        
        # 读取图像
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': '无法解析图像'
            }), 400
        
        # 获取检测器
        detector = get_detector()
        
        # 模拟多模态图像
        modality_images = {
            'darkfield': detector._simulate_darkfield(image),
            'brightfield': image,
            'sidelight': detector._simulate_sidelight(image),
            'coaxial': detector._simulate_coaxial(image)
        }
        
        # 执行检测
        defects, vis_image, report = detector.detect(modality_images)
        
        # 保存结果
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        multimodal_dir = os.path.join(
            os.path.dirname(__file__), '..', 'static', 'data', 
            'multimodal_cache'
        )
        os.makedirs(multimodal_dir, exist_ok=True)
        
        # 保存可视化
        vis_filename = f"{session_id}_{timestamp}_analysis.jpg"
        vis_path = os.path.join(multimodal_dir, vis_filename)
        cv2.imwrite(vis_path, vis_image)
        
        return jsonify({
            'success': True,
            'defects': report['defects'],
            'report': report,
            'visualization': f"/api/multimodal/result/{vis_filename}"
        })
        
    except Exception as e:
        print(f"[多模态API] 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multimodal_bp.route('/clear_cache', methods=['POST'])
def clear_cache():
    """清空多模态缓存"""
    try:
        multimodal_dir = os.path.join(
            os.path.dirname(__file__), '..', 'static', 'data', 
            'multimodal_cache'
        )
        
        if os.path.exists(multimodal_dir):
            import shutil
            shutil.rmtree(multimodal_dir)
            os.makedirs(multimodal_dir, exist_ok=True)
        
        # 清空session
        session.pop('multimodal_session_id', None)
        session.pop('multimodal_images', None)
        session.pop('multimodal_visualization', None)
        session.pop('multimodal_report', None)
        
        return jsonify({
            'success': True,
            'message': '多模态缓存已清空'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multimodal_bp.route('/export_report', methods=['POST'])
def export_report():
    """
    导出详细报告（支持PDF、Excel）
    
    Request Body:
    {
        "format": "pdf" | "excel" | "json",
        "session_id": "xxx"
    }
    """
    try:
        data = request.get_json() or {}
        export_format = data.get('format', 'json')
        session_id = data.get('session_id', session.get('multimodal_session_id'))
        
        report_path = session.get('multimodal_report')
        
        if not report_path or not os.path.exists(report_path):
            return jsonify({
                'success': False,
                'error': '未找到检测报告'
            }), 404
        
        # 读取报告
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        if export_format == 'json':
            return jsonify({
                'success': True,
                'report': report
            })
        
        elif export_format == 'excel':
            # TODO: 实现Excel导出
            return jsonify({
                'success': False,
                'error': 'Excel导出功能开发中'
            }), 501
        
        elif export_format == 'pdf':
            # TODO: 实现PDF导出
            return jsonify({
                'success': False,
                'error': 'PDF导出功能开发中'
            }), 501
        
        else:
            return jsonify({
                'success': False,
                'error': f'不支持的格式：{export_format}'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 健康检查
@multimodal_bp.route('/health', methods=['GET'])
def health_check():
    """系统健康检查"""
    try:
        detector = get_detector()
        
        # 获取相机实例
        CAM = camera_thread_manager.get_camera()
        
        health = {
            'status': 'healthy',
            'detector_ready': detector is not None,
            'camera_status': CAM.device_status if CAM else False,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(health)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("多模态API模块已加载")

