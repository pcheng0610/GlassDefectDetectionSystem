"""
多模态暗场镜片缺陷检测模块

支持的成像模式：
1. 暗场成像 (Dark Field) - 检测微小缺陷
2. 明场成像 (Bright Field) - 检测大面积缺陷
3. 侧光成像 (Side Lighting) - 检测表面划痕
4. 同轴光成像 (Coaxial) - 检测反射缺陷

作者：AI助手
日期：2025-10-22
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class DefectInfo:
    """缺陷信息数据类"""
    defect_type: str  # 缺陷类型
    confidence: float  # 置信度
    location: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    severity: str  # 严重程度：轻微/中等/严重
    modality_source: List[str]  # 检测到该缺陷的成像模式
    description: str  # 详细描述
    root_cause: Optional[str] = None  # 可能的根因


class MultiModalImageProcessor:
    """多模态图像预处理器"""
    
    def __init__(self):
        self.image_size = (640, 640)
        
    def process_darkfield(self, image: np.ndarray) -> np.ndarray:
        """
        暗场图像处理
        目标：增强亮点（缺陷）对比度
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 增强对比度 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 降噪（保留边缘）
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 形态学处理（突出小缺陷）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
        result = cv2.add(denoised, tophat)
        
        return result
    
    def process_brightfield(self, image: np.ndarray) -> np.ndarray:
        """
        明场图像处理
        目标：检测大面积缺陷和污染
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 背景均匀化
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        normalized = cv2.divide(gray, blur, scale=255)
        
        # 自适应阈值
        adaptive = cv2.adaptiveThreshold(
            normalized, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        return adaptive
    
    def process_sidelight(self, image: np.ndarray) -> np.ndarray:
        """
        侧光图像处理
        目标：检测表面划痕和凹凸
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 方向梯度增强（检测线性缺陷）
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mag = np.uint8(255 * gradient_mag / np.max(gradient_mag))
        
        # 形态学开运算去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        opened = cv2.morphologyEx(gradient_mag, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    def process_coaxial(self, image: np.ndarray) -> np.ndarray:
        """
        同轴光图像处理
        目标：检测反射缺陷（如镀膜不均）
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 均值滤波后做差分（检测反射不均匀）
        mean_val = np.mean(gray)
        diff = cv2.absdiff(gray, np.uint8(mean_val))
        
        # 增强微小差异
        enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        return enhanced
    
    def align_images(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        图像配准对齐（如果多个相机拍摄）
        使用特征点匹配进行图像对齐
        """
        if len(images) <= 1:
            return images
        
        # 以第一张图为基准
        base_key = list(images.keys())[0]
        base_img = images[base_key]
        
        aligned_images = {base_key: base_img}
        
        # ORB特征检测器
        orb = cv2.ORB_create(5000)
        
        # 检测基准图像的特征点
        kp_base, des_base = orb.detectAndCompute(base_img, None)
        
        # 对齐其他图像
        for key, img in images.items():
            if key == base_key:
                continue
            
            kp, des = orb.detectAndCompute(img, None)
            
            # 特征匹配
            if des is not None and des_base is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des_base, des)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # 使用最佳匹配点计算变换矩阵
                if len(matches) > 10:
                    src_pts = np.float32([kp_base[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                    
                    # 计算仿射变换
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        h, w = base_img.shape[:2]
                        aligned = cv2.warpPerspective(img, M, (w, h))
                        aligned_images[key] = aligned
                    else:
                        aligned_images[key] = img
                else:
                    aligned_images[key] = img
            else:
                aligned_images[key] = img
        
        return aligned_images


class MultiModalFusionNetwork(nn.Module):
    """
    多模态融合神经网络
    架构：多分支编码器 + 跨模态注意力 + 融合解码器
    """
    
    def __init__(self, num_modalities: int = 4, num_classes: int = 11):
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # 每个模态的独立编码器（轻量级）
        self.encoders = nn.ModuleList([
            self._build_encoder() for _ in range(num_modalities)
        ])
        
        # 跨模态注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8,
            batch_first=True
        )
        
        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * num_modalities, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _build_encoder(self):
        """构建单模态编码器"""
        return nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """构建残差块"""
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(self._residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride=1):
        """残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, modality_inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            modality_inputs: List of [B, 1, H, W] tensors
        
        Returns:
            logits: [B, num_classes]
        """
        # 1. 各模态独立编码
        features = []
        for i, x in enumerate(modality_inputs):
            feat = self.encoders[i](x)  # [B, 256, H', W']
            features.append(feat)
        
        # 2. 跨模态注意力（可选：增强模态间交互）
        # 这里简化为拼接，可以改进为注意力加权
        
        # 3. 特征融合
        fused = torch.cat(features, dim=1)  # [B, 256*M, H', W']
        fused = self.fusion_conv(fused)  # [B, 256, H', W']
        
        # 4. 分类
        logits = self.classifier(fused)  # [B, num_classes]
        
        return logits


class MultiModalDarkfieldDetector:
    """
    多模态暗场缺陷检测器（主类）
    """
    
    def __init__(self, yolo_model_path: str = None):
        self.processor = MultiModalImageProcessor()
        
        # 加载YOLO模型（单模态检测）
        self.yolo_model = None
        if yolo_model_path:
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_model_path)
        
        # 多模态融合网络（如果有训练好的权重）
        self.fusion_model = None
        # self.fusion_model = MultiModalFusionNetwork()
        # self.fusion_model.load_state_dict(torch.load('fusion_model.pth'))
        # self.fusion_model.eval()
        
        self.defect_names = [
            '印痕', '光圈线', '划痕', '坑点', '有线', 
            '气泡', '坑痕', '羽毛纹', '脱膜', '瑕疵点', '黑点'
        ]
    
    def capture_multimodal_images(self, camera_controller) -> Dict[str, np.ndarray]:
        """
        采集多模态图像
        
        硬件要求：
        1. 方案A：单相机 + 可切换光源控制器
        2. 方案B：多相机同步采集系统
        
        Args:
            camera_controller: 相机控制器对象
        
        Returns:
            Dict[模态名称, 图像数据]
        """
        images = {}
        
        # 示例：模拟多光源切换采集
        # 实际实现需要根据硬件接口调整
        
        # 1. 暗场图像
        # camera_controller.set_lighting_mode('darkfield')
        # images['darkfield'] = camera_controller.capture()
        
        # 2. 明场图像
        # camera_controller.set_lighting_mode('brightfield')
        # images['brightfield'] = camera_controller.capture()
        
        # 3. 侧光图像
        # camera_controller.set_lighting_mode('sidelight')
        # images['sidelight'] = camera_controller.capture()
        
        # 4. 同轴光图像
        # camera_controller.set_lighting_mode('coaxial')
        # images['coaxial'] = camera_controller.capture()
        
        # 临时：从单张图像模拟多模态（仅用于演示）
        base_image = camera_controller.get_image()
        if base_image is not None:
            # 转为numpy数组
            temp = np.frombuffer(base_image, dtype=np.uint8)
            width = camera_controller.frame_info.nWidth
            height = camera_controller.frame_info.nHeight
            img = temp.reshape((height, width, 3))
            
            # 模拟不同光源效果（实际应该硬件切换）
            images['darkfield'] = self._simulate_darkfield(img)
            images['brightfield'] = img
            images['sidelight'] = self._simulate_sidelight(img)
            images['coaxial'] = self._simulate_coaxial(img)
        
        return images
    
    def _simulate_darkfield(self, img: np.ndarray) -> np.ndarray:
        """模拟暗场效果（仅用于演示，实际应硬件实现）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        enhanced = cv2.equalizeHist(inverted)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _simulate_sidelight(self, img: np.ndarray) -> np.ndarray:
        """模拟侧光效果"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        sobel = np.uint8(255 * np.abs(sobel) / np.max(np.abs(sobel)))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    
    def _simulate_coaxial(self, img: np.ndarray) -> np.ndarray:
        """模拟同轴光效果"""
        # 增强反射区域
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def detect_single_modality(
        self, 
        image: np.ndarray, 
        modality: str
    ) -> List[DefectInfo]:
        """
        单模态检测
        
        Args:
            image: 输入图像
            modality: 成像模式
        
        Returns:
            检测到的缺陷列表
        """
        defects = []
        
        if self.yolo_model is None:
            return defects
        
        # YOLO检测
        results = self.yolo_model(image, conf=0.25, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                defect = DefectInfo(
                    defect_type=self.defect_names[cls_id],
                    confidence=conf,
                    location=(int(x1), int(y1), int(x2), int(y2)),
                    severity=self._assess_severity(conf, x2-x1, y2-y1),
                    modality_source=[modality],
                    description=f"{modality}模式检测到{self.defect_names[cls_id]}"
                )
                
                defects.append(defect)
        
        return defects
    
    def _assess_severity(self, confidence: float, width: float, height: float) -> str:
        """评估缺陷严重程度"""
        area = width * height
        
        if confidence > 0.8 and area > 10000:
            return "严重"
        elif confidence > 0.5 and area > 5000:
            return "中等"
        else:
            return "轻微"
    
    def fuse_detections(
        self, 
        detections_by_modality: Dict[str, List[DefectInfo]]
    ) -> List[DefectInfo]:
        """
        融合多模态检测结果
        
        策略：
        1. NMS去除重复检测
        2. 多模态一致性验证
        3. 置信度加权融合
        """
        all_defects = []
        for modality, defects in detections_by_modality.items():
            all_defects.extend(defects)
        
        if not all_defects:
            return []
        
        # 简化版：直接返回所有检测（实际应该做NMS和融合）
        # 这里可以实现更复杂的融合逻辑
        fused_defects = self._nms_multimodal(all_defects, iou_threshold=0.5)
        
        # 添加根因分析
        for defect in fused_defects:
            defect.root_cause = self._analyze_root_cause(defect)
        
        return fused_defects
    
    def _nms_multimodal(
        self, 
        defects: List[DefectInfo], 
        iou_threshold: float = 0.5
    ) -> List[DefectInfo]:
        """多模态NMS"""
        if not defects:
            return []
        
        # 按置信度排序
        defects = sorted(defects, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while defects:
            current = defects.pop(0)
            keep.append(current)
            
            # 移除与当前框高度重叠的检测
            defects = [
                d for d in defects 
                if self._calculate_iou(current.location, d.location) < iou_threshold
            ]
        
        return keep
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """计算IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _analyze_root_cause(self, defect: DefectInfo) -> str:
        """根因分析"""
        root_causes = {
            '划痕': '可能由切割工具磨损、搬运不当或清洁过程划伤导致',
            '气泡': '玻璃熔化过程中气体未完全排出，建议检查熔炉温度和保温时间',
            '坑点': '原料杂质或模具表面缺陷导致，建议检查原料纯度',
            '印痕': '压制或脱模过程中压力不均，建议检查模具状态',
            '黑点': '原料中含有杂质颗粒，建议加强原料筛选'
        }
        
        return root_causes.get(defect.defect_type, '需进一步分析')
    
    def generate_report(
        self, 
        defects: List[DefectInfo], 
        modality_images: Dict[str, np.ndarray]
    ) -> Dict:
        """
        生成多模态检测报告
        
        Returns:
            包含检测结果、统计信息、可视化的完整报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_defects': len(defects),
            'modalities_used': list(modality_images.keys()),
            'defects': [],
            'statistics': self._calculate_statistics(defects),
            'quality_grade': self._calculate_quality_grade(defects),
            'recommendations': []
        }
        
        # 缺陷详情
        for defect in defects:
            report['defects'].append({
                'type': defect.defect_type,
                'confidence': round(defect.confidence, 3),
                'location': defect.location,
                'severity': defect.severity,
                'detected_by': defect.modality_source,
                'description': defect.description,
                'root_cause': defect.root_cause
            })
        
        # 生成建议
        report['recommendations'] = self._generate_recommendations(defects)
        
        return report
    
    def _calculate_statistics(self, defects: List[DefectInfo]) -> Dict:
        """计算统计信息"""
        from collections import Counter
        
        defect_types = [d.defect_type for d in defects]
        severity_levels = [d.severity for d in defects]
        
        return {
            'total_defects': len(defects),
            'defect_type_distribution': dict(Counter(defect_types)),
            'severity_distribution': dict(Counter(severity_levels)),
            'avg_confidence': np.mean([d.confidence for d in defects]) if defects else 0
        }
    
    def _calculate_quality_grade(self, defects: List[DefectInfo]) -> str:
        """计算质量等级"""
        if not defects:
            return "优等品"
        
        severe_count = sum(1 for d in defects if d.severity == "严重")
        moderate_count = sum(1 for d in defects if d.severity == "中等")
        
        if severe_count > 0:
            return "不合格"
        elif moderate_count > 2:
            return "三等品"
        elif len(defects) > 5:
            return "二等品"
        else:
            return "一等品"
    
    def _generate_recommendations(self, defects: List[DefectInfo]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        from collections import Counter
        defect_types = Counter([d.defect_type for d in defects])
        
        for defect_type, count in defect_types.most_common(3):
            if defect_type == '气泡' and count > 2:
                recommendations.append("频繁出现气泡缺陷，建议提高熔炉温度5-10℃并延长保温时间")
            elif defect_type == '划痕' and count > 2:
                recommendations.append("划痕缺陷较多，建议检查搬运流程和清洁工具")
            elif defect_type == '坑点' and count > 2:
                recommendations.append("坑点缺陷较多，建议检查原料纯度和模具清洁度")
        
        if len(defects) > 10:
            recommendations.append("缺陷总数较多，建议全面检查生产工艺参数")
        
        return recommendations
    
    def visualize_results(
        self, 
        modality_images: Dict[str, np.ndarray],
        defects: List[DefectInfo],
        save_path: str = None
    ) -> np.ndarray:
        """
        可视化多模态检测结果
        
        Returns:
            拼接后的可视化图像
        """
        vis_images = []
        
        for modality, image in modality_images.items():
            img_copy = image.copy()
            
            # 绘制检测框
            for defect in defects:
                if modality in defect.modality_source:
                    x1, y1, x2, y2 = defect.location
                    
                    # 根据严重程度选择颜色
                    color_map = {
                        '严重': (0, 0, 255),  # 红色
                        '中等': (0, 165, 255),  # 橙色
                        '轻微': (0, 255, 0)  # 绿色
                    }
                    color = color_map.get(defect.severity, (255, 255, 255))
                    
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加标签
                    label = f"{defect.defect_type} {defect.confidence:.2f}"
                    cv2.putText(
                        img_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
            
            # 添加模态名称标题
            cv2.putText(
                img_copy, modality.upper(), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            
            vis_images.append(img_copy)
        
        # 拼接为2x2网格
        if len(vis_images) == 4:
            top_row = np.hstack([vis_images[0], vis_images[1]])
            bottom_row = np.hstack([vis_images[2], vis_images[3]])
            final_vis = np.vstack([top_row, bottom_row])
        else:
            final_vis = np.hstack(vis_images)
        
        # 保存
        if save_path:
            cv2.imwrite(save_path, final_vis)
        
        return final_vis
    
    def detect(
        self, 
        modality_images: Dict[str, np.ndarray]
    ) -> Tuple[List[DefectInfo], np.ndarray, Dict]:
        """
        完整的多模态检测流程
        
        Args:
            modality_images: 多模态图像字典
        
        Returns:
            (缺陷列表, 可视化图像, 检测报告)
        """
        print("[多模态检测] 开始处理...")
        
        # 1. 图像预处理
        processed_images = {}
        for modality, image in modality_images.items():
            if modality == 'darkfield':
                processed_images[modality] = self.processor.process_darkfield(image)
            elif modality == 'brightfield':
                processed_images[modality] = self.processor.process_brightfield(image)
            elif modality == 'sidelight':
                processed_images[modality] = self.processor.process_sidelight(image)
            elif modality == 'coaxial':
                processed_images[modality] = self.processor.process_coaxial(image)
            else:
                processed_images[modality] = image
        
        # 2. 图像对齐（如果需要）
        # aligned_images = self.processor.align_images(processed_images)
        
        # 3. 各模态独立检测
        detections_by_modality = {}
        for modality, image in modality_images.items():
            print(f"[多模态检测] 处理{modality}模式...")
            detections = self.detect_single_modality(image, modality)
            detections_by_modality[modality] = detections
            print(f"[多模态检测] {modality}检测到{len(detections)}个缺陷")
        
        # 4. 融合检测结果
        print("[多模态检测] 融合多模态结果...")
        fused_defects = self.fuse_detections(detections_by_modality)
        print(f"[多模态检测] 融合后共{len(fused_defects)}个缺陷")
        
        # 5. 生成可视化
        vis_image = self.visualize_results(modality_images, fused_defects)
        
        # 6. 生成报告
        report = self.generate_report(fused_defects, modality_images)
        
        print("[多模态检测] 检测完成")
        
        return fused_defects, vis_image, report


# 工具函数
def create_multimodal_detector(yolo_model_path: str = None) -> MultiModalDarkfieldDetector:
    """创建多模态检测器实例"""
    return MultiModalDarkfieldDetector(yolo_model_path)


if __name__ == "__main__":
    # 测试代码
    print("多模态暗场检测模块已加载")
    
    # 示例：创建检测器
    detector = create_multimodal_detector()
    print(f"检测器初始化完成，支持检测的缺陷类型：{detector.defect_names}")

