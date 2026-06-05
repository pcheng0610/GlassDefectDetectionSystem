"""
本地缓存管理系统
支持图片的本地存储、管理和批量操作
分为两个独立的缓存系统：images缓存和results缓存
"""
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from PIL import Image, ExifTags

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available, some features may be limited")

class LocalCacheManager:
    def __init__(self, cache_dir: str = "pythonWeb/static/data/local_cache", 
                 images_dir: str = None, results_dir: str = None,
                 cache_metadata_file: str = "cache_metadata.json", 
                 results_metadata_file: str = "results_metadata.json"):
        self.cache_dir = cache_dir
        self.images_dir = images_dir or os.path.join(cache_dir, "images")
        self.results_dir = results_dir or os.path.join(cache_dir, "results")
        self.images_metadata_file = os.path.join(cache_dir, cache_metadata_file)
        self.results_metadata_file = os.path.join(cache_dir, results_metadata_file)
        self.ensure_cache_dir()
        self.load_metadata()
    
    def ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_metadata(self):
        """加载缓存元数据"""
        # 加载images元数据
        if os.path.exists(self.images_metadata_file):
            try:
                with open(self.images_metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 验证数据结构
                    if isinstance(data, dict) and "images" in data:
                        self.images_metadata = data
                    else:
                        print(f"[警告] images元数据文件格式错误，使用默认结构")
                        self.images_metadata = {"images": [], "batch_id": 0}
            except Exception as e:
                print(f"[警告] 加载images元数据失败: {e}，使用默认结构")
                self.images_metadata = {"images": [], "batch_id": 0}
        else:
            self.images_metadata = {"images": [], "batch_id": 0}
        
        # 确保images元数据结构完整
        if "images" not in self.images_metadata:
            self.images_metadata["images"] = []
        if "batch_id" not in self.images_metadata:
            self.images_metadata["batch_id"] = 0
        
        # 加载results元数据
        if os.path.exists(self.results_metadata_file):
            try:
                with open(self.results_metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 验证数据结构
                    if isinstance(data, dict) and "results" in data:
                        self.results_metadata = data
                    else:
                        print(f"[警告] results元数据文件格式错误，使用默认结构")
                        self.results_metadata = {"results": [], "batch_id": 0}
            except Exception as e:
                print(f"[警告] 加载results元数据失败: {e}，使用默认结构")
                self.results_metadata = {"results": [], "batch_id": 0}
        else:
            self.results_metadata = {"results": [], "batch_id": 0}
        
        # 确保results元数据结构完整
        if "results" not in self.results_metadata:
            self.results_metadata["results"] = []
        if "batch_id" not in self.results_metadata:
            self.results_metadata["batch_id"] = 0
    
    def save_images_metadata(self):
        """保存images缓存元数据"""
        with open(self.images_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.images_metadata, f, ensure_ascii=False, indent=2)
    
    def save_results_metadata(self):
        """保存results缓存元数据"""
        with open(self.results_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.results_metadata, f, ensure_ascii=False, indent=2)
    
    def add_image(self, image_data: bytes = None, filename: str = None, source: str = "upload", 
                  cached_filename: str = None, file_path: str = None, file_size: int = None) -> Dict:
        """添加图片到本地缓存
        
        Args:
            image_data: 图片数据字节流（用于上传的图片）
            filename: 原始文件名
            source: 图片来源
            cached_filename: 缓存文件名（用于已保存的图片）
            file_path: 文件路径（用于已保存的图片）
            file_size: 文件大小（用于已保存的图片）
        """
        # 确保元数据结构正确
        if not isinstance(self.images_metadata, dict):
            self.images_metadata = {"images": [], "batch_id": 0}
        if "images" not in self.images_metadata:
            self.images_metadata["images"] = []
        if "batch_id" not in self.images_metadata:
            self.images_metadata["batch_id"] = 0
            
        image_id = str(uuid.uuid4())
        
        # 处理已保存的图片（用于在线拍照功能）
        if cached_filename and file_path:
            _, ext = os.path.splitext(cached_filename)
            if not ext:
                ext = '.jpg'
            
            # 提取拍摄时间
            capture_time = self._extract_capture_time(file_path)
            timestamp = capture_time.replace('-', '').replace(' ', '_').replace(':', '')[:15]
            
            image_meta = {
                "id": image_id,
                "filename": filename or cached_filename,
                "cached_filename": cached_filename,
                "path": file_path,
                "source": source,
                "capture_time": capture_time,
                "added_time": datetime.now().isoformat(),
                "status": "cached",
                "result": None,
                "confidence": None,
                "annotated_image": None,
                "file_size": file_size
            }
            
            self.images_metadata["images"].append(image_meta)
            self.save_images_metadata()
            return image_meta
        
        # 处理上传的图片数据
        if image_data is None or filename is None:
            raise ValueError("对于上传的图片，必须提供 image_data 和 filename 参数")
            
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = '.jpg'
        
        # 先保存临时文件以提取EXIF信息
        temp_path = os.path.join(self.images_dir, f"temp_{image_id}{ext}")
        
        if CV2_AVAILABLE:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite(temp_path, img)
        else:
            with open(temp_path, 'wb') as f:
                f.write(image_data)
        
        # 提取拍摄时间
        capture_time = self._extract_capture_time(temp_path)
        timestamp = capture_time.replace('-', '').replace(' ', '_').replace(':', '')[:15]
        image_filename = f"{image_id}_{timestamp}_off_line{ext}"
        image_path = os.path.join(self.images_dir, image_filename)
        
        # 重命名文件
        os.rename(temp_path, image_path)
        
        image_meta = {
            "id": image_id,
            "filename": filename,
            "cached_filename": image_filename,
            "path": image_path,
            "source": source,
            "capture_time": capture_time,
            "added_time": datetime.now().isoformat(),
            "status": "cached",
            "result": None,
            "confidence": None,
            "annotated_image": None
        }
        
        self.images_metadata["images"].append(image_meta)
        self.save_images_metadata()
        return image_meta
    
    def add_result(self, image_id: str, result_class: str, confidence: float, annotated_image_path: str, 
                   result_filename: str = None, result_path: str = None) -> Dict:
        """添加识别结果到results缓存"""
        # 确保元数据结构正确
        if not isinstance(self.results_metadata, dict):
            self.results_metadata = {"results": [], "batch_id": 0}
        if "results" not in self.results_metadata:
            self.results_metadata["results"] = []
        if "batch_id" not in self.results_metadata:
            self.results_metadata["batch_id"] = 0
            
        result_id = str(uuid.uuid4())
        
        # 如果提供了自定义文件名和路径，使用它们；否则生成默认的
        if result_filename and result_path:
            final_result_filename = result_filename
            final_result_path = result_path
        else:
            final_result_filename = f"result_{result_id}_{image_id}.jpg"
            final_result_path = os.path.join(self.results_dir, final_result_filename)
            
            # 复制结果图片到results目录
            if os.path.exists(annotated_image_path):
                if CV2_AVAILABLE:
                    img = cv2.imread(annotated_image_path)
                    cv2.imwrite(final_result_path, img)
                else:
                    import shutil
                    shutil.copy2(annotated_image_path, final_result_path)
        
        result_meta = {
            "id": result_id,
            "image_id": image_id,
            "result_class": result_class,
            "confidence": confidence,
            "result_filename": final_result_filename,
            "result_path": final_result_path,
            "created_time": datetime.now().isoformat(),
            "status": "completed"
        }
        
        self.results_metadata["results"].append(result_meta)
        self.save_results_metadata()
        return result_meta
    
    def get_image(self, image_id: str) -> Optional[Dict]:
        """获取指定图片信息"""
        for img in self.images_metadata["images"]:
            if img["id"] == image_id:
                return img
        return None
    
    def get_result(self, result_id: str) -> Optional[Dict]:
        """获取指定结果信息"""
        for result in self.results_metadata["results"]:
            if result["id"] == result_id:
                return result
        return None
    
    def get_results_by_image_id(self, image_id: str) -> List[Dict]:
        """根据图片ID获取所有相关结果"""
        return [result for result in self.results_metadata["results"] if result["image_id"] == image_id]
    
    def update_image_status(self, image_id: str, status: str, result_data: Dict = None):
        """更新图片状态"""
        for img in self.images_metadata["images"]:
            if img["id"] == image_id:
                img["status"] = status
                if result_data:
                    img.update(result_data)
                self.save_images_metadata()
                break
    
    def remove_image(self, image_id: str) -> bool:
        """删除指定图片（包括相关结果）"""
        for i, img in enumerate(self.images_metadata["images"]):
            if img["id"] == image_id:
                # 删除文件
                if os.path.exists(img["path"]):
                    os.remove(img["path"])
                
                # 删除相关结果
                self.remove_results_by_image_id(image_id)
                
                # 从元数据中移除
                del self.images_metadata["images"][i]
                self.save_images_metadata()
                return True
        return False
    
    def remove_image_only(self, image_id: str) -> bool:
        """只删除指定图片（不影响results）"""
        for i, img in enumerate(self.images_metadata["images"]):
            if img["id"] == image_id:
                # 只删除原始图片文件
                if os.path.exists(img["path"]):
                    os.remove(img["path"])
                    print(f"  已删除原始图片: {img['path']}")
                
                # 从元数据中移除（但不删除results）
                del self.images_metadata["images"][i]
                self.save_images_metadata()
                return True
        return False
    
    def remove_result(self, result_id: str) -> bool:
        """删除指定结果"""
        for i, result in enumerate(self.results_metadata["results"]):
            if result["id"] == result_id:
                # 删除文件
                if os.path.exists(result["result_path"]):
                    os.remove(result["result_path"])
                
                # 从元数据中移除
                del self.results_metadata["results"][i]
                self.save_results_metadata()
                return True
        return False
    
    def remove_results_by_image_id(self, image_id: str) -> int:
        """删除指定图片的所有结果"""
        removed_count = 0
        results_to_remove = []
        
        for result in self.results_metadata["results"]:
            if result["image_id"] == image_id:
                # 删除文件
                if os.path.exists(result["result_path"]):
                    os.remove(result["result_path"])
                results_to_remove.append(result)
                removed_count += 1
        
        # 从元数据中移除
        for result in results_to_remove:
            self.results_metadata["results"].remove(result)
        
        if removed_count > 0:
            self.save_results_metadata()
        
        return removed_count
    
    def clear_images_cache(self, status: str = None):
        """清空images缓存（只清空原始图片，不影响results）"""
        try:
            print(f"开始清空images缓存，状态过滤: {status}")
            
            if status:
                to_remove = [img for img in self.images_metadata["images"] if img.get("status") == status]
                print(f"找到 {len(to_remove)} 张状态为 {status} 的图片")
                for img in to_remove:
                    self.remove_image_only(img["id"])
            else:
                all_images = self.images_metadata["images"].copy()
                print(f"找到 {len(all_images)} 张图片需要清空")
                
                for i, img in enumerate(all_images):
                    try:
                        print(f"删除图片 {i+1}/{len(all_images)}: {img.get('cached_filename', 'unknown')}")
                        if os.path.exists(img["path"]):
                            os.remove(img["path"])
                            print(f"  已删除原始图片: {img['path']}")
                    except Exception as e:
                        print(f"删除文件失败 {img.get('path', 'unknown')}: {e}")
                
                print("清空images元数据...")
                self.images_metadata["images"] = []
                self.images_metadata["batch_id"] = 0
                self.save_images_metadata()
                
                print("images缓存清空完成")
                
        except Exception as e:
            print(f"清空images缓存时发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def clear_results_cache(self):
        """清空results缓存"""
        try:
            print("开始清空results缓存")
            
            all_results = self.results_metadata["results"].copy()
            print(f"找到 {len(all_results)} 个结果需要清空")
            
            for i, result in enumerate(all_results):
                try:
                    print(f"删除结果 {i+1}/{len(all_results)}: {result.get('result_filename', 'unknown')}")
                    if os.path.exists(result["result_path"]):
                        os.remove(result["result_path"])
                        print(f"  已删除结果图片: {result['result_path']}")
                except Exception as e:
                    print(f"删除结果文件失败 {result.get('result_path', 'unknown')}: {e}")
            
            print("清空results元数据...")
            self.results_metadata["results"] = []
            self.results_metadata["batch_id"] = 0
            self.save_results_metadata()
            
            print("results缓存清空完成")
            
        except Exception as e:
            print(f"清空results缓存时发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def get_all_images(self, status: str = None) -> List[Dict]:
        """获取所有图片信息"""
        # 确保元数据结构正确
        if not isinstance(self.images_metadata, dict) or "images" not in self.images_metadata:
            return []
        
        if status:
            return [img for img in self.images_metadata["images"] if img.get("status") == status]
        return self.images_metadata["images"]
    
    def get_all_results(self) -> List[Dict]:
        """获取所有结果信息"""
        # 确保元数据结构正确
        if not isinstance(self.results_metadata, dict) or "results" not in self.results_metadata:
            return []
        
        return self.results_metadata["results"]
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        # 确保元数据结构正确
        images = self.images_metadata.get("images", []) if isinstance(self.images_metadata, dict) else []
        results = self.results_metadata.get("results", []) if isinstance(self.results_metadata, dict) else []
        
        total_images = len(images)
        total_results = len(results)
        
        by_status = {}
        for img in images:
            status = img.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        
        # 计算缓存大小
        cache_size = 0
        for img in images:
            try:
                if "path" in img and os.path.exists(img["path"]):
                    cache_size += os.path.getsize(img["path"])
            except:
                pass
        
        for result in results:
            try:
                if "result_path" in result and os.path.exists(result["result_path"]):
                    cache_size += os.path.getsize(result["result_path"])
            except:
                pass
        
        return {
            "total_images": total_images,
            "total_results": total_results,
            "by_status": by_status,
            "cache_size_bytes": cache_size,
            "cache_size_mb": round(cache_size / (1024 * 1024), 2)
        }
    
    def _extract_capture_time(self, image_path: str) -> str:
        """从图片EXIF中提取拍摄时间"""
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if exif is not None:
                    for tag, value in exif.items():
                        if ExifTags.TAGS.get(tag) == 'DateTime':
                            return value
        except Exception as e:
            print(f"提取EXIF信息失败: {e}")
        
        # 如果无法提取EXIF，使用当前时间
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_images_batch(self, images_data: List[bytes], filenames: List[str], source: str = "upload") -> List[Dict]:
        """批量添加图片"""
        results = []
        for image_data, filename in zip(images_data, filenames):
            try:
                result = self.add_image(image_data, filename, source)
                results.append(result)
            except Exception as e:
                print(f"添加图片失败 {filename}: {e}")
                results.append({"error": str(e), "filename": filename})
        return results

# 创建全局缓存管理器实例
cache_manager = LocalCacheManager()