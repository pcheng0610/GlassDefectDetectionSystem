"""
阿里云OSS存储工具类
"""
import os
import oss2
from datetime import datetime
from typing import Optional, Tuple
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

class OSSClient:
    def __init__(self, access_key_id: str, access_key_secret: str, 
                 bucket_name: str, endpoint: str):
        """
        初始化OSS客户端
        
        Args:
            access_key_id: 阿里云AccessKey ID
            access_key_secret: 阿里云AccessKey Secret
            bucket_name: 存储空间名称
            endpoint: OSS实例地址
        """
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        
        # 创建认证对象
        auth = oss2.Auth(access_key_id, access_key_secret)
        
        # 创建Bucket对象
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
        
        # 确保bucket存在
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """确保bucket存在"""
        try:
            self.bucket.get_bucket_info()
        except oss2.exceptions.NoSuchBucket:
            # 如果bucket不存在，创建它
            self.bucket.create_bucket()
    
    def upload_image(self, image_data: np.ndarray, object_key: str, 
                    quality: int = 95) -> str:
        """
        上传图片到OSS
        
        Args:
            image_data: OpenCV图像数据 (numpy array)
            object_key: OSS对象键名
            quality: JPEG压缩质量 (1-100)
            
        Returns:
            str: 图片的OSS URL
        """
        try:
            # 将OpenCV图像转换为JPEG字节流
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, encoded_img = cv2.imencode('.jpg', image_data, encode_param)
            
            if not success:
                raise ValueError("Failed to encode image")
            
            # 上传到OSS
            self.bucket.put_object(object_key, encoded_img.tobytes())
            
            # 返回公开访问的URL
            return f"https://{self.bucket_name}.{self.endpoint.replace('https://', '')}/{object_key}"
            
        except Exception as e:
            raise Exception(f"Failed to upload image to OSS: {str(e)}")
    
    def upload_file(self, file_path: str, object_key: str) -> str:
        """
        上传文件到OSS
        
        Args:
            file_path: 本地文件路径
            object_key: OSS对象键名
            
        Returns:
            str: 文件的OSS URL
        """
        try:
            with open(file_path, 'rb') as f:
                self.bucket.put_object(object_key, f)
            
            return f"https://{self.bucket_name}.{self.endpoint.replace('https://', '')}/{object_key}"
            
        except Exception as e:
            raise Exception(f"Failed to upload file to OSS: {str(e)}")
    
    def download_image(self, object_key: str) -> Optional[np.ndarray]:
        """
        从OSS下载图片
        
        Args:
            object_key: OSS对象键名
            
        Returns:
            numpy array: OpenCV图像数据，如果失败返回None
        """
        try:
            # 从OSS获取对象
            result = self.bucket.get_object(object_key)
            image_data = result.read()
            
            # 将字节流转换为OpenCV图像
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            print(f"Failed to download image from OSS: {str(e)}")
            return None
    
    def file_exists(self, object_key: str) -> bool:
        """
        检查文件是否存在于OSS
        
        Args:
            object_key: OSS对象键名
            
        Returns:
            bool: 文件是否存在
        """
        try:
            self.bucket.head_object(object_key)
            return True
        except oss2.exceptions.NoSuchKey:
            return False
        except Exception:
            return False
    
    def delete_file(self, object_key: str) -> bool:
        """
        从OSS删除文件
        
        Args:
            object_key: OSS对象键名
            
        Returns:
            bool: 删除是否成功
        """
        try:
            self.bucket.delete_object(object_key)
            return True
        except Exception as e:
            print(f"Failed to delete file from OSS: {str(e)}")
            return False
    
    def get_file_url(self, object_key: str) -> str:
        """
        获取文件的公开访问URL
        
        Args:
            object_key: OSS对象键名
            
        Returns:
            str: 文件的公开访问URL
        """
        return f"https://{self.bucket_name}.{self.endpoint.replace('https://', '')}/{object_key}"
    
    def generate_object_key(self, folder: str, filename: str) -> str:
        """
        生成OSS对象键名
        
        Args:
            folder: 文件夹名称 (如 'root', 'result', 'tmp')
            filename: 文件名
            
        Returns:
            str: OSS对象键名
        """
        return f"{folder}/{filename}"


# 全局OSS客户端实例
_oss_client = None

def get_oss_client() -> OSSClient:
    """获取OSS客户端实例"""
    global _oss_client
    if _oss_client is None:
        try:
            from pythonWeb.config.oss_config import get_oss_config
            config = get_oss_config()
        except ImportError:
            # 如果无法导入配置，使用环境变量配置
            config = {
                'ACCESS_KEY_ID': os.getenv('ALIYUN_OSS_ACCESS_KEY_ID', os.getenv('OSS_ACCESS_KEY_ID', '')),
                'ACCESS_KEY_SECRET': os.getenv('ALIYUN_OSS_ACCESS_KEY_SECRET', os.getenv('OSS_ACCESS_KEY_SECRET', '')),
                'BUCKET_NAME': os.getenv('ALIYUN_OSS_BUCKET_NAME', os.getenv('OSS_BUCKET', 'glass-yolo')),
                'ENDPOINT': os.getenv('ALIYUN_OSS_ENDPOINT', os.getenv('OSS_ENDPOINT', 'https://oss-cn-guangzhou.aliyuncs.com'))
            }
        _oss_client = OSSClient(
            access_key_id=config['ACCESS_KEY_ID'],
            access_key_secret=config['ACCESS_KEY_SECRET'],
            bucket_name=config['BUCKET_NAME'],
            endpoint=config['ENDPOINT']
        )
    return _oss_client

def upload_captured_image(image_data: np.ndarray, filename: str) -> str:
    """
    上传捕获的图片到OSS
    
    Args:
        image_data: OpenCV图像数据
        filename: 文件名
        
    Returns:
        str: 图片的OSS URL
    """
    oss_client = get_oss_client()
    object_key = oss_client.generate_object_key("root", filename)
    return oss_client.upload_image(image_data, object_key)

def upload_result_image(image_data: np.ndarray, filename: str) -> str:
    """
    上传识别结果图片到OSS
    
    Args:
        image_data: OpenCV图像数据
        filename: 文件名
        
    Returns:
        str: 图片的OSS URL
    """
    oss_client = get_oss_client()
    object_key = oss_client.generate_object_key("result", filename)
    return oss_client.upload_image(image_data, object_key)

def upload_file_to_oss(file_path: str, folder: str, filename: str) -> str:
    """
    上传文件到OSS指定文件夹
    
    Args:
        file_path: 本地文件路径
        folder: OSS文件夹名称
        filename: 文件名
        
    Returns:
        str: 文件的OSS URL
    """
    oss_client = get_oss_client()
    object_key = oss_client.generate_object_key(folder, filename)
    return oss_client.upload_file(file_path, object_key)

def get_image_url(folder: str, filename: str) -> str:
    """
    获取图片的OSS URL
    
    Args:
        folder: 文件夹名称
        filename: 文件名
        
    Returns:
        str: 图片的OSS URL
    """
    oss_client = get_oss_client()
    object_key = oss_client.generate_object_key(folder, filename)
    return oss_client.get_file_url(object_key)

def check_image_exists(folder: str, filename: str) -> bool:
    """
    检查图片是否存在于OSS
    
    Args:
        folder: 文件夹名称
        filename: 文件名
        
    Returns:
        bool: 图片是否存在
    """
    oss_client = get_oss_client()
    object_key = oss_client.generate_object_key(folder, filename)
    return oss_client.file_exists(object_key)
