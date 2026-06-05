"""
阿里云OSS配置
"""
import os

# OSS配置
OSS_CONFIG = {
    'ACCESS_KEY_ID': os.getenv('ALIYUN_OSS_ACCESS_KEY_ID', os.getenv('OSS_ACCESS_KEY_ID', '')),
    'ACCESS_KEY_SECRET': os.getenv('ALIYUN_OSS_ACCESS_KEY_SECRET', os.getenv('OSS_ACCESS_KEY_SECRET', '')),
    'BUCKET_NAME': os.getenv('ALIYUN_OSS_BUCKET_NAME', os.getenv('OSS_BUCKET', 'glass-yolo')),
    'ENDPOINT': os.getenv('ALIYUN_OSS_ENDPOINT', os.getenv('OSS_ENDPOINT', 'https://oss-cn-guangzhou.aliyuncs.com')),
    'REGION': os.getenv('ALIYUN_OSS_REGION', 'cn-guangzhou')
}

# 文件夹映射
FOLDER_MAPPING = {
    'root': 'root',      # 原始图片（拍照）
    'result': 'result',  # 识别结果图片
    'tmp': 'tmp',        # 临时文件
    'uploads': 'uploads' # 上传文件（本地）
}

def get_oss_config():
    """获取OSS配置"""
    return OSS_CONFIG.copy()

def get_folder_mapping():
    """获取文件夹映射"""
    return FOLDER_MAPPING.copy()
