import json
import os
from datetime import datetime
from typing import Any, Optional

import redis


def _redis_client() -> redis.Redis:
    host = os.getenv('REDIS_HOST', '127.0.0.1')
    port = int(os.getenv('REDIS_PORT', '6379'))
    db = int(os.getenv('REDIS_DB', '1'))
    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


def cache_get(key: str) -> Optional[Any]:
    r = _redis_client()
    value = r.get(key)
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def cache_set(key: str, value: Any, expire_seconds: int = 300) -> None:
    r = _redis_client()
    try:
        # 自定义JSON编码器处理datetime对象
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        payload = json.dumps(value, ensure_ascii=False, default=json_serializer)
    except TypeError:
        # Fallback to string
        payload = str(value)
    r.set(key, payload, ex=expire_seconds)


def cache_delete(key: str) -> None:
    """删除指定键的缓存"""
    r = _redis_client()
    r.delete(key)


def cache_delete_pattern(pattern: str) -> None:
    """根据模式删除匹配的缓存键"""
    r = _redis_client()
    keys = r.keys(pattern)
    if keys:
        r.delete(*keys)


def cache_clear_all() -> None:
    """清空所有缓存"""
    r = _redis_client()
    r.flushdb()


