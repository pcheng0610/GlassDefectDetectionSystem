#数据库连接池
import pymysql
from pymysql import cursors
from dbutils.pooled_db import PooledDB
import hashlib
import json
# from .redis_cache import cache_get, cache_set, cache_delete_pattern

POOL = PooledDB(
    creator=pymysql,
    maxconnections=10,
    mincached=2,
    maxcached=10,
    blocking=True,
    setsession=[],
    ping=0,
    host='127.0.0.1',
    port=3307,
    user='root',
    password='88888888',
    charset='utf8',
    db='challenge',
)

def _generate_cache_key(sql, params):
    """生成缓存键"""
    # 将SQL和参数组合生成唯一键
    key_data = f"{sql}:{json.dumps(params, sort_keys=True) if params else 'None'}"
    return f"db_cache:{hashlib.md5(key_data.encode()).hexdigest()}"

def _is_select_query(sql):
    """判断是否为SELECT查询"""
    return sql.strip().upper().startswith('SELECT')

def _get_table_name_from_sql(sql):
    """从SQL中提取表名（简单实现）"""
    sql_upper = sql.upper()
    if 'FROM' in sql_upper:
        # 简单提取FROM后的表名
        parts = sql_upper.split('FROM')
        if len(parts) > 1:
            table_part = parts[1].split()[0]
            return table_part.lower()
    return None

def _invalidate_table_cache(table_name):
    """清除指定表的所有缓存"""
    pass

#查询一条信息（带缓存）
def fetch_one(sql, params, cache_seconds=300):
    return _fetch_one_direct(sql, params)
    
    return result

#查询所有（带缓存）
def fetch_all(sql, params, cache_seconds=300):
    return _fetch_all_direct(sql, params)

#直接查询一条信息（不缓存）
def _fetch_one_direct(sql, params):
    connection = POOL.connection()
    cursor = connection.cursor(cursor=cursors.DictCursor)
    cursor.execute(sql, params)
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result

#直接查询所有（不缓存）
def _fetch_all_direct(sql, params):
    connection = POOL.connection()
    cursor = connection.cursor(cursor=cursors.DictCursor)
    cursor.execute(sql, params)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result

#插入（清除相关缓存）
def insert(sql, params):
    connection = POOL.connection()
    cursor = connection.cursor(cursor=cursors.DictCursor)
    cursor.execute(sql, params)
    connection.commit()
    cursor.close()
    connection.close()
    
    # 清除相关表的缓存
    table_name = _get_table_name_from_sql(sql)
    _invalidate_table_cache(table_name)

#更新（清除相关缓存）
def update(sql, params):
    connection = POOL.connection()
    cursor = connection.cursor(cursor=cursors.DictCursor)
    cursor.execute(sql, params)
    connection.commit()
    cursor.close()
    connection.close()
    
    # 清除相关表的缓存
    table_name = _get_table_name_from_sql(sql)
    _invalidate_table_cache(table_name)

#删除（清除相关缓存）
def delete(sql, params):
    connection = POOL.connection()
    cursor = connection.cursor(cursor=cursors.DictCursor)
    cursor.execute(sql, params)
    connection.commit()
    cursor.close()
    connection.close()
    
    # 清除相关表的缓存
    table_name = _get_table_name_from_sql(sql)
    _invalidate_table_cache(table_name)

#通用执行（自动判断操作类型并处理缓存）
def execute(sql, params):
    sql_upper = sql.strip().upper()
    if sql_upper.startswith('SELECT'):
        return fetch_all(sql, params)
    elif sql_upper.startswith('INSERT'):
        insert(sql, params)
        return None
    elif sql_upper.startswith('UPDATE'):
        update(sql, params)
        return None
    elif sql_upper.startswith('DELETE'):
        delete(sql, params)
        return None
    else:
        # 其他操作直接执行
        connection = POOL.connection()
        cursor = connection.cursor(cursor=cursors.DictCursor)
        cursor.execute(sql, params)
        connection.commit()
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return result

# 事务支持类
class Transaction:
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def begin(self):
        """开始事务"""
        self.connection = POOL.connection()
        self.cursor = self.connection.cursor(cursor=cursors.DictCursor)
        # 开始事务
        self.cursor.execute("START TRANSACTION")
    
    def commit(self):
        """提交事务"""
        if self.connection and self.cursor:
            self.connection.commit()
            self.cursor.close()
            self.connection.close()
            self.connection = None
            self.cursor = None
    
    def rollback(self):
        """回滚事务"""
        if self.connection and self.cursor:
            self.connection.rollback()
            self.cursor.close()
            self.connection.close()
            self.connection = None
            self.cursor = None
    
    def execute(self, sql, params):
        """在事务中执行SQL"""
        if self.cursor:
            self.cursor.execute(sql, params)
            return self.cursor.fetchall()
        else:
            raise Exception("事务未开始")
    
    def insert(self, sql, params):
        """在事务中插入数据"""
        if self.cursor:
            self.cursor.execute(sql, params)
        else:
            raise Exception("事务未开始")

# 创建全局事务对象
connection = Transaction()



