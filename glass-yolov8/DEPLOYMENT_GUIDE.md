# 大模型镜片瑕疵识别系统 - 开发与部署操作文档

## 目录

- [一、快速开始](#一快速开始)
- [二、开发环境配置](#二开发环境配置)
- [三、部署流程](#三部署流程)
- [四、运维监控](#四运维监控)
- [五、常见问题](#五常见问题)

---

## 一、快速开始

### 1.1 系统启动（开发环境）

#### 启动后端服务
```bash
# 进入后端目录
cd e:\zjzjzj\glass-yolov8

# 启动后端服务
python app.py
```

后端服务运行在：`http://127.0.0.1:5000/`

#### 启动前端服务
```bash
# 进入前端目录
cd e:\zjzjzj\glass-yolov8-v2.0

# 安装依赖（首次运行）
npm install

# 启动前端开发服务器
npm run dev
```

前端服务运行在：`http://localhost:3000/`

#### 访问系统
打开浏览器访问：`http://localhost:3000/`

---

## 二、开发环境配置

### 2.1 前置要求

| 软件 | 版本 | 说明 |
|------|------|------|
| **Python** | 3.10+ | 后端运行环境 |
| **Node.js** | 18+ | 前端构建工具 |
| **CUDA** | 12.1 | GPU 加速（可选） |
| **MySQL** | 8.0+ | 数据库 |
| **Redis** | 7.0+ | 缓存服务 |

### 2.2 后端开发环境

#### 步骤 1：创建虚拟环境
```bash
cd e:\zjzjzj\glass-yolov8
python -m venv venv
```

#### 步骤 2：激活虚拟环境
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 步骤 3：安装依赖
```bash
pip install -r requirements.txt
```

#### 步骤 4：配置环境变量
创建 `.env` 文件：
```bash
# Flask 配置
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key-here

# 数据库配置
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_USER=root
DATABASE_PASSWORD=your-password
DATABASE_NAME=lens_db

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379

# OSS 配置（可选）
OSS_ACCESS_KEY_ID=your-access-key
OSS_ACCESS_KEY_SECRET=your-secret
OSS_BUCKET=lens-images
OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com

# 模型配置
MODEL_PATH=runs/detect/yolov8m_glass_detection/weights/best.pt
MODEL_CONFIDENCE_THRESHOLD=0.5
```

#### 步骤 5：初始化数据库
```bash
python scripts/init_db.py
```

#### 步骤 6：启动后端
```bash
python app.py
```

### 2.3 前端开发环境

#### 步骤 1：安装 Node.js
下载并安装：https://nodejs.org/

#### 步骤 2：安装依赖
```bash
cd e:\zjzjzj\glass-yolov8-v2.0
npm install
```

#### 步骤 3：配置环境变量
创建 `.env` 文件：
```bash
# API 地址
VITE_API_BASE_URL=http://localhost:5000

# 开发环境
NODE_ENV=development
```

#### 步骤 4：启动开发服务器
```bash
npm run dev
```

#### 步骤 5：代码构建（生产环境）
```bash
npm run build
```

### 2.4 代码规范

#### 前端代码规范
```bash
# 安装 ESLint 和 Prettier
npm install --save-dev eslint prettier

# 运行代码检查
npm run lint

# 自动修复格式问题
npx prettier --write src/
```

#### 后端代码规范
```bash
# 安装代码检查工具
pip install black flake8

# 格式化代码
black .

# 检查代码质量
flake8 .
```

### 2.5 Git 工作流

#### 分支管理
```bash
# 查看分支
git branch

# 创建新功能分支
git checkout -b feature/your-feature

# 切换到开发分支
git checkout develop

# 切换到主分支
git checkout main
```

#### 提交代码
```bash
# 添加文件
git add .

# 提交代码（遵循规范）
git commit -m "feat: 添加用户登录功能"

# 推送分支
git push origin feature/your-feature
```

#### Commit 信息规范
```
类型 (范围): 主题

正文（可选）

页脚（可选）

类型说明:
- feat: 新功能
- fix: 修复 bug
- docs: 文档更新
- style: 代码格式
- refactor: 重构
- test: 测试
- chore: 构建/工具
```

---

## 三、部署流程

### 3.1 生产环境准备

#### 服务器要求
| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **CPU** | 4 核 | 8 核 |
| **内存** | 16GB | 32GB |
| **存储** | 200GB SSD | 500GB SSD |
| **GPU** | - | RTX 4090 (可选) |

### 3.2 后端部署

#### 步骤 1：安装系统依赖
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip nginx

# 安装 CUDA（如需 GPU 加速）
# 参考：https://developer.nvidia.com/cuda-downloads
```

#### 步骤 2：克隆代码
```bash
git clone <repository-url> /opt/lens-system
cd /opt/lens-system
```

#### 步骤 3：创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 步骤 4：安装依赖
```bash
pip install -r requirements.txt
```

#### 步骤 5：配置环境变量
```bash
cp .env.example .env
vim .env  # 编辑生产环境配置
```

#### 步骤 6：安装 Gunicorn
```bash
pip install gunicorn
```

#### 步骤 7：创建 systemd 服务
```bash
sudo vim /etc/systemd/system/lens-backend.service
```

内容：
```ini
[Unit]
Description=Lens Detection Backend
After=network.target mysql.service redis.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/lens-system
Environment="PATH=/opt/lens-system/venv/bin"
ExecStart=/opt/lens-system/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 步骤 8：启动服务
```bash
sudo systemctl daemon-reload
sudo systemctl enable lens-backend
sudo systemctl start lens-backend
sudo systemctl status lens-backend
```

### 3.3 前端部署

#### 步骤 1：构建前端
```bash
cd /opt/lens-system/glass-yolov8-v2.0
npm install
npm run build
```

#### 步骤 2：配置 Nginx
```bash
sudo vim /etc/nginx/sites-available/lens
```

内容：
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态资源
    location / {
        root /opt/lens-system/glass-yolov8-v2.0/dist;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API 代理
    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # 静态资源缓存
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip 压缩
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
}
```

#### 步骤 3：启用配置
```bash
sudo ln -s /etc/nginx/sites-available/lens /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 3.4 HTTPS 配置（可选但推荐）

#### 使用 Let's Encrypt
```bash
# 安装 Certbot
sudo apt-get install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo certbot renew --dry-run
```

### 3.5 数据库部署

#### 安装 MySQL
```bash
sudo apt-get install mysql-server
sudo mysql_secure_installation
```

#### 创建数据库和用户
```sql
CREATE DATABASE lens_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'lens_user'@'localhost' IDENTIFIED BY 'your-password';
GRANT ALL PRIVILEGES ON lens_db.* TO 'lens_user'@'localhost';
FLUSH PRIVILEGES;
```

#### 配置主从复制（高可用）
```sql
-- 主服务器配置
[mysqld]
server-id=1
log-bin=mysql-bin
binlog-format=ROW

-- 从服务器配置
[mysqld]
server-id=2
relay-log=mysql-relay-bin
```

### 3.6 Redis 部署

#### 安装 Redis
```bash
sudo apt-get install redis-server
sudo systemctl enable redis
sudo systemctl start redis
```

#### 配置 Redis
```bash
sudo vim /etc/redis/redis.conf
```

关键配置：
```
bind 127.0.0.1
port 6379
requirepass your-password
maxmemory 2gb
maxmemory-policy allkeys-lru
```

### 3.7 自动化部署脚本

创建 `deploy.sh`：
```bash
#!/bin/bash

set -e

echo "=== 开始部署 ==="

# 1. 拉取代码
echo "1. 拉取代码..."
cd /opt/lens-system
git pull origin main

# 2. 前端构建
echo "2. 构建前端..."
cd glass-yolov8-v2.0
npm install
npm run build
cd ..

# 3. 后端依赖
echo "3. 安装后端依赖..."
source venv/bin/activate
pip install -r requirements.txt

# 4. 数据库迁移
echo "4. 执行数据库迁移..."
python scripts/migrate_db.py

# 5. 重启服务
echo "5. 重启服务..."
sudo systemctl restart lens-backend
sudo systemctl reload nginx

# 6. 健康检查
echo "6. 健康检查..."
sleep 10
curl -f http://localhost:5000/api/health || exit 1

echo "=== 部署完成 ==="
```

使用：
```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 四、运维监控

### 4.1 日志管理

#### 查看日志
```bash
# 后端日志
sudo journalctl -u lens-backend -f

# Nginx 日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# 应用日志
tail -f /opt/lens-system/logs/app.log
```

#### 日志轮转
```bash
sudo vim /etc/logrotate.d/lens
```

内容：
```
/opt/lens-system/logs/*.log {
    daily
    rotate 10
    compress
    delaycompress
    missingok
    notifempty
    create 0640 www-data www-data
}
```

### 4.2 监控指标

#### 系统监控
```bash
# CPU 使用率
top

# 内存使用率
free -h

# 磁盘使用率
df -h

# GPU 使用率（如有）
nvidia-smi
```

#### 应用监控
```bash
# 检查服务状态
sudo systemctl status lens-backend
sudo systemctl status nginx

# 检查端口
netstat -tlnp | grep :5000
netstat -tlnp | grep :80

# 检查进程
ps aux | grep gunicorn
ps aux | grep nginx
```

### 4.3 备份策略

#### 数据库备份
```bash
#!/bin/bash
# backup_db.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/mysql"
mkdir -p $BACKUP_DIR

mysqldump -u lens_user -p'your-password' lens_db > $BACKUP_DIR/lens_db_$DATE.sql

# 删除 30 天前的备份
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
```

#### 定时备份
```bash
crontab -e

# 每天凌晨 2 点备份
0 2 * * * /opt/lens-system/scripts/backup_db.sh
```

### 4.4 故障排查

#### 常见问题诊断

**问题 1：后端服务无法启动**
```bash
# 检查日志
sudo journalctl -u lens-backend -n 50

# 检查端口占用
lsof -i :5000

# 检查依赖
pip list | grep -E "flask|torch|opencv"
```

**问题 2：前端页面无法访问**
```bash
# 检查 Nginx 状态
sudo systemctl status nginx

# 检查 Nginx 配置
sudo nginx -t

# 检查防火墙
sudo ufw status
```

**问题 3：数据库连接失败**
```bash
# 检查 MySQL 状态
sudo systemctl status mysql

# 测试连接
mysql -u lens_user -p lens_db

# 检查连接数
mysql -u root -p -e "SHOW PROCESSLIST;"
```

**问题 4：GPU 不可用**
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 PyTorch 是否识别 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 五、常见问题

### 5.1 安装问题

**Q: pip install 失败**
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或配置永久镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: npm install 失败**
```bash
# 使用淘宝镜像
npm config set registry https://registry.npmmirror.com
npm install
```

### 5.2 运行问题

**Q: 相机无法连接**
1. 检查相机是否已正确连接 USB 接口
2. 检查相机驱动是否已安装
3. 查看后端日志中的相机错误信息
4. 尝试重启相机服务

**Q: AI 识别速度慢**
1. 检查 GPU 是否被正确使用：`nvidia-smi`
2. 降低输入图片分辨率
3. 减少并发请求数量
4. 升级 GPU 硬件

**Q: 内存不足**
```bash
# 查看内存使用
free -h

# 减少 Gunicorn worker 数量
# 编辑 /etc/systemd/system/lens-backend.service
ExecStart=... --workers 2 ...

# 添加 swap 空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 5.3 部署问题

**Q: Nginx 502 Bad Gateway**
```bash
# 检查后端是否运行
sudo systemctl status lens-backend

# 检查 Nginx 配置
sudo nginx -t

# 查看错误日志
sudo tail -f /var/log/nginx/error.log
```

**Q: 跨域问题**
```nginx
# 在 Nginx 配置中添加
location /api/ {
    add_header Access-Control-Allow-Origin *;
    add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
    add_header Access-Control-Allow-Headers 'Content-Type';
    
    if ($request_method = 'OPTIONS') {
        return 204;
    }
}
```

### 5.4 性能优化

**Q: 提高并发能力**
```ini
# 增加 Gunicorn worker 数量
[Service]
ExecStart=/opt/lens-system/venv/bin/gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 8 \
    --worker-class gthread \
    --threads 4 \
    app:app
```

**Q: 优化数据库性能**
```sql
-- 添加索引
CREATE INDEX idx_create_time ON on_line(create_time);
CREATE INDEX idx_result_class ON history(result_class);

-- 优化查询
EXPLAIN SELECT * FROM on_line WHERE create_time > '2024-01-01';
```

**Q: 缓存优化**
```python
# 使用 Redis 缓存识别结果
from redis import Redis
redis_client = Redis(host='localhost', port=6379, db=0)

# 设置缓存
redis_client.setex(f"result:{image_id}", 3600, result_json)

# 获取缓存
cached_result = redis_client.get(f"result:{image_id}")
```

---

## 附录

### A. 端口说明

| 端口 | 服务 | 说明 |
|------|------|------|
| 80 | Nginx | HTTP 访问 |
| 443 | Nginx | HTTPS 访问 |
| 5000 | Flask | 后端 API |
| 3000 | Vite | 前端开发 |
| 3306 | MySQL | 数据库 |
| 6379 | Redis | 缓存 |

### B. 目录结构

```
/opt/lens-system/
├── glass-yolov8/              # 后端代码
│   ├── pythonWeb/            # Flask 应用
│   ├── runs/                 # 模型文件
│   ├── utils/                # 工具类
│   ├── requirements.txt      # 依赖列表
│   └── app.py                # 入口文件
├── glass-yolov8-v2.0/        # 前端代码
│   ├── src/                  # 源代码
│   ├── public/               # 静态资源
│   └── dist/                 # 构建输出
├── logs/                     # 日志文件
├── scripts/                  # 运维脚本
└── .env                      # 环境变量
```

### C. 常用命令速查

```bash
# 后端服务管理
sudo systemctl start lens-backend
sudo systemctl stop lens-backend
sudo systemctl restart lens-backend
sudo systemctl status lens-backend

# Nginx 管理
sudo systemctl start nginx
sudo systemctl stop nginx
sudo systemctl restart nginx
sudo nginx -t

# 数据库管理
sudo systemctl start mysql
sudo systemctl stop mysql
sudo systemctl restart mysql

# Redis 管理
sudo systemctl start redis
sudo systemctl stop redis
sudo systemctl restart redis

# 查看日志
sudo journalctl -u lens-backend -f
sudo tail -f /var/log/nginx/error.log

# 进入虚拟环境
cd /opt/lens-system && source venv/bin/activate
```

### D. 紧急联系人

| 角色 | 联系方式 | 职责 |
|------|---------|------|
| 系统管理员 | admin@example.com | 系统运维 |
| 开发负责人 | dev-lead@example.com | 技术支持 |
| 数据库管理员 | dba@example.com | 数据库问题 |

---

**文档版本**: v1.0  
**最后更新**: 2026-03-24  
**维护团队**: 技术部
