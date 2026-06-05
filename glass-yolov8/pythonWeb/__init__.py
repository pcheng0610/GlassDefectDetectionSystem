from flask import Flask, request, redirect, session, render_template, jsonify
from flask_cors import CORS
import requests


#拦截器（已禁用，允许直接访问所有页面）
def auth():
    return None


def create_app():
    import os
    static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pythonWeb', 'static')
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pythonWeb', 'templates')
    app = Flask(__name__, static_folder=static_path, static_url_path='/static', template_folder=template_path)
    #使用这个密钥来加密会话数据（session）
    app.secret_key='your-unique-secret-key-here'
    
    # 配置CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # 预加载模型进程池（Windows 下需确保在主进程中调用）
    try:
        from utils.model_pool import initialize_model_pool
        
        # 使用相对路径，优先查找顺序
        base_dir = os.path.dirname(os.path.dirname(__file__))  # 项目根目录
        model_candidates = [
            os.path.join(base_dir, "runs", "detect", "yolov8m_glass_detection", "weights", "best.pt")
        ]
        
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                print(f"[model_pool] 找到模型文件: {model_path}")
                break
        
        if model_path is None:
            raise FileNotFoundError(f"未找到任何可用的模型文件。查找路径: {model_candidates}")
        
        initialize_model_pool(model_path, processes=1)
        print(f"[model_pool] 模型进程池初始化成功: {model_path}")
        
    except Exception as e:
        # 延迟失败，不影响应用启动，但记录错误
        print(f"[model_pool] 初始化失败: {e}")
        import traceback
        traceback.print_exc()

    #把蓝图注入app里面
    from .views import account
    app.register_blueprint(account.ac)

    from .views import home
    app.register_blueprint(home.ho)

    from .views import recognition
    app.register_blueprint(recognition.re)

    from .views import history
    app.register_blueprint(history.hi)

    from .views import local
    app.register_blueprint(local.lo)

    from .views import select_fan
    app.register_blueprint(select_fan.se)

    # 批量处理接口
    from .views import batch_processing
    app.register_blueprint(batch_processing.batch_bp)

    # 在线处理接口
    from .views import online_processing
    app.register_blueprint(online_processing.online_bp)

    # 多模态暗场检测接口
    from .views import multimodal_api
    app.register_blueprint(multimodal_api.multimodal_bp)

    # 信息可视化接口
    from .views import visualization
    app.register_blueprint(visualization.vi)



    # 添加根路径的重定向路由
    @app.route('/')
    def index():
        """处理根路径，直接重定向到登录页"""
        return redirect('/login')
    
    # 测试路由
    @app.route('/test')
    def test_page():
        return render_template("visualization.html")
    
    # 可视化页面路由
    @app.route('/visualization')
    def visualization_page():
        return render_template("visualization.html")
    
    # 测试API端点
    @app.route('/api/test', methods=['POST'])
    def test_api():
        return jsonify({
            "code": 200,
            "data": {
                "response": "API测试成功！"
            }
        })
    
    # AI 助手 API 代理
    @app.route('/api/openai/chat', methods=['POST'])
    def openai_chat():
        try:
            data = request.json
            if not data or "message" not in data:
                return jsonify({"code": 400, "message": "缺少必要参数"})
            
            # 调用AI API
            api_key = "sk-e9de66e56cb34f65805cb29fbfb150f3"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "qwen-plus",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的镜片检测智能助手，精通数据分析、报告生成和质量优化建议。你的职责是帮助用户分析镜片检测数据、生成专业报告并提供有价值的优化建议。请用简洁专业的中文回答。"
                    },
                    {
                        "role": "user",
                        "content": data["message"]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # 使用阿里云百炼API端点（华北2北京）
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result and "choices" in result and result["choices"]:
                return jsonify({
                    "code": 200,
                    "data": {
                        "response": result["choices"][0]["message"]["content"]
                    }
                })
            else:
                return jsonify({"code": 400, "message": "API返回格式异常"})
                
        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            return jsonify({"code": 500, "message": f"API调用失败: {str(e)}"})
        except Exception as e:
            print(f"处理请求时发生错误: {e}")
            return jsonify({"code": 500, "message": f"处理失败: {str(e)}"})
    




    #使用拦截器
    app.before_request(auth)

    return app