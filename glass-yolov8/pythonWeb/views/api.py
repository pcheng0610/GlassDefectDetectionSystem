from flask import Blueprint, request, jsonify
import requests

api_bp = Blueprint('api', __name__)

@api_bp.route('/openai/chat', methods=['POST'])
def openai_chat():
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"code": 400, "message": "缺少必要参数"})
        
        # 调用阿里云百炼API
        api_key = "sk-e9de66e56cb34f65805cb29fbfb150f3"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
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
        
        # 使用阿里云百炼API端点
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
