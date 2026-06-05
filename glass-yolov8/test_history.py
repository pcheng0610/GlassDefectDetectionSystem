from pythonWeb import create_app
import json

app = create_app()

with app.test_client() as client:
    # 测试历史记录列表 API
    print("Testing /api/history/list API...")
    response = client.get('/api/history/list')
    print(f"Status code: {response.status_code}")
    print(f"Content type: {response.content_type}")
    
    # 解析 JSON 响应
    try:
        data = json.loads(response.data)
        print(f"JSON data: {json.dumps(data, indent=2, ensure_ascii=False)}")
        print(f"Total records: {data.get('total', 0)}")
        print(f"Records: {len(data.get('data', []))}")
    except json.JSONDecodeError:
        print("Response is not valid JSON")
        print(f"Response content: {response.data[:500]}")
    
    print()
    
    # 测试历史记录页面
    print("Testing /history page...")
    response = client.get('/history')
    print(f"Status code: {response.status_code}")
    print(f"Content type: {response.content_type}")
    print(f"Content length: {len(response.data)}")
    print("Page loaded successfully!")
