import http.client

try:
    conn = http.client.HTTPConnection('localhost', 5000, timeout=10)
    conn.request('GET', '/history')
    response = conn.getresponse()
    print(f"状态码: {response.status}")
    print(f"响应头: {response.getheaders()}")
    print(f"响应内容长度: {len(response.read())}")
    conn.close()
    print("测试成功！")
except Exception as e:
    print(f"测试失败: {e}")
