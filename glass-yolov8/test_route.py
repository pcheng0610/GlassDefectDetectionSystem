from pythonWeb import create_app

app = create_app()

with app.test_client() as client:
    print("Testing /history route...")
    response = client.get('/history')
    print(f"Status code: {response.status_code}")
    print(f"Content type: {response.content_type}")
    print(f"Content length: {len(response.data)}")
    print(f"Response data (first 500 chars): {response.data[:500]}")
    print()
    
    print("Testing /api/history/list route...")
    response = client.get('/api/history/list')
    print(f"Status code: {response.status_code}")
    print(f"Content type: {response.content_type}")
    print(f"Content length: {len(response.data)}")
    print(f"Response data (first 500 chars): {response.data[:500]}")
    print()
    
    print("Testing /home route...")
    response = client.get('/home')
    print(f"Status code: {response.status_code}")
    print(f"Content type: {response.content_type}")
    print(f"Content length: {len(response.data)}")
    print(f"Response data (first 500 chars): {response.data[:500]}")
