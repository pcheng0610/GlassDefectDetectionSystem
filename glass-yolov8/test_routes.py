from pythonWeb import create_app

app = create_app()

print('=== 路由列表 ===')
for rule in app.url_map.iter_rules():
    print(f'{rule} -> {rule.endpoint}')
