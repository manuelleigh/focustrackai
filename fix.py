import os

for path in ['app.py', 'ui/charts.py']:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('width=" stretch\\', 'width="stretch"')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
