import json

path = 'notebooks/classification/resnet50-retinal-classification.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

test_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## Edite correctamente', '', 'Este texto confirma que el archivo fue modificado correctamente.']
}

data['cells'].append(test_cell)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print(f'OK - Total cells: {len(data["cells"])}')
