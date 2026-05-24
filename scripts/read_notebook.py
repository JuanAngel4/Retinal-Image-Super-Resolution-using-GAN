import json

with open('notebooks/Transfer_learnig_prediction_DL_model.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, cell in enumerate(data['cells']):
    print(f'\n=== Cell {i} ({cell["cell_type"]}) ===')
    source = cell.get('source', [])
    for line in source:
        print(line, end='')
