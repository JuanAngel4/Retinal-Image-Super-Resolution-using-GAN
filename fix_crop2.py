import json, sys

path = 'notebooks/classification/resnet50-retinal-classification.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

old_start = "def crop_image_from_gray(img, tol=7):"
new_code = "def crop_image_from_gray(img, tol=7):\n    # Disabled - was causing shape errors\n    return img"

modified = False
for cell in data['cells']:
    src = ''.join(cell['source'])
    if old_start in src and 'return img' in src:
        lines = src.split('\n')
        new_lines = []
        in_func = False
        for line in lines:
            if line.startswith('def crop_image_from_gray'):
                in_func = True
                new_lines.append("def crop_image_from_gray(img, tol=7):")
                new_lines.append("    # Disabled - was causing shape errors")
                new_lines.append("    return img")
            elif in_func:
                if line.startswith('def '):
                    in_func = False
                    new_lines.append(line)
                elif line.strip() == '':
                    continue
            else:
                new_lines.append(line)
        
        new_src = '\n'.join(new_lines)
        cell['source'] = [new_src]
        modified = True
        print(f'Modified cell: {cell["cell_type"]}')
        break

if modified:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print('File written successfully')
else:
    print('ERROR: No changes made')
