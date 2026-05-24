import json

path = 'notebooks/classification/resnet50-retinal-classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_func = """def crop_image_from_gray(img, tol=7):
    return img"""

for idx, cell in enumerate(data['cells']):
    src = ''.join(cell['source'])
    if 'def crop_image_from_gray(img, tol=7):' in src and 'def contrast_enhancement(img):' in src:
        start = src.find('def crop_image_from_gray(img, tol=7):')
        end = src.find('def contrast_enhancement(img):')
        old_part = src[start:end]
        new_src = src.replace(old_part, new_func + '\n\n\n')
        cell['source'] = [new_src]
        print(f'Fixed cell {idx}')
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print('File saved')

# Verify
with open(path, 'r', encoding='utf-8') as f:
    data2 = json.load(f)
for cell in data2['cells']:
    src = ''.join(cell['source'])
    if 'crop_image_from_gray' in src:
        has_old = 'np.zeros_like' in src
        is_simple = 'return img' in src and 'def crop_image_from_gray' in src[:50]
        print(f'  Old code: {has_old}, Simple no-op: {is_simple}')
        break
