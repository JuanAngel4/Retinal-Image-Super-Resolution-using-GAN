import json

path = 'notebooks/classification/resnet50-retinal-classification.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

old = """def crop_image_from_gray(img, tol=7):
    \"\"\"
    Crop dark borders from retinal fundus images.
    Pixels with grayscale value <= tol are considered border.
    \"\"\"
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        
        img_cropped = np.zeros_like(img)
        for i in range(3):
            img_cropped[:, :, i] = img[:, :, i][np.ix_(mask.any(1), mask.any(0))]
        return img_cropped
    else:
        return img"""

new = """def crop_image_from_gray(img, tol=7):
    \"\"\"
    Crop dark borders from retinal fundus images.
    Pixels with grayscale value <= tol are considered border.
    Returns the actual cropped region (not padded to original size).
    \"\"\"
    if img.ndim == 2:
        mask = img > tol
        if mask.sum() == 0:
            return img
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return img[y1:y2+1, x1:x2+1]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if mask.sum() == 0:
            return img
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return img[y1:y2+1, x1:x2+1, :]
    else:
        return img"""

found = False
for cell in data['cells']:
    combined = ''.join(cell['source'])
    if old in combined:
        cell['source'] = [combined.replace(old, new)]
        found = True
        break

if found:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print('Fix applied successfully!')
else:
    print('ERROR: old code not found in notebook')
