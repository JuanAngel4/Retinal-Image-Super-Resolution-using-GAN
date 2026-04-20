import os
import csv

def how_many_files(ruta_carpeta):
    """
    Cuenta la cantidad de archivos dentro de una carpeta (no incluye subdirectorios)
    """
    return sum(
        1 for elemento in os.listdir(ruta_carpeta)
        if os.path.isfile(os.path.join(ruta_carpeta, elemento))
    )


def csv_shape(ruta_csv):
    """
    Devuelve el número de filas y columnas de un archivo CSV.
    """
    with open(ruta_csv, newline="", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        filas = list(lector)

    if not filas:
        return 0, 0

    return len(filas), len(filas[0])


def remove_unlabeled_images(ruta_csv, ruta_carpeta_imagenes, extensiones_validas=None):
    """
    Elimina de la carpeta de imágenes todos los archivos que no tengan una etiqueta
    definida en el CSV. El CSV debe contener una columna `id_code` con el nombre
    base de la imagen sin extensión.
    """
    if extensiones_validas is None:
        extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    etiquetas = set()
    with open(ruta_csv, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        filas = list(lector)

    if not filas:
        raise ValueError(f"El CSV está vacío o no se puede leer: {ruta_csv}")

    encabezado = filas[0]
    if 'id_code' in encabezado:
        idx_id = encabezado.index('id_code')
    else:
        idx_id = 0

    for fila in filas[1:]:
        if len(fila) > idx_id:
            id_code = fila[idx_id].strip()
            if id_code:
                etiquetas.add(id_code)

    if not etiquetas:
        raise ValueError(f"No se encontraron IDs válidos en el CSV: {ruta_csv}")

    if not os.path.exists(ruta_carpeta_imagenes):
        raise FileNotFoundError(f"La carpeta de imágenes no existe: {ruta_carpeta_imagenes}")

    extensiones_lower = tuple(ext.lower() for ext in extensiones_validas)
    eliminadas = 0
    conservadas = 0
    total = 0

    for root, _, files in os.walk(ruta_carpeta_imagenes):
        for file in files:
            if not file.lower().endswith(extensiones_lower):
                continue

            total += 1
            nombre_sin_ext = os.path.splitext(file)[0]
            if nombre_sin_ext not in etiquetas:
                os.remove(os.path.join(root, file))
                eliminadas += 1
            else:
                conservadas += 1

    print(f"Resumen limpieza: {total} imágenes revisadas en {ruta_carpeta_imagenes}")
    print(f"✔️ Conservadas: {conservadas}")
    print(f"🗑️ Eliminadas: {eliminadas}")
    return eliminadas, conservadas, total

import shutil
import hashlib

def calcular_md5(ruta_archivo):
    hash_md5 = hashlib.md5()
    with open(ruta_archivo, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def unify_images(origen, destino, eliminar_duplicados_contenido=True):
    """
    Copia imágenes desde 'origen' a 'destino', evitando duplicados:
    - Por nombre (ID)
    - Por contenido (opcional, usando hash MD5)
    """

    if not os.path.exists(destino):
        os.makedirs(destino)

    extensiones_validas = ('.png', '.jpg', '.jpeg')

    nombres_vistos = set()
    hashes_vistos = set()

    copiadas = 0
    duplicados_nombre = 0
    duplicados_contenido = 0

    for root, _, files in os.walk(origen):
        for file in files:
            if not file.lower().endswith(extensiones_validas):
                continue

            ruta_origen = os.path.join(root, file)

            # 🔴 1. Duplicado por nombre
            if file in nombres_vistos:
                duplicados_nombre += 1
                continue

            # 🔴 2. Duplicado por contenido (opcional)
            if eliminar_duplicados_contenido:
                hash_img = calcular_md5(ruta_origen)
                if hash_img in hashes_vistos:
                    duplicados_contenido += 1
                    continue
                hashes_vistos.add(hash_img)

            # Copiar
            ruta_destino = os.path.join(destino, file)
            shutil.copy2(ruta_origen, ruta_destino)

            nombres_vistos.add(file)
            copiadas += 1

    print("Resumen:")
    print(f"✔️ Imágenes copiadas: {copiadas}")
    print(f"⚠️ Duplicados por nombre eliminados: {duplicados_nombre}")
    print(f"⚠️ Duplicados por contenido eliminados: {duplicados_contenido}")


def unify_folders(carpeta1, carpeta2, carpeta_destino):
    """
    Unifica dos carpetas de imágenes en una sola carpeta destino.
    Evita duplicados por nombre de archivo.
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    nombres_vistos = set()
    copiadas = 0
    duplicados = 0

    for carpeta_origen in [carpeta1, carpeta2]:
        if not os.path.exists(carpeta_origen):
            print(f"⚠️ Advertencia: la carpeta {carpeta_origen} no existe, se omite")
            continue

        for file in os.listdir(carpeta_origen):
            ruta_origen = os.path.join(carpeta_origen, file)
            if not os.path.isfile(ruta_origen) or not file.lower().endswith(extensiones_validas):
                continue

            if file in nombres_vistos:
                duplicados += 1
                continue

            ruta_destino = os.path.join(carpeta_destino, file)
            shutil.copy2(ruta_origen, ruta_destino)
            nombres_vistos.add(file)
            copiadas += 1

    print(f"Unificación de carpetas completada:")
    print(f"✔️ Imágenes copiadas: {copiadas}")
    print(f"⚠️ Duplicados por nombre ignorados: {duplicados}")
    return copiadas, duplicados


def merge_csv_files(csv1, csv2, csv_destino):
    """
    Combina dos archivos CSV en uno solo.
    La primera línea de ambos se trata como encabezado y solo aparece una vez en el resultado.
    """
    todas_las_filas = []

    # Leer primer CSV
    with open(csv1, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        filas1 = list(lector)

    # Leer segundo CSV
    with open(csv2, newline='', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        filas2 = list(lector)

    if not filas1:
        raise ValueError(f"El primer CSV está vacío: {csv1}")
    if not filas2:
        raise ValueError(f"El segundo CSV está vacío: {csv2}")

    # Usar encabezado del primer CSV
    todas_las_filas.append(filas1[0])

    # Agregar datos del primer CSV (sin encabezado)
    todas_las_filas.extend(filas1[1:])

    # Agregar datos del segundo CSV (sin encabezado)
    todas_las_filas.extend(filas2[1:])

    # Escribir CSV combinado
    with open(csv_destino, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerows(todas_las_filas)

    total_filas = len(todas_las_filas) - 1  # -1 por el encabezado
    print(f"Combinación de CSVs completada:")
    print(f"✔️ Encabezado: {filas1[0]}")
    print(f"✔️ Filas del primer CSV: {len(filas1) - 1}")
    print(f"✔️ Filas del segundo CSV: {len(filas2) - 1}")
    print(f"✔️ Total de filas en archivo combinado: {total_filas}")
    return total_filas