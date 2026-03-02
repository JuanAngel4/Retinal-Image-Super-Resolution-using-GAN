import kagglehub

# Carpeta donde quieres guardarlo
destination = "E:\\Proyects Python based\\ProyectoAvanzado2\\DATA_KAGGLE"

path = kagglehub.dataset_download(
    "anikbhowmickae20b102/binary-classification-data-aptos-and-messidor",
    path=destination
)

print("Dataset descargado en:", path)