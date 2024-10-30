"""
Proyecto 1 de PDI
Presentado por los estudiantes:
             Jorge Sebastian Arroyo Estrada     sebastian.arroyo1@udea.edu.co
             CC:1193482707
             Daniel Felipe Yépez Taimal         daniel.yepez@udea.edu.co
             CC:1004193180
Proposito: Preprocesado
"""

'''
PREPROCESADO DE BASE DE DATOS
1. Quitar imágenes con fondos que no sean blancos
2. Ampliar base de datos (translaciones y rotaciones)
3. Ampliar base de datos (imágenes con ruido de sal y pimienta)
4. Quitar imágenes que sean muy pequeñas o muy grandes ()
'''

# Librerías
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

# Funciones generales

def read_image(image_path: str) -> np.ndarray:
    """Lee una imagen desde la ruta dada y devuelve un array de numpy."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"La imagen con ruta {image_path} no se pudo leer.")
    return image


def get_image_resolution(image: np.ndarray):
    """Obtiene la resolución de una imagen dada como array de numpy."""
    height, width = image.shape[:2]
    return width, height


def collect_resolutions(folder_path: str):
    """Recopila resoluciones y rutas de las imágenes en la carpeta dada."""
    resolutions = []
    image_paths = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            try:
                image = read_image(file_path)
                resolution = get_image_resolution(image)
                resolutions.append(resolution)
                image_paths.append(file_path)
            except ValueError as e:
                print(e)
                continue

    return resolutions, image_paths


def calculate_min_max_resolutions(
    resolutions: list[tuple[int, int]], image_paths: list[str]
):
    """Calcula la resolución mínima y máxima a partir de la lista de resoluciones y sus rutas."""
    resolutions_array = np.array(resolutions)
    widths, heights = resolutions_array[:, 0], resolutions_array[:, 1]

    min_idx = np.argmin(widths * heights)
    max_idx = np.argmax(widths * heights)

    min_resolution = resolutions[min_idx]
    max_resolution = resolutions[max_idx]
    min_image_path = image_paths[min_idx]
    max_image_path = image_paths[max_idx]

    return min_resolution, min_image_path, max_resolution, max_image_path


def calculate_average_resolution(resolutions: list[tuple[int, int]]):
    """Calcula la resolución promedio a partir de la lista de resoluciones."""
    resolutions_array = np.array(resolutions)
    avg_width = np.mean(resolutions_array[:, 0])
    avg_height = np.mean(resolutions_array[:, 1])
    return int(avg_width), int(avg_height)


def calculate_std_resolution(resolutions: list[tuple[int, int]]):
    """Calcula la desviación estándar de las resoluciones."""
    resolutions_array = np.array(resolutions)
    std_width = np.std(resolutions_array[:, 0])
    std_height = np.std(resolutions_array[:, 1])
    return std_width, std_height


def calculate_resolutions(folder_path: str):
    """Calcula las resoluciones mínima, máxima y promedio de las imágenes en una carpeta."""
    resolutions, image_paths = collect_resolutions(folder_path)
    weight = len(resolutions)
    min_resolution, min_image_path, max_resolution, max_image_path = (
        calculate_min_max_resolutions(resolutions, image_paths)
    )
    avg_resolution = calculate_average_resolution(resolutions)
    std_resolution = calculate_std_resolution(resolutions)
    return (
        min_resolution,
        min_image_path,
        max_resolution,
        max_image_path,
        avg_resolution,
        std_resolution,
        weight,
    )

# Código

DB_PATH = "\Frutas_db"

# Arreglos para almacenar los valores deseados de cada etiqueta
min_resolutions = []
max_resolutions = []
min_image_paths = []
max_image_paths = []
avg_resolutions = []
std_resolutions = []
weights_per_tag = []

for fruit_tag in os.listdir(DB_PATH):
    folder_path = os.path.join(DB_PATH, fruit_tag)
    if os.path.isdir(folder_path):
        print(f"Procesando carpeta: {folder_path}")
        (
            min_resolution,
            min_image_path,
            max_resolution,
            max_image_path,
            avg_resolution,
            std_resolution,
            weight,
        ) = calculate_resolutions(folder_path)

        # Muestra las resoluciones mínima y máxima junto con sus rutas
        print(f"Resolución mínima: {min_resolution[0]}×{min_resolution[1]}")
        print(f"Ruta de la imagen con la mínima resolución: {min_image_path}")
        print(f"Resolución máxima: {max_resolution[0]}×{max_resolution[1]}")
        print(f"Ruta de la imagen con la máxima resolución: {max_image_path}")
        print(f"Resolución promedio: {avg_resolution[0]}×{avg_resolution[1]}")
        print(
            f"Desviación estándar de la resolución: {std_resolution[0]:.0f}×{std_resolution[1]:.0f}\n"
        )

        # Almacena las resoluciones y las rutas hacia los mínimos y máximos
        min_resolutions.append(min_resolution)
        max_resolutions.append(max_resolution)
        min_image_paths.append(min_image_path)
        max_image_paths.append(max_image_path)
        avg_resolutions.append(avg_resolution)
        std_resolutions.append(std_resolution)
        weights_per_tag.append(weight)