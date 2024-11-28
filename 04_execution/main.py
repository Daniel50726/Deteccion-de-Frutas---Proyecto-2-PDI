'''
Proyecto 1 de PDI
Autores:
    Jorge Sebastian Arroyo Estrada     sebastian.arroyo1@udea.edu.co
    CC:1193482707
    Daniel Felipe Yépez Taimal         daniel.yepez@udea.edu.co
    CC:1004193180

PREPROCESADO DE BASE DE DATOS
1. Quitar imágenes con fondos que no sean blancos
2. Ampliar base de datos (translaciones y rotaciones)
3. Ampliar base de datos (imágenes con ruido de sal y pimienta)
4. Quitar imágenes que sean muy pequeñas o muy grandes ()
'''

import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

# Función para capturar y guardar la imagen
def capturar_imagen():
    global cap
    ret, frame = cap.read()
    if ret:
        # Crea la carpeta si no existe
        folder = "captured_images"
        os.makedirs(folder, exist_ok=True)
        
        # Define el nombre del archivo con un índice único
        image_count = len(os.listdir(folder)) + 1
        image_path = os.path.join(folder, f"imagen_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        
        messagebox.showinfo("Captura", f"Imagen guardada en: {image_path}")
    else:
        messagebox.showerror("Error", "No se pudo capturar la imagen.")

# Función para mostrar el video en tiempo real
def mostrar_video():
    global cap, canvas
    ret, frame = cap.read()
    if ret:
        # Convierte el frame a un formato compatible con Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img  # Mantener referencia para evitar que la imagen sea eliminada
    ventana.after(10, mostrar_video)

# Configuración de la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Interfaz gráfica con tkinter
ventana = tk.Tk()
ventana.title("Captura de Imágenes")

# Crea un canvas para mostrar el video
canvas = tk.Canvas(ventana, width=640, height=480)
canvas.pack()

# Botón para capturar la imagen
btn_captura = tk.Button(ventana, text="Capturar Imagen", command=capturar_imagen)
btn_captura.pack(pady=10)

# Configura el cierre del programa
def cerrar_programa():
    cap.release()
    ventana.destroy()

ventana.protocol("WM_DELETE_WINDOW", cerrar_programa)

# Inicia la función para mostrar el video en tiempo real
mostrar_video()

# Inicia la interfaz gráfica
ventana.mainloop()

#-----------------------------------------------------

def resize_image(img, size):
    return img.resize(size)

def preprocess(img, avg_res, std_res):
    # Escala de grises
    img_gray = img.convert("L")
    # Filtro Gaussiano
    img_filtered = img_gray.filter(ImageFilter.GaussianBlur(radius=2))
    # Thresholding
    _, img_thresholded = cv2.threshold(
        np.array(img_filtered), 128, 255, cv2.THRESH_BINARY
    )

    _, img_otsu = cv2.threshold(
        np.array(img_filtered), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return (img_gray, img_filtered, img_thresholded, img_otsu)

def show_preprocessed_images(img, img_gray, img_filtered, img_thresholded):
    plt.figure(figsize=(12, 12), layout="tight")
    images = [img, img_gray, img_filtered, img_thresholded]
    images = [img, img_gray, img_filtered, img_thresholded]
    titles = ["Original", "Escala de grises", "Filtrada", "Umbralizada"]

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 2, i + 1)
        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap="gray")
        plt.title(title, fontsize=18)
        plt.axis("off")

DB_PATH = Path("../captured_images/prueba1.jpg")
size = (64, 64)

# Lee la imagen
img = cv2.imread(str(DB_PATH))  # Asegúrate de convertir Path a string
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()
    
# Cambia el tamaño de la imagen
img_resized = cv2.resize(img, size)

# Obtén las dimensiones originales
height, width = img.shape[:2]
print(f"Dimensiones originales: {width}x{height}")
print(f"Dimensiones redimensionadas: {img_resized.shape[1]}x{img_resized.shape[0]}")





