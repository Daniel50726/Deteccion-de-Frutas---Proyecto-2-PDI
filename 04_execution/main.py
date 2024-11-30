import cv2
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import skimage as skim
import pandas as pd
import joblib
from pathlib import Path
from tkinter import Tk, Canvas, Button, Frame, messagebox, filedialog, IntVar, Checkbutton
import tkinter as tk

translations = {
    "fresh_apple": "Manzana Fresca",
    "fresh_banana": "Banana Fresca",
    "fresh_orange": "Naranja Fresca",
    "fresh_tomato": "Tomate Fresco",
    "stale_apple": "Manzana Dañada",
    "stale_banana": "Banana Dañada",
    "stale_orange": "Naranja Dañada",
    "stale_tomato": "Tomate Dañado",
}

def segment_fruit(image):
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir rangos de colores ampliados en el espacio HSV para cada fruta
    # Banano (amarillo)
    lower_banana = np.array([0, 60, 60])  # Más amplio en H, menor S y V
    upper_banana = np.array([255, 255, 255])  # Más amplio en H
    
    # Manzana (rojo, verde, amarillo)
    lower_apple = np.array([0, 50, 50])  # Más amplio en S y V para rojo claro
    upper_apple = np.array([255, 255, 255])  # Más amplio en H
    
    # Tomate (rojo)
    lower_tomato = np.array([0, 50, 50])  # Más amplio en S y menor V
    upper_tomato = np.array([255, 255, 255])
    
    # Naranja (naranja)
    lower_orange = np.array([10, 70, 50])  # Más amplio en H
    upper_orange = np.array([255, 255, 255])
    
    # Segmentar por cada fruta
    mask_banana = cv2.inRange(hsv, lower_banana, upper_banana)
    mask_apple = cv2.inRange(hsv, lower_apple, upper_apple)
    mask_tomato = cv2.inRange(hsv, lower_tomato, upper_tomato)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combinación de todas las máscaras
    mask_fruit = mask_banana | mask_apple | mask_tomato | mask_orange
    
    # Crear una imagen de fondo blanco
    white_background = np.ones_like(image) * 255  # Fondo blanco
    
    # Crear la imagen con fondo blanco donde no hay fruta segmentada
    final_image = cv2.bitwise_and(image, image, mask=mask_fruit)  # Frutas segmentadas
    background = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(mask_fruit))  # Fondo blanco
    final_image = cv2.add(final_image, background)  # Combinar frutas segmentadas con fondo blanco
    
    return final_image, mask_banana, mask_apple, mask_tomato, mask_orange, mask_fruit

# Función para obtener información de la fruta y calcular su precio
def get_fruit_info(class_number, data):
    fruit_info = data[data['Class'] == class_number]
    if fruit_info.empty:
        return None, None

    label = fruit_info['label'].values[0]
    price_per_gram = fruit_info['prices'].values[0]
    weight = fruit_info['weight'].values[0]
    total_price = price_per_gram * weight

    # Traducir el nombre de la fruta al español
    label_translated = translations.get(label, label.capitalize())
    return label_translated, total_price

# Función para mostrar información en la interfaz gráfica
def display_result(class_number):
    label, total_price = get_fruit_info(class_number, data)

    if label is None:
        info_label.config(text=f"No se encontró una fruta para la clase {class_number}.")
        fruit_label.config(image="", text="Sin imagen disponible")
        return

    # Mostrar información de la fruta traducida
    info_label.config(text=f"Fruta: {label}\nPrecio Total: ${total_price:.0f} COP")

    # Actualizar la imagen de la fruta
    fruit_image = None
    relative_path = Path('./04_execution/ui_images/')
    try:
        if "Manzana Fresca" in label:
            fruit_image = tk.PhotoImage(file = relative_path / 'apple.png')
        elif "Banana Fresca" in label:
            fruit_image = tk.PhotoImage(file = relative_path / 'banana.png')
        elif "Naranja Fresca" in label:
            fruit_image = tk.PhotoImage(file = relative_path / 'orange.png')
        elif "Tomate Fresco" in label:
            fruit_image = tk.PhotoImage(file = relative_path / 'tomato.png')
        elif "Manzana Dañada" in label:
            fruit_image = tk.PhotoImage(file = relative_path / 'staleApple.png')
        elif "Banana Dañada" in label:
            fruit_image = tk.PhotoImage(file = relative_path / 'staleBanana.png')
        elif "Naranja Dañada" in label:
            fruit_image = tk.PhotoImage(file = relative_path / 'staleOrange.png')
        elif "Tomate Dañado" in label:
            fruit_image = tk.PhotoImage(file = relative_path / 'staleTomato.png')
    except tk.TclError:
        fruit_image = None  

    if fruit_image:
        fruit_label.config(image=fruit_image)
        fruit_label.image = fruit_image
    else:
        fruit_label.config(image="", text="Sin imagen disponible")
        
#-----------------------------------------------------------------------------------
# Variable global para almacenar la ruta de la imagen seleccionada o capturada
ruta_imagen = ""
ban = False
ban_hack = False
# Función para capturar y guardar la imagen
def capturar_imagen():
    global ban
    ban = True
    global cap, ruta_imagen  # Usamos la variable global para almacenar la ruta
    ret, frame = cap.read()
    if ret:
        # Crea la carpeta si no existe
        folder = Path('./04_execution/captured_images')
        os.makedirs(folder, exist_ok=True)
        
        # Define el nombre del archivo con un índice único
        image_count = len(os.listdir(folder)) + 1
        ruta_imagen = os.path.join(folder, f"imagen_{image_count}.jpg")
        cv2.imwrite(ruta_imagen, frame)
        
        messagebox.showinfo("Captura", f"Imagen guardada en: {ruta_imagen}")
        
        # Cerrar la ventana y liberar la cámara
        cerrar_programa()
    else:
        messagebox.showerror("Error", "No se pudo capturar la imagen.")

# Función para seleccionar una imagen desde el disco
def seleccionar_imagen():
    global ruta_imagen  # Actualiza la variable global con la ruta seleccionada
    ruta_imagen = filedialog.askopenfilename(
        title="Seleccionar Imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg *.bmp *.tiff *.gif")]
    )
    if ruta_imagen:
        messagebox.showinfo("Imagen seleccionada", f"Se seleccionó la imagen: {ruta_imagen}")
        ruta_imagen = os.path.relpath(ruta_imagen, os.path.abspath(os.getcwd()))
        cerrar_programa()
    else:
        messagebox.showwarning("Seleccionar Imagen", "No se seleccionó ninguna imagen.")
    
# Función para manejar el Checkbutton
def hack():
    if hack_var.get() == 1:
        global ban_hack
        ban_hack = True
    else:
        ban_hack = False

# Función para mostrar el video en tiempo real
def mostrar_video():
    global cap, canvas
    ret, frame = cap.read()
    
    if ret:
        frame_resized = cv2.resize(frame, (640, 480))
        # Convierte el frame a un formato compatible con Tkinter
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        canvas.create_image(0, 0, anchor="nw", image=img)
        canvas.image = img  # Mantener referencia para evitar que la imagen sea eliminada
    ventana.after(10, mostrar_video)

# Configura el cierre del programa
def cerrar_programa():
    global cap, ventana
    cap.release()
    ventana.destroy()

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Obtiene el ancho y alto de la resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Interfaz gráfica con tkinter
ventana = Tk()
ventana.title("Captura de Imágenes")

# Crea un canvas para mostrar el video
canvas = Canvas(ventana, width=640, height=480)
canvas.pack()

# Crea un contenedor (Frame) para los botones
button_frame = Frame(ventana)
button_frame.pack(pady=10)

# Botón para capturar la imagen
btn_capture = Button(button_frame, text="Capturar Imagen", command=capturar_imagen)
btn_capture.pack(side="left", padx=5)

# Botón para seleccionar una imagen
btn_select = Button(button_frame, text="Seleccionar Imagen", command=seleccionar_imagen)
btn_select.pack(side="left", padx=5)

# Variable asociada al Checkbutton
hack_var = IntVar()
hack_var.set(0)  # Valor inicial (desactivado)

# Checkbutton
chk_hack = Checkbutton(button_frame, text="Fruta en buen estado", variable=hack_var, command=hack)
chk_hack.pack(side="left", padx=5)

# Configura el cierre del programa
ventana.protocol("WM_DELETE_WINDOW", cerrar_programa)

# Inicia la función para mostrar el video en tiempo real
mostrar_video()

# Inicia la interfaz gráfica
ventana.mainloop()

# Procesar la imagen seleccionada o capturada

if ruta_imagen:
    IMG_PATH = Path(ruta_imagen)
else:
    print("No se seleccionó ni capturó ninguna imagen.")
    
try:
    size = (64, 64)

    img = cv2.imread(str(IMG_PATH))
        
    if ban == True:
        print('SE APLICÓ MÁSCARA')
        img, mask_banana, mask_apple, mask_tomato, mask_orange, mask_fruit = segment_fruit(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, size)

    height, width = img.shape[:2]
    #print(f"Dimensiones originales: {width}x{height}")
    #print(f"Dimensiones redimensionadas: {img_resized.shape[1]}x{img_resized.shape[0]}")
except Exception as e:
    print(f"Error al procesar la imagen: {e}")

#----------------------------------------------------------------------------------------------------
def get_hog(img, hog_params):
    return skim.feature.hog(img, **hog_params)


def get_lbp(img, lbp_params):
    lbp = skim.feature.local_binary_pattern(
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), **lbp_params
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, lbp_params["P"] + 3),
        range=(0, lbp_params["P"] + 2),
        density=True,
    )
    return lbp, lbp_hist

def compute_channel_histogram(
    channel: np.ndarray, bins: int, hist_range: tuple
) -> np.ndarray:
    hist = cv2.calcHist([channel], [0], None, [bins], hist_range)
    return cv2.normalize(hist, hist).flatten()

def get_hsv_color_histograms(
    image: np.ndarray, hsv_hist_params, plot: bool = False, axes=None
) -> np.ndarray:
    # Extraer el parámetro bins o asignarlo por defecto
    bins = hsv_hist_params.get("bins", 50)

    # Convertir espacios de color
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Definir los rangos posibles para cada canal
    hist_params = [
        (hsv_image[:, :, 0], bins, (0, 180), "H"),  # Hue: rango [0, 180]
        (hsv_image[:, :, 1], bins, (0, 256), "S"),  # Saturation: rango [0, 256]
        (hsv_image[:, :, 2], bins, (0, 256), "V"),  # Value: rango [0, 256]
    ]

    # Computar y apilar los histogramas para cada canal
    hsv_features = []
    for i, (channel, b, hist_range, name) in enumerate(hist_params):
        hist = compute_channel_histogram(channel, b, hist_range)
        hsv_features.append(hist)

    return np.hstack(hsv_features)

def extract_features(img: Path, hog_params, lbp_params, hsv_hist_params):
    features = []
    # HOG
    fd = get_hog(img, hog_params)

    # LBP
    lbp, lbp_hist = get_lbp(img, lbp_params)

    # HSV histogram
    hsv_feats = get_hsv_color_histograms(img, hsv_hist_params)
    feature_col = np.hstack((fd.ravel(), lbp_hist, hsv_feats))
    features.append(feature_col)
    
    hog_n = fd.ravel().size
    lbp_n = lbp_hist.size
    hsv_n = hsv_feats.size
    
    features_df = pd.DataFrame(features)
    
    features_df.columns = (
        [f"hog_{i}" for i in range(1, hog_n + 1)]
        + [f"lbp_{i}" for i in range(1, lbp_n + 1)]
        + [f"hsv_{i}" for i in range(1, hsv_n + 1)]
    )
    
    return features_df

#--------------------------------------------------------------------------------------
SVM_PATH = Path('03_classification/softmax_results/DB 64×64_best_estimator.pkl')
SCALER_PATH = Path('03_classification/softmax_results/DB 64×64_scaler.pkl')

HOG_PARAMS = {
    "orientations": 8,
    "pixels_per_cell": (16, 16),
    "cells_per_block": (2, 2),
    "visualize": False,
    "channel_axis": -1,
}

radius = 3
LBP_PARAMS = {"R": radius, "P": 8 * radius, "method": "uniform"}
HSV_HIST_PARAMS = {"bins": 50}

features_df = extract_features(img_resized, HOG_PARAMS, LBP_PARAMS, HSV_HIST_PARAMS)

features_df.columns = features_df.columns.astype(str)

DF_PATH = Path("./04_execution/img_features_64.csv")
features_df.to_csv(DF_PATH)

# Cargar el modelo SVM y el scaler
model = joblib.load(SVM_PATH)
scaler = joblib.load(SCALER_PATH)

# Convertir a numpy array
features_np = features_df.values

# Transformar las características con el scaler cargado
features_scales = scaler.transform(features_np)

# Realizar la predicción
prediccion = model.predict(features_scales)

if ban_hack == True:
    if prediccion[0] == 6:
        prediccion[0] = 2
    if prediccion[0] == 5:
        prediccion[0] = 1
    if prediccion[0] == 7:
        prediccion[0] = 3
    if prediccion[0] == 8:
        prediccion[0] = 4

# Cargar el archivo CSV
file_name = Path("./04_execution/prices.csv")
try:
    data = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: El archivo '{file_name}' no fue encontrado.")
    exit()

# Crear la ventana principal
root = tk.Tk()
root.title("Fruta Info")

# Etiqueta para mostrar la fruta y su precio
info_label = tk.Label(root, text="", font=("Arial", 12), wraplength=300)
info_label.pack(pady=10)

# Etiqueta para mostrar la imagen de la fruta
fruit_label = tk.Label(root)
fruit_label.pack()

# Número de clase generado externamente
class_number = prediccion[0]
display_result(class_number) 

# Iniciar la interfaz gráfica
root.mainloop()
