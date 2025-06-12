import os
from dotenv import load_dotenv
import google.generativeai as genai

# Cargar variables de entorno
load_dotenv()

# Configurar la API key de Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY no está configurada en el archivo .env")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

try:
    print("Listando modelos disponibles de Google Gemini...\n")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"Nombre del Modelo: {m.name}")
            print(f"Descripción: {m.description}")
            print(f"Métodos Soportados: {m.supported_generation_methods}\n")
except Exception as e:
    print(f"Ocurrió un error al listar los modelos: {e}")
    print("Por favor, verifica tu GOOGLE_API_KEY y tu conexión a internet.")