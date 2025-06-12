import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# CAMBIO IMPORTANTE: Usamos GoogleGenerativeAIEmbeddings en lugar de OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Cargar variables de entorno desde .env
load_dotenv()

# Asegúrate de que la API key de Google esté configurada
# Obtén tu clave de Google AI Studio: https://aistudio.google.com/
# Y agrégala a tu archivo .env como GOOGLE_API_KEY="tu_clave_aqui"
# LangChain la buscará automáticamente en las variables de entorno o en .env
# Puedes descomentar la siguiente línea si necesitas forzar la carga, pero load_dotenv debería bastar.
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def ingest_documents(pdf_paths, collection_name_prefix="elecciones", persist_directory="./chroma_db"):
    """
    Procesa una lista de rutas de PDFs, extrae texto, crea embeddings
    y los guarda en ChromaDB, categorizados por candidato.
    """
    # 3. Crear Embeddings
    # CAMBIO IMPORTANTE: Inicializamos GoogleGenerativeAIEmbeddings
    # El modelo 'models/embedding-001' es el modelo de embeddings recomendado por Google.
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Pequeña prueba para ver si la API key es válida
        _ = embeddings.embed_query("test") 
        print("Google Gemini Embeddings inicializado correctamente.")
    except Exception as e:
        print(f"Error al inicializar Google Gemini Embeddings: {e}")
        print("Asegúrate de que tu GOOGLE_API_KEY esté correctamente configurada en el archivo .env")
        return

    # Limpiar la base de datos ChromaDB existente si quieres re-ingestar desde cero
    # Esto es útil si modificas los PDFs o el chunking
    if os.path.exists(persist_directory):
        import shutil
        print(f"Limpiando directorio persistente de ChromaDB: {persist_directory}")
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)


    for candidate_name, pdf_path in pdf_paths.items():
        print(f"Procesando PDF para el candidato: {candidate_name} ({pdf_path})...")

        # 1. Cargar el documento PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 2. Dividir el documento en "chunks"
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Añadir metadatos a cada chunk para identificar al candidato
        for i, text in enumerate(texts):
            text.metadata["source"] = pdf_path
            text.metadata["candidate"] = candidate_name
            text.metadata["chunk_id"] = i

        # 4. Almacenar en ChromaDB
        print(f"Creando embeddings y almacenando en ChromaDB para {candidate_name}...")
        
        # Para Google Gemini con Chroma, es mejor usar la función .from_documents
        # que se encargará de añadir y persistir correctamente.
        # Si ya existe la base de datos, la carga y añade nuevos documentos.
        # Es crucial que la colección se llame igual cada vez si quieres agregar o persistir sobre la misma.
        db = Chroma.from_documents(
            texts, 
            embeddings, 
            collection_name=f"{collection_name_prefix}_candidates",
            persist_directory=persist_directory
        )
        
        print(f"Documentos del candidato {candidate_name} añadidos a ChromaDB.")
        # db.persist() # No es necesario llamar persist() explícitamente después de from_documents si la persistencia está configurada
        print(f"Base de datos ChromaDB persistida en '{persist_directory}'.")


if __name__ == "__main__":
    # Asegúrate de tener tus PDFs en una carpeta 'data' y especifica las rutas correctas
    pdf_files = {
        "Jaime_Mulet": "./data/mulet.pdf",
        "Carolina_Toha": "./data/toha.pdf",
        "Jeanette_Jara": "./data/jara.pdf",
        "Gonzalo_Winter": "./data/winter.pdf",
    }

    # Crea las carpetas necesarias si no existen
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)

    # Verificar que los PDFs existen antes de intentar procesarlos
    all_pdfs_exist = True
    for candidate, path in pdf_files.items():
        if not os.path.exists(path):
            print(f"Error: El archivo PDF para {candidate} no se encontró en {path}")
            all_pdfs_exist = False

    if all_pdfs_exist:
        ingest_documents(pdf_files)
        print("Proceso de ingesta de documentos completado.")
    else:
        print("Por favor, asegúrate de que todos los PDFs estén en las rutas especificadas y vuelve a intentarlo.")