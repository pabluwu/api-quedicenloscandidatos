from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# Importa tu función de chatbot
from chatbot_app import create_chatbot_chain, parse_candidate_responses 

# Cargar variables de entorno
load_dotenv()

API_KEY_NAME = "X-API-Key" # Nombre común para cabeceras de API Key
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Depends(api_key_header)):
    # Obtener la API Key esperada de las variables de entorno
    # Si no se encuentra, raise una excepción
    expected_api_key = os.getenv("API_KEY")
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY no configurada en el servidor."
        )

    # Compara la API Key proporcionada con la esperada
    if api_key == expected_api_key:
        return api_key # Si es válida, retorna la clave (o cualquier cosa, solo indica éxito)
    else:
        # Si no es válida, eleva una excepción HTTP 401 Unauthorized o 403 Forbidden
        # APIKeyHeader con auto_error=True ya maneja esto, pero puedes personalizar
        # el mensaje o el status code aquí si lo deseas.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # 401 para credenciales inválidas
            detail="API Key inválida o faltante.",
            headers={"WWW-Authenticate": "Bearer"},)

# Cargar el chatbot una vez al iniciar la aplicación
# Esto asegura que la base de datos ChromaDB se cargue solo una vez
# y el modelo Gemini se inicialice al inicio.
chatbot_rag_chain = None
try:
    chatbot_rag_chain = create_chatbot_chain()
    if chatbot_rag_chain is None:
        raise RuntimeError("Failed to initialize chatbot chain. Check logs from create_chatbot_chain.")
    print("Chatbot electoral cargado y listo para recibir peticiones.")
except Exception as e:
    print(f"Error fatal al iniciar el chatbot: {e}")
    # Aquí puedes decidir si quieres que la API no se inicie o que devuelva errores 500
    # por ahora, dejaremos que falle al iniciar si hay un problema crítico.
    exit(1) # Salir si el chatbot no pudo cargarse

app = FastAPI(
    title="Chatbot Electoral API",
    description="API para consultar programas electorales de candidatos usando Gemini y LangChain.",
    version="1.0.0"
)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Depends(api_key_header)):
    # Obtener la API Key esperada de las variables de entorno
    # Si no se encuentra, raise una excepción
    expected_api_key = os.getenv("API_KEY")
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY no configurada en el servidor."
        )

    # Compara la API Key proporcionada con la esperada
    if api_key == expected_api_key:
        return api_key # Si es válida, retorna la clave (o cualquier cosa, solo indica éxito)
    else:
        # Si no es válida, eleva una excepción HTTP 401 Unauthorized o 403 Forbidden
        # APIKeyHeader con auto_error=True ya maneja esto, pero puedes personalizar
        # el mensaje o el status code aquí si lo deseas.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # 401 para credenciales inválidas
            detail="API Key inválida o faltante.",
            headers={"WWW-Authenticate": "Bearer"},)

# Cargar el chatbot una vez al iniciar la aplicación
# Esto asegura que la base de datos ChromaDB se cargue solo una vez
# y el modelo Gemini se inicialice al inicio.
chatbot_rag_chain = None
try:
    chatbot_rag_chain = create_chatbot_chain()
    if chatbot_rag_chain is None:
        raise RuntimeError("Failed to initialize chatbot chain. Check logs from create_chatbot_chain.")
    print("Chatbot electoral cargado y listo para recibir peticiones.")
except Exception as e:
    print(f"Error fatal al iniciar el chatbot: {e}")
    # Aquí puedes decidir si quieres que la API no se inicie o que devuelva errores 500
    # por ahora, dejaremos que falle al iniciar si hay un problema crítico.
    exit(1) # Salir si el chatbot no pudo cargarse

# Modelo de Pydantic para la entrada de la API
class QueryRequest(BaseModel):
    question: str

# Modelo de Pydantic para la entrada de la API
class QueryRequest(BaseModel):
    question: str

# Nuevo modelo de Pydantic para la respuesta de cada candidato
class CandidateResponse(BaseModel):
    candidate: str
    response: str

# Nuevo modelo de Pydantic para la respuesta completa de la API
class QueryResponse(BaseModel):
    question: str
    answers: list[CandidateResponse]
    source: str 

# Endpoint para la consulta del chatbot
@app.post("/query", response_model=QueryResponse) # Especifica el modelo de respuesta
async def query_chatbot(request: QueryRequest, api_key: str = Depends(get_api_key)):
    """
    Consulta el chatbot sobre los programas electorales de los candidatos.
    Devuelve las respuestas de cada candidato en un array de objetos.
    """
    # 1. Intentar obtener de la caché
    # cached_response_str = get_cached_response(request.question) # Esto sigue siendo una cadena
    # if cached_response_str:
    #     print(f"Respuesta encontrada en caché para: {request.question}")
    #     # Parsear la cadena cacheadas en el formato de array de objetos
    #     parsed_answers = parse_candidate_responses(cached_response_str)
    #     return QueryResponse(question=request.question, answers=parsed_answers, source="cache")


    # 2. Si no está en caché, invocar al chatbot
    if chatbot_rag_chain is None:
        raise HTTPException(status_code=500, detail="Chatbot no inicializado.")

    try:
        llm_raw_response = await chatbot_rag_chain.ainvoke({"question": request.question})
        
        # 3. Parsear la respuesta cruda del LLM
        parsed_answers = parse_candidate_responses(llm_raw_response)

        # 4. Guardar la respuesta ORIGINAL (la cadena cruda del LLM) en caché
        # Esto es importante para que la lógica de parsing se aplique siempre igual al recuperar.
        # save_cached_response(request.question, llm_raw_response)
        
        print(f"Respuesta generada por LLM y guardada en caché para: {request.question}")
        return QueryResponse(question=request.question, answers=parsed_answers, source="llm")
    except Exception as e:
        print(f"Error al procesar la petición: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

# Endpoint de salud (para verificar si la API está viva)
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}

# Para ejecutar la API (solo si ejecutas este archivo directamente)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # En producción, usa un servidor ASGI como Gunicorn + Uvicorn workers
    # uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)q