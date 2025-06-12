import os
import re
from dotenv import load_dotenv
# CAMBIO IMPORTANTE: Usamos ChatGoogleGenerativeAI y GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Cargar variables de entorno
load_dotenv()

# Asegúrate de que la API key de Google esté configurada en el archivo .env
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def create_chatbot_chain(persist_directory="./chroma_db", collection_name="elecciones_candidates"):
    """
    Crea la cadena LangChain para el chatbot, incluyendo el modelo de embeddings,
    la base de datos vectorial y el LLM.
    """
    # 1. Cargar el modelo de embeddings (DEBE ser el mismo que usaste para la ingesta)
    # CAMBIO IMPORTANTE: Usamos GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 2. Cargar la base de datos vectorial persistente
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        print(f"Base de datos ChromaDB cargada desde '{persist_directory}'.")
    except Exception as e:
        print(f"Error al cargar la base de datos ChromaDB: {e}")
        print("Asegúrate de haber ejecutado 'ingest_data.py' con los embeddings de Gemini primero.")
        return None

    # 3. Configurar el modelo de lenguaje (LLM)
    # CAMBIO IMPORTANTE: Usamos ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

    # 4. Definir el prompt para el LLM
    template = """
    Eres un asistente imparcial que analiza programas electorales de diferentes candidatos.
    Tu tarea es responder a la pregunta del usuario basándote **SOLO** en la información proporcionada por los documentos de cada candidato.
    Si no encuentras información sobre un candidato para la pregunta, indícalo claramente como "No se encontró información relevante para [Candidato X] sobre este tema."

    Responde de forma concisa y estructurada para cada candidato por separado.
    Utiliza el siguiente formato para cada candidato:
    Candidato [Nombre del Candidato]: [Respuesta basada en su programa]

    ---
    **Pregunta del Usuario:** {question}

    **Contexto de los Candidatos:**
    {context}
    ---
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Definimos los nombres de los candidatos que tenemos
    # Asegúrate de que estos nombres coincidan con los que usaste en ingest_data.py
    candidate_names = ["Jaime_Mulet", "Carolina_Toha", "Jeanette_Jara", "Gonzalo_Winter"]

    def get_candidate_context(question: str, candidate_name: str, vectorstore: Chroma):
        """
        Busca los documentos más relevantes para una pregunta y un candidato específico,
        y devuelve el contenido combinado.
        """
        # Filtrar por el metadato 'candidate'
        retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"candidate": candidate_name}, "k": 5} # k=5: top 5 chunks
        )
        docs = retriever.invoke(question)
        # Combina los contenidos de los documentos en un solo string
        # Si no se encontraron documentos, devuelve un mensaje claro
        if not docs:
            return f"No se encontraron datos de programa para {candidate_name}."
        return "\n\n".join([doc.page_content for doc in docs])

    # Esta parte de la cadena recuperará el contexto para cada candidato
    # y lo combinará en un solo string 'context' para el prompt.
    retrieval_chain = RunnablePassthrough.assign(
        context=lambda x: "\n\n".join([
            f"**Información de {cand}:**\n{get_candidate_context(x['question'], cand, vectorstore)}"
            for cand in candidate_names
        ])
    )

    # La cadena final
    rag_chain = (
        retrieval_chain
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def parse_candidate_responses(llm_response: str) -> list[dict]:
    """
    Parsea la respuesta del LLM en una lista de diccionarios,
    donde cada diccionario representa la respuesta de un candidato.
    """
    parsed_responses = []
    # Usar una expresión regular para encontrar patrones como "Candidato [Nombre]: [Respuesta]"
    # Esta regex busca "Candidato " seguido de cualquier cosa que no sea nueva línea hasta ":"
    # y luego captura el resto hasta el siguiente "Candidato " o el final de la cadena.
    # El patrón r"(?s)" hace que '.' incluya saltos de línea
    pattern = re.compile(r"Candidato\s+([^:]+):\s*(.*?)(?=\nCandidato\s+[^:]+:|\Z)", re.DOTALL)
    
    matches = pattern.findall(llm_response)

    for name, response_text in matches:
        # Limpiar el nombre del candidato (eliminar espacios extra)
        candidate_name = name.strip()
        # Limpiar la respuesta (eliminar espacios al inicio/final)
        candidate_response = response_text.strip()
        
        parsed_responses.append({
            "candidate": candidate_name,
            "response": candidate_response
        })
            
    # Manejo de casos donde no se encuentra el formato esperado (ej. no se encontró información)
    if not matches and "No se encontró información relevante" in llm_response:
        # Si el LLM indicó que no hay info, podemos pasarlo como un mensaje global
        parsed_responses.append({
            "candidate": "General",
            "response": llm_response.strip() # O un mensaje más específico
        })
    elif not matches and parsed_responses == []:
        # Si no hubo matches y tampoco el mensaje de "no info",
        # simplemente devolvemos la respuesta completa como un objeto genérico
        # Esto es un fallback
        parsed_responses.append({
            "candidate": "General",
            "response": llm_response.strip()
        })


    return parsed_responses

if __name__ == "__main__":
    chatbot = create_chatbot_chain()

    if chatbot:
        print("Chatbot listo. Escribe 'salir' para terminar.")
        while True:
            user_question = input("\nTu pregunta: ")
            if user_question.lower() == 'salir':
                break

            # Invocar la cadena con la pregunta del usuario
            response = chatbot.invoke({"question": user_question})
            print("\n--- Respuesta de la IA ---")
            print(response)
            print("-------------------------")