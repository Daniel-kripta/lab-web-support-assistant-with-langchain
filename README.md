![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Asistente de Soporte con LangChain Completo

## Objetivo

Construir un asistente de soporte al cliente que combine todo lo aprendido hoy:
- **RAG** para consultar una base de conocimiento de la empresa (FAQs, políticas)
- **Memoria** para mantener el contexto de la conversación
- **Tools** para acciones concretas (buscar pedido, calcular reembolso)
- **LangGraph** para controlar el flujo con lógica condicional
- **FastAPI** como interfaz HTTP

## Setup

```bash
# fork & clone the repository
cd lab-web-support-assistant-with-langchain
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install langchain langchain-openai langgraph chromadb python-dotenv
pip install langchain-community langchain-text-splitters fastapi uvicorn
pip freeze > requirements.txt
```

## Paso 1 — Base de conocimiento (RAG)

Crea un archivo `politicas.txt` con al menos 10 políticas de la empresa (devoluciones, envíos, garantía, etc.). Ingestalo en ChromaDB:

```python
# ingestar.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader("politicas.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

print(f"Indexados {len(chunks)} chunks")
```

## Paso 2 — Tools del asistente

```python
from langchain_core.tools import tool

@tool
def buscar_pedido(pedido_id: str) -> str:
    """Busca el estado de un pedido por su ID. Ejemplo: buscar_pedido('PED-1234')"""
    pedidos = {
        "PED-1234": {"estado": "enviado", "fecha_entrega": "15/05/2026", "total": 89.99},
        "PED-5678": {"estado": "en preparación", "fecha_entrega": "18/05/2026", "total": 45.50},
    }
    pedido = pedidos.get(pedido_id.upper())
    return str(pedido) if pedido else f"Pedido {pedido_id} no encontrado"

@tool
def calcular_reembolso(total: float, porcentaje: float) -> str:
    """Calcula el importe de un reembolso parcial."""
    reembolso = round(total * porcentaje / 100, 2)
    return f"Reembolso del {porcentaje}% sobre {total}€: {reembolso}€"
```

## Paso 3 — Agente LangGraph con RAG + Tools + Memoria

```python
# agente.py
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import operator

# Estado del agente
class EstadoSoporte(TypedDict):
    mensajes: Annotated[Sequence[BaseMessage], operator.add]

# Retriever para RAG
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

tools = [buscar_pedido, calcular_reembolso]
modelo = ChatOpenAI(model="gpt-4o", temperature=0)
modelo_con_tools = modelo.bind_tools(tools)

def nodo_llm(estado: EstadoSoporte) -> dict:
    # Recupera contexto relevante del último mensaje del usuario
    ultimo_humano = next(
        (m.content for m in reversed(estado["mensajes"]) if isinstance(m, HumanMessage)),
        ""
    )
    docs = retriever.invoke(ultimo_humano)
    contexto = "\n".join(d.page_content for d in docs)

    system = SystemMessage(content=f"""Eres un asistente de soporte amable y preciso.
Usa las herramientas disponibles para consultar pedidos y calcular reembolsos.
Responde preguntas sobre políticas usando este contexto:

{contexto}

Si no tienes información, dilo claramente. No inventes datos.""")

    mensajes_con_system = [system] + list(estado["mensajes"])
    respuesta = modelo_con_tools.invoke(mensajes_con_system)
    return {"mensajes": [respuesta]}

def debe_continuar(estado: EstadoSoporte) -> str:
    ultimo = estado["mensajes"][-1]
    if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
        return "usar_tool"
    return END

nodo_tools = ToolNode(tools)

grafo = StateGraph(EstadoSoporte)
grafo.add_node("llm", nodo_llm)
grafo.add_node("tools", nodo_tools)
grafo.set_entry_point("llm")
grafo.add_conditional_edges("llm", debe_continuar, {"usar_tool": "tools", END: END})
grafo.add_edge("tools", "llm")

checkpointer = MemorySaver()
agente = grafo.compile(checkpointer=checkpointer)
```

## Paso 4 — API FastAPI

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

app = FastAPI()

class MensajeRequest(BaseModel):
    session_id: str
    mensaje: str

@app.post("/chat")
def chat(request: MensajeRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    resultado = agente.invoke(
        {"mensajes": [HumanMessage(content=request.mensaje)]},
        config=config
    )
    return {"respuesta": resultado["mensajes"][-1].content}

@app.delete("/chat/{session_id}")
def limpiar_sesion(session_id: str):
    # El MemorySaver no expone borrado directo; en producción usar PostgresCheckpointer
    return {"mensaje": f"Sesión {session_id} cerrada"}
```

## Requisitos

- [ ] La base de conocimiento tiene al menos 10 políticas indexadas en ChromaDB
- [ ] El agente usa RAG para responder preguntas sobre políticas
- [ ] El agente usa las tools para buscar pedidos y calcular reembolsos
- [ ] La memoria mantiene el contexto entre turnos de la misma sesión
- [ ] El endpoint `POST /chat` funciona correctamente
- [ ] Dos sesiones distintas no comparten historial

## Bonus

- Añade una tool `escalar_a_humano(motivo: str)` que registre el caso en un archivo JSON
- Implementa `PostgresCheckpointer` para memoria persistente real
- Añade un endpoint `GET /chat/{session_id}/historial` que devuelva todos los mensajes