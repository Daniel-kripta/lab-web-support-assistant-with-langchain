![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Asistente de Soporte con LangChain Completo

## Objetivo

Construir un asistente de soporte al cliente que combine todo lo aprendido hoy:
- **RAG** para consultar una base de conocimiento de la empresa (FAQs, políticas)
- **Memoria** para mantener el contexto de la conversación
- **Tools** para acciones concretas (buscar pedido, calcular reembolso)
- **LangGraph** para controlar el flujo con lógica condicional
- **Express** como interfaz HTTP

## Setup

```bash
# fork & clone the repository
cd lab-web-support-assistant-with-langchain
npm install
cp .env.example .env
# Edita .env y añade tu OPENAI_API_KEY
```

## Paso 1 — Base de conocimiento (RAG)

Crea un archivo `politicas.txt` con al menos 10 políticas de la empresa (devoluciones, envíos, garantía, etc.). Ingestalo en el vector store:

```typescript
// ingestar.ts
import * as dotenv from "dotenv";
dotenv.config();

import * as fs from "fs";
import * as path from "path";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

async function ingestar() {
  const loader = new TextLoader(path.resolve("politicas.txt"));
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });
  const chunks = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
  const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

  fs.writeFileSync(
    path.resolve("vector_db.json"),
    JSON.stringify(vectorStore.memoryVectors, null, 2)
  );
  console.log(`Indexados ${chunks.length} chunks en vector_db.json`);
}

ingestar().catch(console.error);
```

Ejecuta el script de ingestión:

```bash
npm run ingestar
```

## Paso 2 — Tools del asistente

```typescript
// src/tools.ts
import { tool } from "@langchain/core/tools";
import { z } from "zod";

export const buscarPedido = tool(
  async ({ pedidoId }) => {
    const pedidos: Record<string, object> = {
      "PED-1234": { estado: "enviado", fechaEntrega: "15/05/2026", total: 89.99 },
      "PED-5678": { estado: "en preparación", fechaEntrega: "18/05/2026", total: 45.5 },
    };
    const pedido = pedidos[pedidoId.toUpperCase()];
    return pedido ? JSON.stringify(pedido) : `Pedido ${pedidoId} no encontrado`;
  },
  {
    name: "buscar_pedido",
    description: "Busca el estado de un pedido por su ID. Ejemplo: buscar_pedido('PED-1234')",
    schema: z.object({
      pedidoId: z.string().describe("El ID del pedido, por ejemplo PED-1234"),
    }),
  }
);

export const calcularReembolso = tool(
  async ({ total, porcentaje }) => {
    const reembolso = Math.round((total * porcentaje) / 100 * 100) / 100;
    return `Reembolso del ${porcentaje}% sobre ${total}€: ${reembolso}€`;
  },
  {
    name: "calcular_reembolso",
    description: "Calcula el importe de un reembolso parcial dado el total y el porcentaje.",
    schema: z.object({
      total: z.number().describe("Importe total del pedido en euros"),
      porcentaje: z.number().describe("Porcentaje de reembolso a aplicar"),
    }),
  }
);
```

## Paso 3 — Agente LangGraph con RAG + Tools + Memoria

```typescript
// src/agente.ts
import * as dotenv from "dotenv";
dotenv.config();

import * as fs from "fs";
import * as path from "path";
import { Annotation, END, StateGraph } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { buscarPedido, calcularReembolso } from "./tools";

// Estado del agente
const EstadoSoporte = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
});

// RAG: carga el vector store serializado desde disco
const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });

function cargarVectorStore(): MemoryVectorStore {
  const vectorStore = new MemoryVectorStore(embeddings);
  const dbPath = path.resolve("vector_db.json");
  if (fs.existsSync(dbPath)) {
    vectorStore.memoryVectors = JSON.parse(fs.readFileSync(dbPath, "utf-8"));
  } else {
    console.warn("vector_db.json no encontrado. Ejecuta 'npm run ingestar' primero.");
  }
  return vectorStore;
}

const vectorStore = cargarVectorStore();
const retriever = vectorStore.asRetriever(3);

const tools = [buscarPedido, calcularReembolso];
const modelo = new ChatOpenAI({ model: "gpt-4o", temperature: 0 });
const modeloConTools = modelo.bindTools(tools);

async function nodoLlm(estado: typeof EstadoSoporte.State) {
  const ultimoHumano = [...estado.messages]
    .reverse()
    .find((m) => m instanceof HumanMessage)?.content ?? "";

  const docs = await retriever.invoke(ultimoHumano as string);
  const contexto = docs.map((d) => d.pageContent).join("\n");

  const system = new SystemMessage(`Eres un asistente de soporte amable y preciso.
Usa las herramientas disponibles para consultar pedidos y calcular reembolsos.
Responde preguntas sobre políticas usando este contexto:

${contexto}

Si no tienes información, dilo claramente. No inventes datos.`);

  const respuesta = await modeloConTools.invoke([system, ...estado.messages]);
  return { messages: [respuesta] };
}

function debeContinuar(estado: typeof EstadoSoporte.State): string {
  const ultimo = estado.messages.at(-1);
  if (ultimo instanceof AIMessage && ultimo.tool_calls?.length) {
    return "tools";
  }
  return END;
}

const nodoTools = new ToolNode(tools);

const grafo = new StateGraph(EstadoSoporte)
  .addNode("llm", nodoLlm)
  .addNode("tools", nodoTools)
  .addEdge("__start__", "llm")
  .addConditionalEdges("llm", debeContinuar)
  .addEdge("tools", "llm");

const checkpointer = new MemorySaver();
export const agente = grafo.compile({ checkpointer });
```

## Paso 4 — API Express

```typescript
// src/main.ts
import * as dotenv from "dotenv";
dotenv.config();

import express, { Request, Response } from "express";
import { HumanMessage } from "@langchain/core/messages";
import { agente } from "./agente";

const app = express();
app.use(express.json());

interface ChatBody {
  session_id: string;
  mensaje: string;
}

app.post("/chat", async (req: Request<{}, {}, ChatBody>, res: Response) => {
  const { session_id, mensaje } = req.body;

  if (!session_id || !mensaje) {
    res.status(400).json({ error: "Se requieren session_id y mensaje" });
    return;
  }

  const config = { configurable: { thread_id: session_id } };
  const resultado = await agente.invoke(
    { messages: [new HumanMessage(mensaje)] },
    config
  );
  res.json({ respuesta: resultado.messages.at(-1)?.content });
});

app.delete("/chat/:session_id", (req: Request, res: Response) => {
  const { session_id } = req.params;
  // MemorySaver no expone borrado directo; en producción usar un checkpointer con BD
  res.json({ mensaje: `Sesión ${session_id} cerrada` });
});

app.listen(process.env.PORT ?? 3000, () => {
  console.log(`Servidor escuchando en http://localhost:${process.env.PORT ?? 3000}`);
});
```

## Ejecución

```bash
# Paso 1: indexar la base de conocimiento (genera vector_db.json)
npm run ingestar

# Paso 2: iniciar el servidor en modo desarrollo
npm run dev
```

## Probar la API

```bash
# Preguntar sobre políticas (usa RAG)
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "usuario-1", "mensaje": "¿Cuál es la política de devoluciones?"}'

# Consultar un pedido (usa tool)
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "usuario-1", "mensaje": "¿Cuál es el estado del pedido PED-1234?"}'

# Calcular reembolso (usa tool)
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "usuario-1", "mensaje": "Calcula el reembolso del 50% sobre 89.99€"}'

# Cerrar sesión
curl -X DELETE http://localhost:3000/chat/usuario-1
```

## Equivalencias Python → TypeScript

| Python | TypeScript |
|---|---|
| `@tool` decorator | `tool()` de `@langchain/core/tools` + schema Zod |
| `TypedDict` + `operator.add` | `Annotation.Root` + reducer |
| `FastAPI` + `uvicorn` | `Express` + `tsx` |
| `ChromaDB` (embebido) | `MemoryVectorStore` serializado a JSON |
| `MemorySaver` | `MemorySaver` (mismo nombre) |
| `ToolNode` | `ToolNode` (mismo nombre) |
| `set_entry_point("llm")` | `.addEdge("__start__", "llm")` |

## Requisitos

- [ ] La base de conocimiento tiene al menos 10 políticas indexadas
- [ ] El agente usa RAG para responder preguntas sobre políticas
- [ ] El agente usa las tools para buscar pedidos y calcular reembolsos
- [ ] La memoria mantiene el contexto entre turnos de la misma sesión
- [ ] El endpoint `POST /chat` funciona correctamente
- [ ] Dos sesiones distintas no comparten historial

## Bonus 1 — Tool `escalarAHumano`

Añade en `src/tools.ts` una tool que registre el caso en `casos.json` cuando el agente no pueda resolver el problema:

```typescript
export const escalarAHumano = tool(
  async ({ motivo }) => {
    const caso = { fecha: new Date().toISOString(), motivo };
    const casos = fs.existsSync("casos.json")
      ? JSON.parse(fs.readFileSync("casos.json", "utf-8"))
      : [];
    casos.push(caso);
    fs.writeFileSync("casos.json", JSON.stringify(casos, null, 2));
    return "Caso escalado correctamente. Un agente humano se pondrá en contacto contigo pronto.";
  },
  {
    name: "escalar_a_humano",
    description: "Escala el caso a un agente humano cuando no puedes resolver el problema del cliente.",
    schema: z.object({
      motivo: z.string().describe("Motivo por el que se escala el caso"),
    }),
  }
);
```

Añádela al array `tools` en `agente.ts` e indica al agente en el system prompt que la use cuando no pueda resolver el problema.

## Bonus 2 — Checkpointer persistente con SQLite

Por defecto `MemorySaver` guarda la memoria en RAM — se pierde al reiniciar. Para persistirla en disco instala el paquete de checkpointer SQLite (requiere `@langchain/core >= 1.x`):

```bash
npm install @langchain/langgraph-checkpoint-sqlite
```

Y sustitúyelo en `agente.ts`:

```typescript
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";

const checkpointer = SqliteSaver.fromConnString("checkpoints.db");
```

## Bonus 3 — Endpoint `GET /chat/:session_id/historial`

Añade en `src/routes/chat.ts` un endpoint que devuelva todos los mensajes de una sesión:

```typescript
router.get("/:session_id/historial", async (req, res) => {
  const { session_id } = req.params;
  const config = { configurable: { thread_id: session_id } };
  const state = await agente.getState(config);
  const historial = (state.values.messages ?? []).map((m: BaseMessage) => ({
    tipo: m instanceof HumanMessage ? "human" : m instanceof AIMessage ? "ai" : "tool",
    contenido: m.content,
  }));
  res.json({ historial });
});
```
