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

const EstadoSoporte = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
});

const embeddings = new OpenAIEmbeddings({
  model: process.env.OLLAMA_EMBEDDING_MODEL,
  configuration: {
    baseURL: process.env.OLLAMA_BASE_URL,
    apiKey: "ollama",
  },
});

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
const modelo = new ChatOpenAI({
  model: process.env.OLLAMA_CHAT_MODEL,
  temperature: 0,
  configuration: {
    baseURL: process.env.OLLAMA_BASE_URL,
    apiKey: "ollama",
  },
});
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