# Manual del Asistente de Soporte con LangChain y LangGraph

## Índice

1. [¿Qué hemos construido?](#1-qué-hemos-construido)
2. [Arquitectura general](#2-arquitectura-general)
3. [La base de conocimiento — RAG](#3-la-base-de-conocimiento--rag)
4. [Las Tools — herramientas del agente](#4-las-tools--herramientas-del-agente)
5. [El agente LangGraph](#5-el-agente-langgraph)
6. [El servidor Express](#6-el-servidor-express)
7. [Flujo completo de una petición](#7-flujo-completo-de-una-petición)
8. [Variables de entorno y configuración](#8-variables-de-entorno-y-configuración)

---

## 1. ¿Qué hemos construido?

Hemos construido un **asistente de soporte al cliente** que puede hacer tres cosas distintas dependiendo de lo que el usuario le pregunte:

- Consultar la **base de conocimiento interna** de la empresa (políticas de devoluciones, envíos, garantías...) para responder preguntas sobre ellas. Esto es RAG.
- Usar **herramientas concretas** para realizar acciones: buscar el estado de un pedido, calcular un reembolso, o escalar el caso a un agente humano.
- **Recordar el contexto** de la conversación dentro de la misma sesión, de forma que si el usuario pregunta "¿y cuánto sería el reembolso?" después de haber consultado un pedido, el asistente sabe a qué pedido se refiere.

Todo ello se expone a través de una **API HTTP** con Express, de modo que cualquier frontend (o Postman, o curl) puede interactuar con el asistente.

Las tecnologías principales son:

| Tecnología | Para qué sirve |
|---|---|
| **LangChain** | Marco para trabajar con LLMs, tools, vectorstores y retrievers |
| **LangGraph** | Construir el flujo del agente como un grafo con nodos y aristas |
| **Ollama** | LLM y modelo de embeddings corriendo localmente |
| **Express** | Servidor HTTP que expone la API |
| **Zod** | Validación de esquemas para las tools |

---

## 2. Arquitectura general

Antes de entrar en cada pieza por separado, es útil ver cómo encajan todas juntas. El sistema tiene dos fases bien diferenciadas:

### Fase 1 — Ingestión (se ejecuta una sola vez)

Esta fase ocurre antes de arrancar el servidor. Su único propósito es preparar la base de conocimiento.

```
politicas.txt
     │
     ▼
 TextLoader          ← Lee el archivo como documento
     │
     ▼
 RecursiveCharacter  ← Trocea el texto en fragmentos manejables
 TextSplitter
     │
     ▼
 OpenAIEmbeddings    ← Convierte cada fragmento en un vector numérico
 (bge-m3 / Ollama)
     │
     ▼
 MemoryVectorStore   ← Almacena los vectores en memoria
     │
     ▼
 vector_db.json      ← Serializa los vectores a disco para persistirlos
```

### Fase 2 — Ejecución (mientras el servidor está corriendo)

Cuando llega una petición HTTP al servidor, el flujo es este:

```
Cliente (Postman/curl)
     │
     │  POST /chat  { session_id, mensaje }
     ▼
  Express (app.ts / routes/chat.ts)
     │
     ▼
  Agente LangGraph (agente.ts)
     │
     ├──► Retriever (vector_db.json) ──► contexto RAG
     │
     ├──► LLM (qwen3:14b / Ollama)
     │         │
     │    ¿Necesita tool?
     │         │
     │    Sí ──┴──► ToolNode ──► buscarPedido / calcularReembolso / escalarAHumano
     │    No ──────────────────► Respuesta final
     │
     ▼
  { respuesta: "..." }
     │
     ▼
  Cliente
```

---

## 3. La base de conocimiento — RAG

### ¿Qué es RAG?

RAG son las siglas de *Retrieval-Augmented Generation*, que en español vendría a ser "generación aumentada por recuperación". La idea es simple: en lugar de que el LLM responda solo con lo que aprendió durante su entrenamiento (que puede estar desactualizado o no incluir información privada de la empresa), le damos contexto relevante en cada petición.

El proceso tiene dos partes separadas en el tiempo: primero **ingestamos** la información (la convertimos en vectores y la guardamos), y luego en cada consulta **recuperamos** los fragmentos más relevantes y se los pasamos al LLM junto con la pregunta del usuario.

### El archivo `src/ingestar.ts`

Este script se ejecuta una sola vez con `npm run ingestar`. Su trabajo es leer `politicas.txt`, procesarlo y guardar el resultado en `vector_db.json`.

```typescript
const loader = new TextLoader(path.resolve("politicas.txt"));
const docs = await loader.load();
```

`TextLoader` de LangChain lee el archivo de texto y lo convierte en un array de documentos (`Document[]`). Cada documento tiene un campo `pageContent` con el texto y un campo `metadata` con información sobre el origen (nombre del archivo, etc.).

```typescript
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});
const chunks = await splitter.splitDocuments(docs);
```

Aquí está una de las decisiones más importantes del sistema. Un LLM no puede trabajar eficientemente con documentos muy largos, y además queremos recuperar solo los fragmentos *relevantes* para la pregunta, no todo el documento. El `RecursiveCharacterTextSplitter` trocea el texto en fragmentos de máximo 500 caracteres.

El parámetro `chunkOverlap: 100` es crucial: significa que cada fragmento comparte 100 caracteres con el fragmento anterior. Esto evita que una frase quede cortada justo en la frontera entre dos chunks y se pierda contexto.

```
Chunk 1: [...100 chars solapado con chunk 0...][...400 chars nuevos...]
Chunk 2: [...100 chars solapado con chunk 1...][...400 chars nuevos...]
```

```typescript
const embeddings = new OpenAIEmbeddings({
  model: process.env.OLLAMA_EMBEDDING_MODEL,  // bge-m3
  configuration: {
    baseURL: process.env.OLLAMA_BASE_URL,
    apiKey: "ollama",
  },
});
const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);
```

Los **embeddings** son la pieza mágica de todo esto. Un modelo de embeddings convierte texto en un vector de números (una lista de ~1500 números en el caso de bge-m3). La propiedad clave es que textos con significados similares producen vectores cercanos en el espacio matemático. Así, cuando el usuario pregunta "¿cuánto tarda el envío?", el vector de esa pregunta estará cerca del vector del chunk que habla sobre política de envíos, aunque las palabras exactas no coincidan.

`MemoryVectorStore.fromDocuments` hace dos cosas: genera el embedding de cada chunk llamando al modelo, y almacena el par (texto, vector) en memoria.

```typescript
fs.writeFileSync(
  path.resolve("vector_db.json"),
  JSON.stringify(vectorStore.memoryVectors, null, 2)
);
```

Por defecto, `MemoryVectorStore` vive solo en RAM. Para que el agente pueda reutilizar los embeddings sin tener que regenerarlos cada vez (lo que costaría tiempo y llamadas al modelo), serializamos el array `memoryVectors` a un archivo JSON en disco.

---

## 4. Las Tools — herramientas del agente

### ¿Qué es una tool en este contexto?

Cuando hablamos de "tools" en LangChain, nos referimos a funciones que el LLM puede decidir invocar. El LLM no ejecuta código directamente — lo que hace es "decir" que quiere usar una herramienta concreta con unos argumentos concretos, y es LangChain quien la ejecuta y devuelve el resultado al LLM para que pueda continuar.

Este patrón se llama **function calling** y es fundamental para construir agentes que hagan cosas reales más allá de generar texto.

### El archivo `src/tools.ts`

Cada tool se define con la función `tool()` de `@langchain/core/tools`, que recibe dos argumentos: la función que ejecuta la acción, y un objeto de metadatos que describe la tool al LLM.

```typescript
export const buscarPedido = tool(
  async ({ pedidoId }) => {               // ← función que se ejecuta
    const pedidos = { ... };
    const pedido = pedidos[pedidoId.toUpperCase()];
    return pedido ? JSON.stringify(pedido) : `Pedido ${pedidoId} no encontrado`;
  },
  {
    name: "buscar_pedido",                // ← nombre con el que el LLM la invoca
    description: "Busca el estado...",   // ← descripción para que el LLM sepa cuándo usarla
    schema: z.object({                   // ← qué parámetros espera y de qué tipo
      pedidoId: z.string().describe("El ID del pedido, por ejemplo PED-1234"),
    }),
  }
);
```

Es importante entender la separación de roles aquí:

- El **`name`** y la **`description`** van al LLM para que sepa que esta tool existe y cuándo es apropiado usarla.
- El **`schema`** (definido con Zod) le dice al LLM exactamente qué argumentos tiene que proporcionar y de qué tipo.
- La **función** nunca la ve el LLM — la ejecuta LangChain cuando el LLM decide invocar la tool.

### Zod y la validación de esquemas

Zod es una librería de validación de tipos en tiempo de ejecución. En el contexto de las tools, sirve para definir de forma explícita y tipada qué argumentos acepta cada función. Cuando el LLM decide llamar a `buscar_pedido`, Zod valida que los argumentos que proporciona son correctos antes de ejecutar la función.

```typescript
schema: z.object({
  pedidoId: z.string().describe("El ID del pedido, por ejemplo PED-1234"),
})
```

El método `.describe()` es especialmente importante: ese texto forma parte del mensaje que recibe el LLM, ayudándole a entender qué debe poner en ese campo.

### Las tres tools del proyecto

**`buscarPedido`** simula una consulta a una base de datos de pedidos. En un sistema real, aquí harías una consulta a tu base de datos o a una API externa. El resultado siempre es un string (los LLMs trabajan con texto), y por eso el objeto del pedido se serializa con `JSON.stringify`.

**`calcularReembolso`** realiza un cálculo matemático. Fíjate en el redondeo:
```typescript
const reembolso = Math.round((total * porcentaje) / 100 * 100) / 100;
```
El truco del `* 100 / 100` es para evitar errores de coma flotante típicos de JavaScript (por ejemplo, `89.99 * 0.5` puede dar `44.994999...`).

**`escalarAHumano`** registra el caso en un archivo `casos.json` usando el módulo `fs` de Node. La lógica es sencilla: si el archivo ya existe, lee su contenido, añade el nuevo caso al array, y lo vuelve a escribir. Si no existe, empieza con un array vacío.

---

## 5. El agente LangGraph

### ¿Qué es LangGraph?

LangGraph es una librería que permite construir agentes como **grafos**: estructuras de nodos (donde ocurre el procesamiento) conectados por aristas (que determinan el flujo). La ventaja sobre un simple bucle es que puedes definir lógica condicional — "si el LLM quiere usar una tool, ve al nodo de tools; si no, termina" — de forma explícita y controlada.

Piensa en ello como una máquina de estados: el agente siempre está en alguno de los nodos, y las aristas determinan a qué nodo va a continuación.

### El archivo `src/agente.ts`

#### El estado

Todo en LangGraph gira en torno al **estado**, que es la información compartida entre todos los nodos del grafo.

```typescript
const EstadoSoporte = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
});
```

El estado de nuestro agente tiene un único campo: `messages`, que es un array de mensajes. Lo importante aquí es el **reducer**: es la función que define cómo se actualiza el estado cuando un nodo devuelve nuevos datos. En este caso, los mensajes nuevos se **concatenan** a los existentes. Esto es lo que implementa la memoria de la conversación — los mensajes se van acumulando, y en cada turno el LLM los ve todos.

#### Cómo se carga el vector store

```typescript
function cargarVectorStore(): MemoryVectorStore {
  const vectorStore = new MemoryVectorStore(embeddings);
  const dbPath = path.resolve("vector_db.json");
  if (fs.existsSync(dbPath)) {
    vectorStore.memoryVectors = JSON.parse(fs.readFileSync(dbPath, "utf-8"));
  }
  return vectorStore;
}

const vectorStore = cargarVectorStore();
const retriever = vectorStore.asRetriever(3);
```

Aquí se invierte el proceso de ingestión: creamos un `MemoryVectorStore` vacío y le inyectamos directamente los vectores que guardamos en `vector_db.json`. El `retriever` es la interfaz de búsqueda — `asRetriever(3)` significa "devuélveme los 3 chunks más relevantes para cada consulta".

#### El nodo LLM — el corazón del agente

```typescript
async function nodoLlm(estado: typeof EstadoSoporte.State) {
  const ultimoHumano = [...estado.messages]
    .reverse()
    .find((m) => m instanceof HumanMessage)?.content ?? "";

  const docs = await retriever.invoke(ultimoHumano as string);
  const contexto = docs.map((d) => d.pageContent).join("\n");

  const system = new SystemMessage(`Eres un asistente de soporte...
  
${contexto}

Si no tienes información, dilo claramente. No inventes datos.`);

  const respuesta = await modeloConTools.invoke([system, ...estado.messages]);
  return { messages: [respuesta] };
}
```

Este nodo hace cuatro cosas en cada invocación:

1. **Busca el último mensaje humano** en el historial (recorriendo el array al revés) para usarlo como consulta al retriever.
2. **Recupera los 3 chunks más relevantes** de la base de conocimiento usando el retriever.
3. **Construye el system message** inyectando el contexto recuperado. Esto es el RAG en acción: el LLM recibe el contexto pertinente justo antes de responder.
4. **Invoca el LLM** con el system message y todos los mensajes de la conversación. `modeloConTools` es el modelo con las tools ya registradas, de modo que el LLM puede decidir invocarlas.

El nodo devuelve `{ messages: [respuesta] }`, que el reducer concatenará al array de mensajes existente.

#### El enrutador condicional

```typescript
function debeContinuar(estado: typeof EstadoSoporte.State): string {
  const ultimo = estado.messages.at(-1);
  if (ultimo instanceof AIMessage && ultimo.tool_calls?.length) {
    return "tools";
  }
  return END;
}
```

Después de que el LLM responde, hay que decidir qué hacer a continuación. Si el último mensaje es del LLM y contiene `tool_calls` (es decir, el LLM ha decidido usar una o más tools), la función devuelve `"tools"` para ir al nodo de tools. Si no hay tool calls, devuelve `END` para terminar el grafo y devolver la respuesta.

#### El grafo completo

```typescript
const grafo = new StateGraph(EstadoSoporte)
  .addNode("llm", nodoLlm)
  .addNode("tools", nodoTools)
  .addEdge("__start__", "llm")
  .addConditionalEdges("llm", debeContinuar)
  .addEdge("tools", "llm");
```

Visualmente, el grafo tiene esta forma:

```
         ┌──────────────────────────────┐
         │                              │
    ▼    │                              │
[__start__] ──► [llm] ──┬──► [tools] ──┘
                         │
                         └──► [END]
```

- El punto de entrada siempre es `llm`.
- Después de `llm`, hay una arista condicional: si hay tool calls, va a `tools`; si no, termina.
- Después de `tools`, siempre vuelve a `llm` para que el LLM pueda procesar los resultados de las tools y generar una respuesta final.

Este bucle puede repetirse varias veces si el LLM necesita usar varias tools en secuencia.

#### La memoria entre sesiones

```typescript
const checkpointer = new MemorySaver();
export const agente = grafo.compile({ checkpointer });
```

`MemorySaver` es el mecanismo de persistencia del estado del grafo. Cada vez que el agente completa un turno, guarda el estado completo (todos los mensajes) indexado por un `thread_id`. Cuando llega el siguiente mensaje de la misma sesión, el agente recupera ese estado y continúa desde donde lo dejó.

La clave es que cada sesión de usuario tiene su propio `thread_id`, y por tanto su propio historial completamente separado.

---

## 6. El servidor Express

La API está dividida en tres capas bien diferenciadas:

```
main.ts          ← Arranca el servidor (puerto, listen)
   │
app.ts           ← Configura Express (middleware, monta routers)
   │
routes/chat.ts   ← Define los endpoints y la lógica HTTP
```

### `src/main.ts` — el punto de entrada

```typescript
const PORT = process.env.PORT ?? 3000;
app.listen(PORT, () => {
  console.log(`Servidor escuchando en http://localhost:${PORT}`);
});
```

`main.ts` es el archivo que arrancas con `npm run dev`. Su única responsabilidad es iniciar el servidor en el puerto configurado. No sabe nada de rutas ni de agentes — solo arranca el motor.

### `src/app.ts` — la configuración de Express

```typescript
const app = express();
app.use(express.json());
app.use("/chat", chatRouter);
```

`app.ts` configura Express: activa el middleware que parsea el cuerpo de las peticiones como JSON (`express.json()`), y monta el router de chat bajo el prefijo `/chat`. Esto significa que todas las rutas definidas en `chat.ts` estarán disponibles bajo `/chat/...`.

### `src/routes/chat.ts` — los endpoints

**`POST /chat`** — el endpoint principal

```typescript
router.post("/", async (req, res) => {
  const { session_id, mensaje } = req.body;
  const config = { configurable: { thread_id: session_id } };
  const resultado = await agente.invoke(
    { messages: [new HumanMessage(mensaje)] },
    config
  );
  res.json({ respuesta: resultado.messages.at(-1)?.content });
});
```

Recibe `session_id` y `mensaje` del cuerpo de la petición. El `session_id` se convierte en `thread_id` para el checkpointer — así el agente sabe qué historial cargar. El mensaje del usuario se envuelve en un `HumanMessage` (el tipo de mensaje que representa al usuario en LangChain) y se pasa al agente.

**`GET /chat/:session_id/historial`** — ver el historial de una sesión

```typescript
router.get("/:session_id/historial", async (req, res) => {
  const config = { configurable: { thread_id: session_id } };
  const state = await agente.getState(config);
  const historial = (state.values.messages ?? []).map((m) => ({
    tipo: m instanceof HumanMessage ? "human" : m instanceof AIMessage ? "ai" : "tool",
    contenido: m.content,
  }));
  res.json({ historial });
});
```

`agente.getState(config)` recupera el estado actual del grafo para ese `thread_id` sin ejecutarlo. Esto nos da acceso a todos los mensajes acumulados en esa sesión, que formateamos y devolvemos.

**`DELETE /chat/:session_id`** — cerrar una sesión

Endpoint simbólico: devuelve un mensaje de confirmación. `MemorySaver` no implementa borrado directo, pero en producción con un checkpointer de base de datos real sí se podría eliminar el historial.

---

## 7. Flujo completo de una petición

Para que todo quede claro, veamos paso a paso qué ocurre desde que el usuario manda un mensaje hasta que recibe la respuesta. Usamos como ejemplo: *"¿Cuál es el estado del pedido PED-1234?"*

```
1. Cliente envía:
   POST /chat
   { "session_id": "usuario-1", "mensaje": "¿Cuál es el estado del pedido PED-1234?" }

2. Express (routes/chat.ts) recibe la petición:
   - Valida que session_id y mensaje existen
   - Crea config = { configurable: { thread_id: "usuario-1" } }
   - Llama a agente.invoke({ messages: [HumanMessage("¿Cuál es el estado...")] }, config)

3. LangGraph carga el estado de "usuario-1":
   - Si hay historial previo, lo recupera del MemorySaver
   - Concatena el nuevo HumanMessage al historial

4. Entra al nodo "llm" (nodoLlm):
   - Busca el último mensaje humano: "¿Cuál es el estado del pedido PED-1234?"
   - Consulta el retriever con esa frase → recupera 3 chunks de politicas sobre pedidos/envíos
   - Construye el system message con ese contexto
   - Invoca el LLM con [system, ...historial_completo]

5. El LLM analiza la petición:
   - Ve que el usuario pregunta por un pedido concreto
   - Decide usar la tool "buscar_pedido" con { pedidoId: "PED-1234" }
   - Devuelve un AIMessage con tool_calls = [{ name: "buscar_pedido", args: { pedidoId: "PED-1234" } }]

6. debeContinuar detecta tool_calls → va al nodo "tools"

7. ToolNode ejecuta buscarPedido({ pedidoId: "PED-1234" }):
   - Busca en el diccionario de pedidos
   - Devuelve: '{"estado":"enviado","fechaEntrega":"15/05/2026","total":89.99}'
   - Añade un ToolMessage con ese resultado al estado

8. Vuelve al nodo "llm":
   - El LLM ahora ve el resultado de la tool en el historial
   - Genera una respuesta en lenguaje natural: "El estado de tu pedido PED-1234 es enviado..."
   - Devuelve un AIMessage sin tool_calls

9. debeContinuar no detecta tool_calls → END

10. agente.invoke devuelve el estado completo
    routes/chat.ts extrae el último mensaje: resultado.messages.at(-1).content

11. Express responde:
    { "respuesta": "El estado de tu pedido PED-1234 es enviado, con fecha de entrega el 15/05/2026..." }
```

Lo más importante de este flujo es entender que el LLM puede pasar por el nodo `llm` más de una vez en la misma petición si necesita usar varias tools. LangGraph gestiona ese bucle automáticamente.

---

## 8. Variables de entorno y configuración

Todas las variables sensibles o configurables se guardan en el archivo `.env` en la raíz del proyecto. Nunca se sube al repositorio (está en `.gitignore`), pero `.env.example` documenta las variables necesarias.

```
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_CHAT_MODEL=qwen3:14b
OPENAI_API_KEY=ollama
PORT=3000
```

**`OLLAMA_BASE_URL`** — Ollama expone una API compatible con OpenAI en este endpoint. Por eso usamos las clases `OpenAIEmbeddings` y `ChatOpenAI` de LangChain aunque estemos hablando con Ollama: simplemente le cambiamos la URL base.

**`OLLAMA_EMBEDDING_MODEL`** — el modelo que convierte texto en vectores. `bge-m3` es un modelo especializado en embeddings multilingüe, mucho más eficiente para esta tarea que usar un modelo de chat.

**`OLLAMA_CHAT_MODEL`** — el modelo de lenguaje que genera las respuestas. `qwen3:14b` es el modelo de 14 mil millones de parámetros de Alibaba, con buen rendimiento para tareas de razonamiento y seguimiento de instrucciones.

**`OPENAI_API_KEY=ollama`** — las clases de LangChain para OpenAI exigen que haya una API key configurada, aunque Ollama no la use. Poner cualquier valor (como la cadena "ollama") satisface esa validación sin exponer nada sensible.

**`PORT`** — puerto en el que escucha el servidor Express. Si no se define, el valor por defecto es 3000.
