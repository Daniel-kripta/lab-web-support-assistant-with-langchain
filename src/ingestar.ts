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

  const embeddings = new OpenAIEmbeddings({
    model: process.env.OLLAMA_EMBEDDING_MODEL,
    configuration: {
      baseURL: process.env.OLLAMA_BASE_URL,
      apiKey: "ollama",
    },
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

  fs.writeFileSync(
    path.resolve("vector_db.json"),
    JSON.stringify(vectorStore.memoryVectors, null, 2)
  );
  console.log(`Indexados ${chunks.length} chunks en vector_db.json`);
}

ingestar().catch(console.error);