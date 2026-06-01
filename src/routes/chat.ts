import { Router, Request, Response } from "express";
import { HumanMessage } from "@langchain/core/messages";
import { agente } from "../agente";

const router = Router();

interface ChatBody {
  session_id: string;
  mensaje: string;
}

router.post("/", async (req: Request<{}, {}, ChatBody>, res: Response) => {
  const { session_id, mensaje } = req.body;

  if (!session_id || !mensaje) {
    res.status(400).json({ error: "Se requieren session_id y mensaje" });
    return;
  }

  try {
    const config = { configurable: { thread_id: session_id } };
    const resultado = await agente.invoke(
      { messages: [new HumanMessage(mensaje)] },
      config
    );
    res.json({ respuesta: resultado.messages.at(-1)?.content ?? "Sin respuesta" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error interno del servidor" });
  }
});

router.delete("/:session_id", (req: Request, res: Response) => {
  const { session_id } = req.params;
  res.json({ mensaje: `Sesión ${session_id} cerrada` });
});

export default router;
