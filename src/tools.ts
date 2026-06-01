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