import express from "express";
import chatRouter from "./routes/chat";

const app = express();
app.use(express.json());
app.use("/chat", chatRouter);

export default app;
