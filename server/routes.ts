import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertDemoExecutionSchema } from "@shared/schema";
import { ZodError } from "zod";
import { fromZodError } from "zod-validation-error";

export async function registerRoutes(app: Express): Promise<Server> {
  // Demo execution endpoint
  app.post("/api/demo/execute", async (req, res) => {
    try {
      const demoData = insertDemoExecutionSchema.parse({
        ...req.body,
        userId: req.body.userId || null // Allow anonymous demos
      });
      
      // In a real implementation, this would connect to Browser Use
      // For now, we'll simulate a demo execution with the storage
      const demoExecution = await storage.createDemoExecution({
        ...demoData,
        createdAt: new Date().toISOString(),
      });
      
      // Simulate an asynchronous process for demo execution
      // In a real implementation, this would call Browser Use library
      setTimeout(async () => {
        const result = `Task completed successfully!\n\nFirst comment from Reddit post about browser-use:\n\n"This tool is incredible! I've been using it to automate some tedious tasks on our company website and it's saving me hours every week. The element tracking feature is especially useful."`;
        
        await storage.updateDemoExecution(demoExecution.id, {
          result,
          status: "completed"
        });
      }, 3000);
      
      res.status(201).json({ 
        id: demoExecution.id,
        status: "pending" 
      });
    } catch (error) {
      if (error instanceof ZodError) {
        const validationError = fromZodError(error);
        res.status(400).json({ message: validationError.message });
      } else {
        console.error("Error executing demo:", error);
        res.status(500).json({ message: "Failed to execute demo" });
      }
    }
  });
  
  // Get demo execution status
  app.get("/api/demo/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({ message: "Invalid demo execution ID" });
      }
      
      const demoExecution = await storage.getDemoExecution(id);
      if (!demoExecution) {
        return res.status(404).json({ message: "Demo execution not found" });
      }
      
      res.json(demoExecution);
    } catch (error) {
      console.error("Error retrieving demo execution:", error);
      res.status(500).json({ message: "Failed to retrieve demo execution" });
    }
  });
  
  // Get examples
  app.get("/api/examples", async (req, res) => {
    try {
      const examples = await storage.getAllExamples();
      res.json(examples);
    } catch (error) {
      console.error("Error retrieving examples:", error);
      res.status(500).json({ message: "Failed to retrieve examples" });
    }
  });
  
  // Get featured examples
  app.get("/api/examples/featured", async (req, res) => {
    try {
      const examples = await storage.getFeaturedExamples();
      res.json(examples);
    } catch (error) {
      console.error("Error retrieving featured examples:", error);
      res.status(500).json({ message: "Failed to retrieve featured examples" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
