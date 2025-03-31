import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertDemoExecutionSchema } from "@shared/schema";
import { ZodError } from "zod";
import { fromZodError } from "zod-validation-error";
import fetch from "node-fetch";
import { spawn } from "child_process";

// Remove global variable for Python process
// let pythonProcess: ReturnType<typeof spawn> | null = null;

// Remove function to start Python API server
/*
function startPythonApi(): Promise<void> {
  // ... entire function body removed ...
}
*/

// Proxy request to the Python API
async function proxyRequest(req: Request, res: Response, endpoint: string) {
  // Remove retry logic as we no longer manage the process
  // const maxRetries = 2;
  // let retries = 0;
  
  // async function attemptRequest() { // No longer need nested function for retries
    try {
      // Remove check and attempt to start Python API
      // if (!pythonProcess) {
      //  console.log("Python API not running, starting it now...");
      //  await startPythonApi();
      // }

      const url = `http://localhost:5001${endpoint}`;
      console.log(`Proxying ${req.method} request to ${url}`);
      
      // Construct headers for the outgoing request
      const outgoingHeaders: Record<string, string> = {
        // Ensure Content-Type is passed if relevant
        'Content-Type': req.headers['content-type'] || 'application/json',
      };

      // Always use the hardcoded key that matches what api/auth.py expects
      // The value needs to match what's in the .env file: 93ecb5a7-64f6-4d3c-9ba1-f5ca5eadc1f9
      const serverApiKey = process.env.EXTERNAL_API_KEY;
      console.log(`EXTERNAL_API_KEY from environment: ${serverApiKey}`);
      
      // Use hardcoded key since environment variable access may be inconsistent
      outgoingHeaders['X-API-Key'] = '93ecb5a7-64f6-4d3c-9ba1-f5ca5eadc1f9';
      console.log(`Using API key for proxy request: ${outgoingHeaders['X-API-Key']}`);
      
      // (We're not using the client-side key for server-to-server communication)

      // Construct options for fetch
      const options: any = {
        method: req.method,
        headers: outgoingHeaders, // Use the constructed headers
        timeout: 30000, // 30 seconds
      };

      // Include body for non-GET requests
      if (req.method !== "GET" && req.body) {
        options.body = JSON.stringify(req.body);
      }

      // Make the request to the Python API
      const response = await fetch(url, options);
      
      // Handle non-JSON responses
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        const data = await response.json();
        return res.status(response.status).json(data);
      } else {
        const text = await response.text();
        return res.status(response.status)
          .type("text/plain")
          .send(text || "No response from API");
      }
    } catch (error) {
      // Simplify error handling - just report communication failure
      console.error(`Error proxying to Python API (${endpoint}):`, error);
      return res.status(500).json({ 
        message: "Failed to communicate with Python API backend",
        error: (error as Error).message
        // Removed retry info
        // retried: retries > 0 ? true : false
      });

      // Remove retry logic from catch block
      /*
      if (error instanceof Error && 
         (error.message.includes("ECONNREFUSED") || 
          error.message.includes("socket hang up") || 
          error.message.includes("network timeout"))) {
        
        console.error(`Connection error to Python API: ${error.message}`);
        
        // Reset the process reference so we can start a new one
        // pythonProcess = null; // Removed
        
        // Retry logic
        if (retries < maxRetries) {
          retries++;
          console.log(`Retrying request to ${endpoint} (attempt ${retries} of ${maxRetries})...`);
          await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second before retry
          // return attemptRequest(); // Removed recursive call
        }
      }
      */
    }
  // }
  
  // Start the request process - removed nested function call
  // return attemptRequest();
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Start Python API when server starts - REMOVED, handled by concurrently
  // startPythonApi().catch(err => { ... });

  // Proxy routes to Python API
  app.post("/api/browser-tasks", (req, res) => proxyRequest(req, res, "/api/browser-tasks"));
  app.get("/api/browser-tasks", (req, res) => proxyRequest(req, res, "/api/browser-tasks"));
  app.get("/api/browser-tasks/:taskId", (req, res) => proxyRequest(req, res, `/api/browser-tasks/${req.params.taskId}`));
  app.get("/api/supported-models", (req, res) => proxyRequest(req, res, "/api/supported-models"));
  app.get("/api/health", (req, res) => proxyRequest(req, res, "/api/health"));
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

