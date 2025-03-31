import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertDemoExecutionSchema } from "@shared/schema";
import { ZodError } from "zod";
import { fromZodError } from "zod-validation-error";
import fetch from "node-fetch";
import { spawn } from "child_process";

// Global variable to hold reference to Python process
let pythonProcess: ReturnType<typeof spawn> | null = null;

// Start Python API server
function startPythonApi(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (pythonProcess) {
      console.log("Python API already running");
      resolve();
      return;
    }

    console.log("Starting Python API...");
    // Pass environment variables to Python process
    pythonProcess = spawn("python", ["api/app.py"], {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1", // Ensure Python output is not buffered
      }
    });

    // Track if the server has started successfully
    let serverStarted = false;
    const startupTimeout = setTimeout(() => {
      if (!serverStarted) {
        console.error("Python API startup timed out after 10 seconds");
        reject(new Error("Python API startup timed out"));
      }
    }, 10000);

    pythonProcess.stdout?.on("data", (data) => {
      const output = data.toString().trim();
      console.log(`Python API: ${output}`);
      
      // Look for indications that the server is running
      if (output.includes("Running on") || output.includes("debugger is active")) {
        serverStarted = true;
        clearTimeout(startupTimeout);
        resolve();
      }
    });

    pythonProcess.stderr?.on("data", (data) => {
      console.error(`Python API Error: ${data.toString().trim()}`);
    });

    pythonProcess.on("error", (error) => {
      console.error(`Failed to start Python API: ${error.message}`);
      clearTimeout(startupTimeout);
      pythonProcess = null;
      reject(error);
    });

    pythonProcess.on("close", (code) => {
      console.log(`Python API process exited with code ${code}`);
      clearTimeout(startupTimeout);
      pythonProcess = null;
      
      // If the server was never marked as started and this isn't during shutdown
      if (!serverStarted) {
        reject(new Error(`Python API process exited with code ${code}`));
      }
    });

    // Fallback resolution if we don't see explicit startup messages
    // but the process is still running after 5 seconds
    setTimeout(() => {
      if (!serverStarted && pythonProcess) {
        console.log("Python API assumed to be running (no explicit startup message detected)");
        serverStarted = true;
        clearTimeout(startupTimeout);
        resolve();
      }
    }, 5000);
  });
}

// Proxy request to the Python API
async function proxyRequest(req: Request, res: Response, endpoint: string) {
  const maxRetries = 2;
  let retries = 0;
  
  async function attemptRequest() {
    try {
      // Ensure Python API is running
      if (!pythonProcess) {
        console.log("Python API not running, starting it now...");
        await startPythonApi();
      }

      const url = `http://localhost:5001${endpoint}`;
      console.log(`Proxying ${req.method} request to ${url}`);
      
      const options: any = {
        method: req.method,
        headers: {
          "Content-Type": "application/json",
        },
        // Add a timeout to the request
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
      // Connection errors might indicate the Python process died
      if (error instanceof Error && 
         (error.message.includes("ECONNREFUSED") || 
          error.message.includes("socket hang up") || 
          error.message.includes("network timeout"))) {
        
        console.error(`Connection error to Python API: ${error.message}`);
        
        // Reset the process reference so we can start a new one
        pythonProcess = null;
        
        // Retry logic
        if (retries < maxRetries) {
          retries++;
          console.log(`Retrying request to ${endpoint} (attempt ${retries} of ${maxRetries})...`);
          await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second before retry
          return attemptRequest();
        }
      }
      
      // If we get here, either it's not a connection error or we've exhausted retries
      console.error(`Error proxying to Python API (${endpoint}):`, error);
      return res.status(500).json({ 
        message: "Failed to communicate with Browser Use API",
        error: (error as Error).message,
        retried: retries > 0 ? true : false
      });
    }
  }
  
  // Start the request process
  return attemptRequest();
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Start Python API when server starts
  startPythonApi().catch(err => {
    console.error("Failed to start Python API on initialization:", err);
  });

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
