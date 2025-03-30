import { 
  users, type User, type InsertUser,
  demoExecutions, type DemoExecution, type InsertDemoExecution,
  examples, type Example, type InsertExample
} from "@shared/schema";

// Interface defining all storage operations
export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  // Demo execution operations
  getDemoExecution(id: number): Promise<DemoExecution | undefined>;
  createDemoExecution(demo: InsertDemoExecution & { createdAt: string }): Promise<DemoExecution>;
  updateDemoExecution(id: number, updates: Partial<DemoExecution>): Promise<DemoExecution | undefined>;
  
  // Example operations
  getExample(id: number): Promise<Example | undefined>;
  getAllExamples(): Promise<Example[]>;
  getFeaturedExamples(): Promise<Example[]>;
  createExample(example: InsertExample): Promise<Example>;
}

// In-memory storage implementation
export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private demoExecutions: Map<number, DemoExecution>;
  private examples: Map<number, Example>;
  private userId: number;
  private demoId: number;
  private exampleId: number;

  constructor() {
    this.users = new Map();
    this.demoExecutions = new Map();
    this.examples = new Map();
    this.userId = 1;
    this.demoId = 1;
    this.exampleId = 1;
    
    // Initialize with some example data
    this.initExamples();
  }

  // User operations
  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.userId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }
  
  // Demo execution operations
  async getDemoExecution(id: number): Promise<DemoExecution | undefined> {
    return this.demoExecutions.get(id);
  }
  
  async createDemoExecution(demo: InsertDemoExecution & { createdAt: string }): Promise<DemoExecution> {
    const id = this.demoId++;
    const demoExecution: DemoExecution = { 
      ...demo, 
      id, 
      result: null,
      status: "pending"
    };
    this.demoExecutions.set(id, demoExecution);
    return demoExecution;
  }
  
  async updateDemoExecution(id: number, updates: Partial<DemoExecution>): Promise<DemoExecution | undefined> {
    const demoExecution = this.demoExecutions.get(id);
    if (!demoExecution) return undefined;
    
    const updatedExecution = { ...demoExecution, ...updates };
    this.demoExecutions.set(id, updatedExecution);
    return updatedExecution;
  }
  
  // Example operations
  async getExample(id: number): Promise<Example | undefined> {
    return this.examples.get(id);
  }
  
  async getAllExamples(): Promise<Example[]> {
    return Array.from(this.examples.values());
  }
  
  async getFeaturedExamples(): Promise<Example[]> {
    return Array.from(this.examples.values()).filter(example => example.featured);
  }
  
  async createExample(example: InsertExample): Promise<Example> {
    const id = this.exampleId++;
    const newExample: Example = { ...example, id };
    this.examples.set(id, newExample);
    return newExample;
  }
  
  // Initialize some example data
  private initExamples() {
    const examples = [
      {
        title: 'AI Did My Groceries',
        description: 'Add grocery items to cart, and checkout.',
        task: 'Go to an online grocery store, add milk, bread, and eggs to the cart, then proceed to checkout.',
        imageUrl: 'https://placehold.co/600x400/e2e8f0/1e293b?text=Grocery+Shopping+Automation',
        codeSnippet: `from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio

async def shop_groceries():
    agent = Agent(
        task="Go to Walmart.com, search for milk, bread, and eggs, add them to cart and proceed to checkout.",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)

asyncio.run(shop_groceries())`,
        featured: true
      },
      {
        title: 'LinkedIn to Salesforce',
        description: 'Add my latest LinkedIn follower to my leads in Salesforce.',
        task: 'Login to LinkedIn, check the latest follower, then login to Salesforce and add them as a lead.',
        imageUrl: 'https://placehold.co/600x400/e2e8f0/1e293b?text=LinkedIn+to+Salesforce',
        codeSnippet: `from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio

async def linkedin_to_salesforce():
    agent = Agent(
        task="Login to LinkedIn, get my latest follower's details, then log into Salesforce and add them as a new lead.",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)

asyncio.run(linkedin_to_salesforce())`,
        featured: true
      },
      {
        title: 'ML Job Application Helper',
        description: 'Read my CV & find ML jobs, save them to a file, and start applying for them.',
        task: 'Upload CV to job site, search for machine learning jobs, save the top 5 to a file, then apply to each one.',
        imageUrl: 'https://placehold.co/600x400/e2e8f0/1e293b?text=Job+Application+Automation',
        codeSnippet: `from langchain_openai import ChatOpenAI
from browser_use import Agent, Action
import asyncio

def save_jobs(jobs, filename="ml_jobs.txt"):
    with open(filename, "w") as f:
        for i, job in enumerate(jobs, 1):
            f.write(f"{i}. {job}\\n")
    return f"Saved {len(jobs)} jobs to {filename}"

async def apply_for_jobs():
    custom_actions = [
        Action(
            name="save_jobs",
            function=save_jobs
        )
    ]
    
    agent = Agent(
        task="Go to Indeed.com, search for 'machine learning engineer', save the top 5 job listings, then apply to each with my CV.",
        llm=ChatOpenAI(model="gpt-4o"),
        actions=custom_actions
    )
    result = await agent.run()
    print(result)

asyncio.run(apply_for_jobs())`,
        featured: true
      }
    ];
    
    examples.forEach((example, index) => {
      this.examples.set(index + 1, { ...example, id: index + 1 });
    });
    
    this.exampleId = examples.length + 1;
  }
}

export const storage = new MemStorage();
