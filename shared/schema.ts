import { pgTable, text, serial, integer, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User model for auth
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

// Define demo execution schema
export const demoExecutions = pgTable("demo_executions", {
  id: serial("id").primaryKey(),
  task: text("task").notNull(),
  llmModel: text("llm_model").notNull(),
  result: text("result"),
  status: text("status").notNull().default("pending"),
  createdAt: text("created_at").notNull(),
  userId: integer("user_id").references(() => users.id),
});

// Define examples schema
export const examples = pgTable("examples", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  task: text("task").notNull(),
  imageUrl: text("image_url"),
  codeSnippet: text("code_snippet"),
  featured: boolean("featured").default(false),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertDemoExecutionSchema = createInsertSchema(demoExecutions).pick({
  task: true,
  llmModel: true,
  userId: true,
});

export const insertExampleSchema = createInsertSchema(examples).pick({
  title: true,
  description: true,
  task: true,
  imageUrl: true,
  codeSnippet: true,
  featured: true,
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertDemoExecution = z.infer<typeof insertDemoExecutionSchema>;
export type DemoExecution = typeof demoExecutions.$inferSelect;

export type InsertExample = z.infer<typeof insertExampleSchema>;
export type Example = typeof examples.$inferSelect;
