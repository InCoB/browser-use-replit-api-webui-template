import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';

interface LlmModel {
  id: string;
  name: string;
}

const llmModels: LlmModel[] = [
  { id: 'gpt4o', name: 'GPT-4o' },
  { id: 'claude3', name: 'Claude 3' },
  { id: 'llama2', name: 'Llama 2' },
  { id: 'gemini', name: 'Gemini Pro' }
];

export function DemoConsole() {
  const [task, setTask] = useState('Go to Reddit, search for \'browser-use\', click on the first post and return the first comment.');
  const [selectedModel, setSelectedModel] = useState<string>('gpt4o');
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<string | null>(null);
  const [browserPreview, setBrowserPreview] = useState<string | null>(null);
  
  const handleRunDemo = async () => {
    setIsRunning(true);
    setResults(null);
    setBrowserPreview(null);
    
    try {
      // Simulate API call to run the task
      // In a real implementation, this would call the backend
      setTimeout(() => {
        setResults(`Task completed successfully!\n\nFirst comment from Reddit post about browser-use:\n\n"This tool is incredible! I've been using it to automate some tedious tasks on our company website and it's saving me hours every week. The element tracking feature is especially useful."`);
        setBrowserPreview('Browser preview displayed here');
        setIsRunning(false);
      }, 3000);
    } catch (error) {
      setResults(`Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`);
      setIsRunning(false);
    }
  };
  
  const handleClearDemo = () => {
    setResults(null);
    setBrowserPreview(null);
  };
  
  return (
    <section id="demo" className="py-16 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-display font-bold text-gray-900 sm:text-4xl">
            Try it yourself
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-lg text-gray-600">
            Enter a task for the AI to perform in the browser
          </p>
        </div>
        
        <Card className="mt-12 bg-gray-50 shadow-md overflow-hidden">
          <div className="p-6">
            <div className="grid gap-6 md:grid-cols-5">
              <div className="md:col-span-3">
                <label htmlFor="task" className="block text-sm font-medium text-gray-700 mb-1">Task Description</label>
                <Textarea
                  id="task"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="E.g., Go to Wikipedia, search for 'artificial intelligence', and save the first paragraph"
                  className="min-h-[100px]"
                />
                
                <div className="mt-4 flex flex-wrap gap-2">
                  <div className="text-sm font-medium text-gray-700 mb-1 w-full">LLM Model:</div>
                  {llmModels.map((model) => (
                    <Button
                      key={model.id}
                      variant={selectedModel === model.id ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedModel(model.id)}
                    >
                      {model.name}
                    </Button>
                  ))}
                </div>
                
                <div className="mt-6">
                  <Button 
                    onClick={handleRunDemo} 
                    disabled={isRunning || !task.trim()}
                    className="gap-2"
                  >
                    {isRunning ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Running...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-play"></i> Run Task
                      </>
                    )}
                  </Button>
                  
                  <Button
                    variant="outline"
                    onClick={handleClearDemo}
                    disabled={isRunning || (!results && !browserPreview)}
                    className="ml-3 gap-2"
                  >
                    <i className="fas fa-trash-alt"></i> Clear
                  </Button>
                </div>
              </div>
              
              <div className="md:col-span-2">
                <div className="h-full flex flex-col">
                  <div className="text-sm font-medium text-gray-700 mb-1">Results</div>
                  <div className="flex-1 bg-white border border-gray-200 rounded-md p-4 h-48 overflow-auto">
                    {isRunning ? (
                      <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <svg className="animate-spin h-4 w-4 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Running task...</span>
                      </div>
                    ) : results ? (
                      <pre className="text-sm whitespace-pre-wrap">{results}</pre>
                    ) : (
                      <div className="text-sm text-gray-500 italic">
                        Results will appear here after running the task...
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-4">
                    <div className="text-sm font-medium text-gray-700 mb-1">Browser View</div>
                    <div className="bg-white border border-gray-200 rounded-md p-2 h-32 flex items-center justify-center">
                      {isRunning ? (
                        <div className="w-full h-full bg-gray-100 animate-pulse rounded flex items-center justify-center">
                          <span className="text-sm text-gray-500">Loading browser preview...</span>
                        </div>
                      ) : browserPreview ? (
                        <div className="text-sm">{browserPreview}</div>
                      ) : (
                        <div className="text-center text-gray-400">
                          <i className="fas fa-desktop text-3xl mb-2"></i>
                          <p className="text-sm">Browser preview will appear here</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-100 px-6 py-4 border-t border-gray-200">
            <div className="text-sm text-gray-500">
              <i className="fas fa-info-circle mr-1"></i> This demo runs in the cloud. For local usage, check out our <a href="#documentation" className="text-primary hover:text-primary-700">installation guide</a>.
            </div>
          </div>
        </Card>
      </div>
    </section>
  );
}
