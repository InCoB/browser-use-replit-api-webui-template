import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { useQuery, useMutation } from '@tanstack/react-query';
import { useToast } from '@/hooks/use-toast';

// Simple fetch function
async function fetchApi(url: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }
  return response.json();
}

// Simple post function
async function postApi(url: string, data: any) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }
  return response.json();
}

interface LlmModel {
  id: string;
  name: string;
}

interface SimulatedResult {
  model: string;
  result: string;
  screenshot: string;
  simulation: boolean;
  task: string;
}

interface BrowserTask {
  id: string;
  task: string;
  model: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: string | SimulatedResult;
  error?: string;
  created_at: string;
  updated_at: string;
}

interface ApiResponse<T> {
  json(): Promise<T>;
  status: number;
}

export function DemoConsole() {
  const [task, setTask] = useState('Go to Reddit, search for \'browser-use\', click on the first post and return the first comment.');
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [browserPreview, setBrowserPreview] = useState<string | null>(null);
  const { toast } = useToast();
  // Store the task result
  const [results, setResults] = useState<string | null>(null);
  
  // Default models in case API fails
  const defaultModels: LlmModel[] = [
    { id: 'gpt-4o', name: 'GPT-4o' },
    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
    { id: 'gpt-4', name: 'GPT-4' }
  ];
  
  // Fetch supported models from the API
  const { data: models = defaultModels } = useQuery({
    queryKey: ['/api/supported-models'],
    queryFn: async ({ queryKey }) => {
      try {
        const response = await fetchApi('/api/supported-models');
        return response as LlmModel[];
      } catch (error) {
        console.error('Failed to fetch models:', error);
        // Return default models if the API fails
        return defaultModels;
      }
    }
  });
  
  const [selectedModel, setSelectedModel] = useState<string>('gpt-4o');
  
  // Set default model when models are loaded
  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      setSelectedModel(models[0].id);
    }
  }, [models]);
  
  // Create a new browser task
  const createTaskMutation = useMutation({
    mutationFn: async () => {
      const response = await postApi('/api/browser-tasks', {
        task,
        model: selectedModel,
      });
      return response as { id: string; status: string };
    },
    onSuccess: (data) => {
      setCurrentTaskId(data.id);
      toast({
        title: 'Task created',
        description: 'Browser task has been created and is now running.',
      });
    },
    onError: (error) => {
      toast({
        title: 'Error',
        description: `Failed to create task: ${error instanceof Error ? error.message : 'Unknown error'}`,
        variant: 'destructive',
      });
    },
  });
  
  // Get task status and result
  const { data: taskData, refetch } = useQuery({
    queryKey: ['/api/browser-tasks', currentTaskId],
    queryFn: async ({ queryKey }) => {
      if (!currentTaskId) return null;
      const response = await fetchApi(`/api/browser-tasks/${currentTaskId}`);
      return response as BrowserTask;
    },
    enabled: !!currentTaskId,
    refetchInterval: (data: any) => {
      if (!data) return 2000;
      // Poll more frequently if task is still running
      return (data.status === 'completed' || data.status === 'failed') ? false : 2000;
    },
  });
  
  // Start a browser task
  const handleRunDemo = async () => {
    try {
      createTaskMutation.mutate();
    } catch (error) {
      toast({
        title: 'Error',
        description: `Failed to start browser task: ${error instanceof Error ? error.message : 'Unknown error'}`,
        variant: 'destructive',
      });
    }
  };
  
  // Clear the current task
  const handleClearDemo = () => {
    setCurrentTaskId(null);
    setBrowserPreview(null);
  };
  
  // Update results when taskData changes
  useEffect(() => {
    if (taskData) {
      if (taskData.status === 'completed' && taskData.result) {
        // Handle the result which could be a string or an object
        if (typeof taskData.result === 'string') {
          setResults(taskData.result);
        } else if (typeof taskData.result === 'object') {
          // Extract the result string from the object
          const resultObj = taskData.result as any;
          // Update browser preview if there's a screenshot
          if (resultObj.screenshot) {
            setBrowserPreview(resultObj.screenshot);
          }
          // Use the string result if available
          setResults(resultObj.result || JSON.stringify(resultObj, null, 2));
        }
      } else if (taskData.status === 'failed' && taskData.error) {
        setResults(`Error: ${taskData.error}`);
      }
    }
  }, [taskData]);
  
  // Clear results when clearing the task
  useEffect(() => {
    if (!currentTaskId) {
      setResults(null);
    }
  }, [currentTaskId]);
  
  // Determine if the task is currently running
  const isRunning = createTaskMutation.isPending || 
                   (taskData?.status === 'pending' || taskData?.status === 'running');
  
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
                  {models.map((model: LlmModel) => (
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
                        <span>Running task... {taskData?.status === 'running' && 'The AI agent is now controlling the browser.'}</span>
                      </div>
                    ) : results ? (
                      typeof results === 'string' && results.startsWith('Error:') ? (
                        <div className="text-sm text-red-600 whitespace-pre-wrap">
                          <div className="font-bold mb-1">An error occurred:</div>
                          <div>{(results as string).replace('Error: ', '')}</div>
                          
                          {(results as string).includes('Browser automation error') && (
                            <div className="mt-3 p-2 bg-red-50 rounded border border-red-200">
                              <strong>Browser dependency issue detected</strong>
                              <p className="mt-1">This error is often caused by missing system dependencies required by Playwright for browser automation. The browser-use library needs these dependencies to control web browsers.</p>
                            </div>
                          )}
                        </div>
                      ) : (
                        <pre className="text-sm whitespace-pre-wrap">{results}</pre>
                      )
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
                        browserPreview.startsWith('data:image') ? (
                          <img src={browserPreview} alt="Browser preview" className="max-w-full max-h-full object-contain" />
                        ) : (
                          <div className="text-sm">{browserPreview}</div>
                        )
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
