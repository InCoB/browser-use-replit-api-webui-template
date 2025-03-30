import React from 'react';
import { Button } from '@/components/ui/button';
import { CodeBlock } from './codeblock';
import { YCombinatorLogo } from './icons';

export function Hero() {
  return (
    <div className="bg-gradient-to-r from-primary to-secondary-600 text-white py-16">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="lg:flex lg:items-center lg:justify-between">
          <div className="lg:w-6/12">
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-bold tracking-tight">
              Enable AI to control your browser
            </h1>
            <p className="mt-6 text-xl max-w-3xl">
              Browser Use makes websites accessible for AI agents by extracting all interactive elements,
              allowing agents to focus on solving your tasks seamlessly.
            </p>
            <div className="mt-8 flex flex-col sm:flex-row gap-4">
              <Button asChild variant="secondary" size="lg">
                <a href="#demo">Try the Demo</a>
              </Button>
              <Button asChild variant="outline" size="lg" className="bg-primary-800 bg-opacity-60 hover:bg-opacity-70 text-white border-transparent">
                <a href="#documentation">View Documentation</a>
              </Button>
            </div>
            <div className="mt-6 flex items-center">
              <div className="flex-shrink-0">
                <YCombinatorLogo className="h-10 w-10" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-white">Backed by Y Combinator</p>
                <p className="text-xs text-white text-opacity-80">$17M seed round to build the 🦄</p>
              </div>
            </div>
          </div>
          
          <div className="mt-10 lg:mt-0 lg:w-5/12">
            <CodeBlock>
              <div><span className="text-purple-400">import</span> <span className="text-white">{'{ Agent }'}</span> <span className="text-purple-400">from</span> <span className="text-green-400">'browser-use'</span>;</div>
              <div><span className="text-purple-400">import</span> <span className="text-white">{'{ ChatOpenAI }'}</span> <span className="text-purple-400">from</span> <span className="text-green-400">'langchain_openai'</span>;</div>
              <br />
              <div><span className="text-purple-400">const</span> <span className="text-blue-400">agent</span> <span className="text-white">=</span> <span className="text-purple-400">new</span> <span className="text-yellow-400">Agent</span><span className="text-white">({'{'}</span></div>
              <div>&nbsp;&nbsp;<span className="text-blue-300">task</span><span className="text-white">:</span> <span className="text-green-400">"Search for best laptops for developers"</span><span className="text-white">,</span></div>
              <div>&nbsp;&nbsp;<span className="text-blue-300">llm</span><span className="text-white">:</span> <span className="text-purple-400">new</span> <span className="text-yellow-400">ChatOpenAI</span><span className="text-white">({'{'}</span> <span className="text-blue-300">model</span><span className="text-white">:</span> <span className="text-green-400">"gpt-4o"</span> <span className="text-white">{'}'})</span></div>
              <div><span className="text-white">{'}'})</span>;</div>
              <br />
              <div><span className="text-blue-400">agent</span><span className="text-white">.</span><span className="text-yellow-400">run</span><span className="text-white">()</span></div>
              <div>&nbsp;&nbsp;<span className="text-white">.</span><span className="text-yellow-400">then</span><span className="text-white">(</span><span className="text-purple-400">result</span> <span className="text-white">{'=>'}</span> <span className="text-blue-400">console</span><span className="text-white">.</span><span className="text-yellow-400">log</span><span className="text-white">(</span><span className="text-purple-400">result</span><span className="text-white">))</span></div>
              <div>&nbsp;&nbsp;<span className="text-white">.</span><span className="text-yellow-400">catch</span><span className="text-white">(</span><span className="text-purple-400">error</span> <span className="text-white">{'=>'}</span> <span className="text-blue-400">console</span><span className="text-white">.</span><span className="text-yellow-400">error</span><span className="text-white">(</span><span className="text-purple-400">error</span><span className="text-white">));</span></div>
            </CodeBlock>
          </div>
        </div>
      </div>
    </div>
  );
}
