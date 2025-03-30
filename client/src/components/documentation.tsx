import React from 'react';
import { CodeBlock } from './codeblock';

export function Documentation() {
  return (
    <section id="documentation" className="py-16 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-display font-bold text-gray-900 sm:text-4xl">
            Get Started
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-lg text-gray-600">
            Browser Use is easy to set up and use
          </p>
        </div>
        
        <div className="mt-12 grid gap-8 md:grid-cols-2">
          <div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Installation</h3>
            <CodeBlock language="bash">
              <div className="mb-2"># Install with pip</div>
              <div className="text-green-400">pip install browser-use</div>
              <div className="mt-4 mb-2"># Install playwright</div>
              <div className="text-green-400">playwright install</div>
            </CodeBlock>
            
            <h3 className="text-xl font-semibold text-gray-900 mt-8 mb-4">Basic Usage</h3>
            <CodeBlock language="python">
              <div><span className="text-blue-400">from</span> langchain_openai <span className="text-blue-400">import</span> ChatOpenAI</div>
              <div><span className="text-blue-400">from</span> browser_use <span className="text-blue-400">import</span> Agent</div>
              <div><span className="text-blue-400">import</span> asyncio</div>
              <div><span className="text-blue-400">from</span> dotenv <span className="text-blue-400">import</span> load_dotenv</div>
              <div>load_dotenv()</div>
              <div>&nbsp;</div>
              <div><span className="text-blue-400">async def</span> <span className="text-yellow-400">main</span>():</div>
              <div>    agent = Agent(</div>
              <div>        task=<span className="text-green-400">"Go to Reddit, search for 'browser-use'"</span>,</div>
              <div>        llm=ChatOpenAI(model=<span className="text-green-400">"gpt-4o"</span>),</div>
              <div>    )</div>
              <div>    result = <span className="text-blue-400">await</span> agent.run()</div>
              <div>    <span className="text-blue-400">print</span>(result)</div>
              <div>&nbsp;</div>
              <div>asyncio.run(main())</div>
            </CodeBlock>
            
            <h3 className="text-xl font-semibold text-gray-900 mt-8 mb-4">Required Environment Variables</h3>
            <CodeBlock language="bash">
              <div><span className="text-gray-400"># Add to your .env file</span></div>
              <div>OPENAI_API_KEY=your_api_key_here</div>
            </CodeBlock>
          </div>
          
          <div>
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Advanced Configuration</h3>
            <div className="bg-gray-50 rounded-lg p-6 shadow-sm">
              <div className="mb-6">
                <h4 className="text-lg font-medium text-gray-900 mb-2">Custom Actions</h4>
                <p className="text-gray-600 text-sm mb-3">Define your own custom actions to extend functionality.</p>
                <CodeBlock language="python">
                  <div><span className="text-blue-400">from</span> browser_use <span className="text-blue-400">import</span> Agent, Action</div>
                  <div>&nbsp;</div>
                  <div><span className="text-blue-400">def</span> <span className="text-yellow-400">save_to_file</span>(content, filename=<span className="text-green-400">"output.txt"</span>):</div>
                  <div>    <span className="text-blue-400">with</span> <span className="text-blue-400">open</span>(filename, <span className="text-green-400">"w"</span>) <span className="text-blue-400">as</span> f:</div>
                  <div>        f.write(content)</div>
                  <div>    <span className="text-blue-400">return</span> <span className="text-green-400">"Saved to " + filename</span></div>
                  <div>&nbsp;</div>
                  <div>custom_actions = [</div>
                  <div>    Action(</div>
                  <div>        name=<span className="text-green-400">"save_to_file"</span>,</div>
                  <div>        function=save_to_file</div>
                  <div>    )</div>
                  <div>]</div>
                  <div>&nbsp;</div>
                  <div>agent = Agent(</div>
                  <div>    task=<span className="text-green-400">"Go to Wikipedia..."</span>,</div>
                  <div>    llm=ChatOpenAI(model=<span className="text-green-400">"gpt-4o"</span>),</div>
                  <div>    actions=custom_actions</div>
                  <div>)</div>
                </CodeBlock>
              </div>
              
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-2">Using Different LLMs</h4>
                <p className="text-gray-600 text-sm mb-3">Browser Use works with any LangChain-compatible LLM.</p>
                <div className="space-y-2">
                  <div className="bg-gray-100 rounded p-2 text-sm">
                    <i className="fas fa-check-circle text-green-500 mr-1"></i>
                    <span className="font-medium">OpenAI</span>: GPT-3.5, GPT-4, GPT-4o
                  </div>
                  <div className="bg-gray-100 rounded p-2 text-sm">
                    <i className="fas fa-check-circle text-green-500 mr-1"></i>
                    <span className="font-medium">Anthropic</span>: Claude 3, Claude Instant
                  </div>
                  <div className="bg-gray-100 rounded p-2 text-sm">
                    <i className="fas fa-check-circle text-green-500 mr-1"></i>
                    <span className="font-medium">Meta</span>: Llama 2, Llama 3
                  </div>
                  <div className="bg-gray-100 rounded p-2 text-sm">
                    <i className="fas fa-check-circle text-green-500 mr-1"></i>
                    <span className="font-medium">Google</span>: Gemini Pro, Gemini Ultra
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mt-6 text-center">
              <a href="#" className="text-primary hover:text-primary-800 font-medium inline-flex items-center">
                View Full Documentation <i className="fas fa-arrow-right ml-1"></i>
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
