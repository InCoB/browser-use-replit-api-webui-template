import React from 'react';

interface Feature {
  icon: string;
  title: string;
  description: string;
}

const features: Feature[] = [
  {
    icon: 'fa-eye',
    title: 'Vision + HTML Extraction',
    description: 'Combines visual understanding with HTML structure extraction for comprehensive web interaction.'
  },
  {
    icon: 'fa-window-restore',
    title: 'Multi-tab Management',
    description: 'Automatically handles multiple browser tabs for complex workflows and parallel processing.'
  },
  {
    icon: 'fa-sitemap',
    title: 'Element Tracking',
    description: 'Extracts clicked elements XPaths and repeats exact LLM actions for consistent automation.'
  },
  {
    icon: 'fa-puzzle-piece',
    title: 'Custom Actions',
    description: 'Add your own actions like saving to files, database operations, notifications, or human input handling.'
  },
  {
    icon: 'fa-sync-alt',
    title: 'Self-correcting',
    description: 'Intelligent error handling and automatic recovery for robust automation workflows.'
  },
  {
    icon: 'fa-brain',
    title: 'Any LLM Support',
    description: 'Compatible with all LangChain LLMs including GPT-4, Claude 3, and Llama 2.'
  }
];

export function Features() {
  return (
    <section id="features" className="py-16 bg-gray-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-display font-bold text-gray-900 sm:text-4xl">
            Features
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-lg text-gray-600">
            Powerful Browser Automation
          </p>
          <p className="mt-3 max-w-3xl mx-auto text-md text-gray-500">
            Browser Use combines advanced AI capabilities with robust browser automation to make web interactions seamless for AI agents.
          </p>
        </div>
        
        <div className="mt-12 grid gap-8 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => (
            <div key={index} className="bg-white rounded-lg shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
              <div className="h-12 w-12 rounded-md bg-primary-100 text-primary flex items-center justify-center mb-4">
                <i className={`fas ${feature.icon} text-xl`}></i>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
