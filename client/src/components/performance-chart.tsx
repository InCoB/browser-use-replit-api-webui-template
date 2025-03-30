import React from 'react';
import { BarChart, Bar } from './ui/chart';

interface Tool {
  name: string;
  accuracy: number;
  highlight?: boolean;
}

const tools: Tool[] = [
  { name: 'Browser Use', accuracy: 89, highlight: true },
  { name: 'Web Voyager', accuracy: 50 },
  { name: 'Computer Use', accuracy: 52 },
  { name: 'Runner H 0.1', accuracy: 67 },
  { name: 'Operator', accuracy: 87 }
];

export function PerformanceChart() {
  return (
    <section className="py-16 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-display font-bold text-gray-900 sm:text-4xl">
            State of the art performance
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-lg text-gray-600">
            Browser Use outperforms other tools with higher accuracy and reliability.
          </p>
        </div>
        
        <div className="mt-12">
          <div className="bg-gray-50 rounded-lg p-6 shadow-sm">
            <h3 className="text-xl font-display font-semibold mb-6">Web Agent Accuracy</h3>
            
            <BarChart>
              {tools.map((tool) => (
                <Bar
                  key={tool.name}
                  label={tool.name}
                  value={tool.accuracy}
                  color={tool.highlight ? 'bg-primary' : 'bg-gray-500'}
                />
              ))}
            </BarChart>
            
            <div className="text-right text-xs text-gray-500 mt-4">
              *For more details, see the <a href="#" className="text-primary hover:text-primary-800">technical report</a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
