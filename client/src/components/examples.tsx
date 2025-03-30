import React from 'react';
import { Button } from '@/components/ui/button';

interface Example {
  title: string;
  description: string;
  image: string;
}

const examples: Example[] = [
  {
    title: 'AI Did My Groceries',
    description: 'Add grocery items to cart, and checkout.',
    image: 'https://placehold.co/600x400/e2e8f0/1e293b?text=Grocery+Shopping+Automation'
  },
  {
    title: 'LinkedIn to Salesforce',
    description: 'Add my latest LinkedIn follower to my leads in Salesforce.',
    image: 'https://placehold.co/600x400/e2e8f0/1e293b?text=LinkedIn+to+Salesforce'
  },
  {
    title: 'ML Job Application Helper',
    description: 'Read my CV & find ML jobs, save them to a file, and start applying for them.',
    image: 'https://placehold.co/600x400/e2e8f0/1e293b?text=Job+Application+Automation'
  }
];

export function Examples() {
  return (
    <section id="examples" className="py-16 bg-gray-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-display font-bold text-gray-900 sm:text-4xl">
            Examples
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-lg text-gray-600">
            See what others are building with Browser Use
          </p>
        </div>
        
        <div className="mt-12 grid gap-8 md:grid-cols-2 lg:grid-cols-3">
          {examples.map((example, index) => (
            <div key={index} className="bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow">
              <div className="aspect-w-16 aspect-h-9 bg-gray-100">
                <img 
                  src={example.image} 
                  alt={example.title} 
                  className="object-cover w-full h-full" 
                />
              </div>
              <div className="p-5">
                <h3 className="font-semibold text-lg text-gray-900 mb-2">{example.title}</h3>
                <p className="text-gray-600 mb-3 text-sm">{example.description}</p>
                <div className="pt-2 border-t border-gray-100">
                  <Button variant="link" className="text-primary p-0 h-auto text-sm font-medium flex items-center">
                    See this example <i className="fas fa-chevron-right ml-1 text-xs"></i>
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-12 text-center">
          <Button variant="outline">
            View more examples
          </Button>
        </div>
      </div>
    </section>
  );
}
