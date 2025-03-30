import React from 'react';

interface ComparisonFeature {
  name: string;
  browserUse: boolean;
  webVoyager: boolean;
  computerUse: boolean;
  operator: boolean;
}

const features: ComparisonFeature[] = [
  {
    name: 'Vision + HTML Extraction',
    browserUse: true,
    webVoyager: false,
    computerUse: true,
    operator: true
  },
  {
    name: 'Multi-tab Management',
    browserUse: true,
    webVoyager: true,
    computerUse: false,
    operator: true
  },
  {
    name: 'Element Tracking',
    browserUse: true,
    webVoyager: false,
    computerUse: false,
    operator: true
  },
  {
    name: 'Custom Actions',
    browserUse: true,
    webVoyager: false,
    computerUse: false,
    operator: true
  },
  {
    name: 'Self-correcting',
    browserUse: true,
    webVoyager: false,
    computerUse: true,
    operator: false
  },
  {
    name: 'Any LLM Support',
    browserUse: true,
    webVoyager: false,
    computerUse: true,
    operator: false
  }
];

export function Comparison() {
  return (
    <section id="compare" className="py-16 bg-gray-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-display font-bold text-gray-900 sm:text-4xl">
            Feature Comparison
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-lg text-gray-600">
            See how Browser Use stacks up against other solutions
          </p>
        </div>
        
        <div className="mt-12 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 rounded-lg overflow-hidden shadow-sm">
              <thead className="bg-gray-100">
                <tr>
                  <th scope="col" className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 sm:pl-6">Feature</th>
                  <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900 w-40">Browser Use</th>
                  <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900 w-40">Web Voyager</th>
                  <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900 w-40">Computer Use</th>
                  <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900 w-40">Operator</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {features.map((feature, index) => (
                  <tr key={index}>
                    <td className="py-4 pl-4 pr-3 text-sm font-medium text-gray-900 sm:pl-6">{feature.name}</td>
                    <td className="px-3 py-4 text-center text-sm text-gray-900">
                      {feature.browserUse ? (
                        <i className="fas fa-check text-green-500"></i>
                      ) : (
                        <i className="fas fa-times text-gray-300"></i>
                      )}
                    </td>
                    <td className="px-3 py-4 text-center text-sm text-gray-900">
                      {feature.webVoyager ? (
                        <i className="fas fa-check text-green-500"></i>
                      ) : (
                        <i className="fas fa-times text-gray-300"></i>
                      )}
                    </td>
                    <td className="px-3 py-4 text-center text-sm text-gray-900">
                      {feature.computerUse ? (
                        <i className="fas fa-check text-green-500"></i>
                      ) : (
                        <i className="fas fa-times text-gray-300"></i>
                      )}
                    </td>
                    <td className="px-3 py-4 text-center text-sm text-gray-900">
                      {feature.operator ? (
                        <i className="fas fa-check text-green-500"></i>
                      ) : (
                        <i className="fas fa-times text-gray-300"></i>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
