import React from 'react';
import { Button } from '@/components/ui/button';

interface PricingFeature {
  name: string;
}

interface PricingPlan {
  name: string;
  price: string;
  unit?: string;
  description: string;
  features: PricingFeature[];
  buttonText: string;
  buttonLink: string;
  highlighted?: boolean;
}

const pricingPlans: PricingPlan[] = [
  {
    name: 'Open Source',
    price: '$0',
    description: 'Perfect for individual developers and open source projects.',
    features: [
      { name: 'Full library access' },
      { name: 'Self-hosted version' },
      { name: 'All core features' },
      { name: 'MIT License' }
    ],
    buttonText: 'View on GitHub',
    buttonLink: 'https://github.com/browser-use/browser-use'
  },
  {
    name: 'Pro',
    price: '$30',
    unit: '/month',
    description: 'For teams and businesses that need advanced features and support.',
    features: [
      { name: 'Everything in Open Source' },
      { name: 'Priority support' },
      { name: 'Includes 30 USD of API credits per month' },
      { name: 'Unlimited access' }
    ],
    buttonText: 'Get Started',
    buttonLink: '#',
    highlighted: true
  },
  {
    name: 'Enterprise',
    price: '$Yes',
    unit: '/month',
    description: 'We\'ll build custom agents tailored to your organization\'s needs.',
    features: [
      { name: 'Dedicated support team' },
      { name: 'SLA guarantees' },
      { name: 'On-premise deployment' },
      { name: 'Custom integrations' }
    ],
    buttonText: 'Contact Us',
    buttonLink: '#'
  }
];

export function Pricing() {
  return (
    <section id="pricing" className="py-16 bg-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-display font-bold text-gray-900 sm:text-4xl">
            Pricing
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-lg text-gray-600">
            Choose Your Plan
          </p>
          <p className="mt-3 max-w-2xl mx-auto text-gray-600">
            From open source to enterprise, we have a plan that fits your needs.
          </p>
        </div>
        
        <div className="mt-12 grid gap-8 md:grid-cols-3">
          {pricingPlans.map((plan, index) => (
            <div 
              key={index} 
              className={`bg-white rounded-lg ${plan.highlighted ? 'shadow-md border-primary-100' : 'shadow-sm border-gray-100'} border overflow-hidden hover:shadow-md transition-shadow relative`}
            >
              {plan.highlighted && (
                <div className="absolute top-0 inset-x-0">
                  <div className="h-1 w-full bg-primary"></div>
                </div>
              )}
              <div className="p-6 border-b border-gray-100">
                <h3 className="text-lg font-display font-semibold text-gray-900">{plan.name}</h3>
                <div className="mt-4 flex items-baseline">
                  <span className="text-5xl font-display font-bold tracking-tight text-gray-900">{plan.price}</span>
                  {plan.unit && <span className="ml-1 text-xl text-gray-500">{plan.unit}</span>}
                </div>
                <p className="mt-5 text-sm text-gray-600">
                  {plan.description}
                </p>
              </div>
              <div className="px-6 pt-6 pb-8">
                <ul role="list" className="space-y-4">
                  {plan.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-start">
                      <div className="flex-shrink-0">
                        <i className="fas fa-check text-green-500"></i>
                      </div>
                      <p className="ml-3 text-sm text-gray-700">{feature.name}</p>
                    </li>
                  ))}
                </ul>
                <div className="mt-8">
                  <Button 
                    asChild
                    variant={plan.highlighted ? "default" : "outline"}
                    className="w-full"
                  >
                    <a href={plan.buttonLink}>{plan.buttonText}</a>
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
