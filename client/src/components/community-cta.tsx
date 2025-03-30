import React from 'react';
import { Button } from '@/components/ui/button';

export function CommunityCta() {
  return (
    <section className="py-16 bg-primary">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="lg:flex lg:items-center lg:justify-between">
          <div>
            <h2 className="text-3xl font-display font-bold tracking-tight text-white sm:text-4xl">
              Join our Community
            </h2>
            <p className="mt-3 text-lg text-primary-100">
              Share your ideas, ask questions, and collaborate with other developers. The fastest growing community for AI web agents.
            </p>
          </div>
          <div className="mt-8 flex lg:mt-0 lg:flex-shrink-0 gap-4">
            <Button asChild variant="secondary" size="lg">
              <a href="#" className="gap-2">
                <i className="fab fa-discord"></i> Discord
              </a>
            </Button>
            <Button asChild variant="outline" size="lg" className="bg-primary-800 bg-opacity-70 hover:bg-opacity-80 text-white border-transparent">
              <a href="#" className="gap-2">
                <i className="fab fa-twitter"></i> Twitter Follow
              </a>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
