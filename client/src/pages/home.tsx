import React from 'react';
import { Navbar } from '@/components/navbar';
import { Hero } from '@/components/hero';
import { PerformanceChart } from '@/components/performance-chart';
import { Features } from '@/components/features';
import { DemoConsole } from '@/components/demo-console';
import { Examples } from '@/components/examples';
import { Documentation } from '@/components/documentation';
import { Comparison } from '@/components/comparison';
import { Pricing } from '@/components/pricing';
import { CommunityCta } from '@/components/community-cta';
import { Footer } from '@/components/footer';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-grow">
        <Hero />
        <PerformanceChart />
        <Features />
        <DemoConsole />
        <Examples />
        <Documentation />
        <Comparison />
        <Pricing />
        <CommunityCta />
      </main>
      <Footer />
    </div>
  );
}
