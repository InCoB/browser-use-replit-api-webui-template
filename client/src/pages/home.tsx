import React from 'react';
// import { Navbar } from '@/components/navbar'; // Removed
// import { Hero } from '@/components/hero'; // Removed
// import { PerformanceChart } from '@/components/performance-chart'; // Removed
// import { Features } from '@/components/features'; // Removed
import { DemoConsole } from '@/components/demo-console'; // Keep
// import { Examples } from '@/components/examples'; // Removed
// import { Documentation } from '@/components/documentation'; // Removed
// import { Comparison } from '@/components/comparison'; // Removed
// import { Pricing } from '@/components/pricing'; // Removed
// import { CommunityCta } from '@/components/community-cta'; // Removed
// import { Footer } from '@/components/footer'; // Removed

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* <Navbar /> */}
      <main className="flex-grow container mx-auto px-4 py-8"> {/* Added some basic padding/container */}
        {/* <Hero /> */}
        {/* <PerformanceChart /> */}
        {/* <Features /> */}
        <DemoConsole /> {/* Keep only the demo console */}
        {/* <Examples /> */}
        {/* <Documentation /> */}
        {/* <Comparison /> */}
        {/* <Pricing /> */}
        {/* <CommunityCta /> */}
      </main>
      {/* <Footer /> */}
    </div>
  );
}
