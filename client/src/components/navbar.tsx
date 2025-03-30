import React, { useState } from 'react';
import { Link } from 'wouter';
import { Button } from '@/components/ui/button';
import { BrowserUseIcon } from './icons';
import { cn } from '@/lib/utils';
import { useIsMobile } from '@/hooks/use-mobile';

const navLinks = [
  { href: '#features', label: 'Features' },
  { href: '#demo', label: 'Demo' },
  { href: '#examples', label: 'Examples' },
  { href: '#compare', label: 'Compare' },
  { href: '#pricing', label: 'Pricing' },
  { href: '#documentation', label: 'Docs' },
];

export function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const isMobile = useIsMobile();

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="sticky top-0 z-50 w-full bg-background border-b shadow-sm">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 justify-between">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link href="/" className="flex items-center">
                <BrowserUseIcon className="h-6 w-6 text-primary mr-2" />
                <span className="text-primary font-display font-bold text-xl">Browser Use</span>
              </Link>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              {navLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-600 hover:border-primary hover:text-primary transition-colors"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>
          <div className="hidden sm:ml-6 sm:flex sm:items-center space-x-4">
            <a 
              href="https://github.com/browser-use/browser-use" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-gray-900 p-2"
            >
              <i className="fab fa-github text-xl"></i>
            </a>
            <Button asChild>
              <a href="#demo">Try Cloud</a>
            </Button>
          </div>
          
          {isMobile && (
            <div className="flex items-center sm:hidden">
              <button
                type="button"
                onClick={toggleMenu}
                className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none"
                aria-expanded={isMenuOpen ? 'true' : 'false'}
              >
                <span className="sr-only">Open main menu</span>
                <i className={`fas ${isMenuOpen ? 'fa-times' : 'fa-bars'}`} />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Mobile menu */}
      <div className={cn('sm:hidden', isMenuOpen ? 'block' : 'hidden')}>
        <div className="pt-2 pb-3 space-y-1">
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-600 hover:bg-gray-50 hover:border-primary hover:text-primary transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              {link.label}
            </a>
          ))}
          <div className="flex items-center mt-4 pl-3 pr-4 py-2">
            <a 
              href="https://github.com/browser-use/browser-use" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-gray-900 p-2"
            >
              <i className="fab fa-github text-xl"></i>
            </a>
            <Button asChild className="ml-3">
              <a href="#demo" onClick={() => setIsMenuOpen(false)}>Try Cloud</a>
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
}
