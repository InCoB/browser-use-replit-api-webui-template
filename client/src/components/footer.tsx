import React from 'react';
import { BrowserUseIcon } from './icons';

interface FooterLinkGroup {
  title: string;
  links: Array<{
    name: string;
    href: string;
  }>;
}

const footerGroups: FooterLinkGroup[] = [
  {
    title: 'Resources',
    links: [
      { name: 'Documentation', href: '#' },
      { name: 'Posts', href: '#' },
      { name: 'PyPI', href: 'https://pypi.org/project/browser-use/' }
    ]
  },
  {
    title: 'Community',
    links: [
      { name: 'Twitter', href: '#' },
      { name: 'LinkedIn', href: '#' },
      { name: 'GitHub', href: 'https://github.com/browser-use/browser-use' },
      { name: 'Discord', href: '#' }
    ]
  },
  {
    title: 'Legal',
    links: [
      { name: 'Privacy Policy', href: '#' },
      { name: 'Terms of Service', href: '#' },
      { name: 'License', href: 'https://github.com/browser-use/browser-use/blob/main/LICENSE' }
    ]
  }
];

export function Footer() {
  return (
    <footer className="bg-gray-900 text-white">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center text-xl font-display font-bold mb-4">
              <BrowserUseIcon className="h-6 w-6 mr-2" />
              <span>Browser Use</span>
            </div>
            <p className="text-gray-400 text-sm mb-4">Making websites accessible for AI agents</p>
            <p className="text-gray-500 text-xs">© {new Date().getFullYear()} Browser Use. All rights reserved.</p>
          </div>
          
          {footerGroups.map((group, groupIndex) => (
            <div key={groupIndex}>
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">{group.title}</h3>
              <ul className="space-y-3">
                {group.links.map((link, linkIndex) => (
                  <li key={linkIndex}>
                    <a href={link.href} className="text-gray-300 hover:text-white text-sm">
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        
        <div className="mt-12 pt-8 border-t border-gray-800 flex flex-col md:flex-row items-center justify-between">
          <div className="flex items-center mb-4 md:mb-0">
            <span className="text-gray-400 text-sm">Made with ❤️ in Zurich and San Francisco</span>
          </div>
          <div className="flex space-x-6">
            <a href="https://github.com/browser-use/browser-use" className="text-gray-400 hover:text-gray-300">
              <i className="fab fa-github text-xl"></i>
            </a>
            <a href="#" className="text-gray-400 hover:text-gray-300">
              <i className="fab fa-twitter text-xl"></i>
            </a>
            <a href="#" className="text-gray-400 hover:text-gray-300">
              <i className="fab fa-discord text-xl"></i>
            </a>
            <a href="#" className="text-gray-400 hover:text-gray-300">
              <i className="fab fa-linkedin text-xl"></i>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
