import React from 'react';
import { cn } from '@/lib/utils';

interface CodeBlockProps {
  children: React.ReactNode;
  className?: string;
  language?: 'javascript' | 'typescript' | 'python' | 'bash';
  dark?: boolean;
}

export function CodeBlock({ 
  children, 
  className, 
  language = 'javascript',
  dark = true,
  ...props 
}: CodeBlockProps) {
  return (
    <div className={cn(
      'rounded-lg overflow-hidden',
      dark ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900',
      className
    )}>
      <div className={cn(
        'px-4 py-2 flex items-center',
        dark ? 'bg-white bg-opacity-20' : 'bg-gray-100'
      )}>
        <div className="h-3 w-3 rounded-full bg-red-500 mr-2"></div>
        <div className="h-3 w-3 rounded-full bg-yellow-500 mr-2"></div>
        <div className="h-3 w-3 rounded-full bg-green-500"></div>
        <div className="ml-3 text-sm font-mono">{language === 'bash' ? 'terminal' : `${language}.${language === 'python' ? 'py' : language === 'javascript' ? 'js' : 'ts'}`}</div>
      </div>
      <pre className="p-4 font-mono text-sm overflow-auto">
        <code>{children}</code>
      </pre>
    </div>
  );
}

interface InlineCodeProps {
  children: React.ReactNode;
  className?: string;
}

export function InlineCode({ children, className }: InlineCodeProps) {
  return (
    <code className={cn('px-1.5 py-0.5 rounded bg-gray-100 text-primary-600 font-mono text-sm', className)}>
      {children}
    </code>
  );
}
