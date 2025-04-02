#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log("Starting the browser-companion application...");

// Ensure Python environment
try {
  console.log("Checking Python environment...");
  
  // Create .pythonlibs directory if it doesn't exist
  if (!fs.existsSync('.pythonlibs')) {
    fs.mkdirSync('.pythonlibs');
    fs.mkdirSync('.pythonlibs/bin');
  }
  
  // Create a simple Python wrapper script
  const pythonWrapper = `#!/bin/bash
python "$@"
`;
  
  fs.writeFileSync('.pythonlibs/bin/python', pythonWrapper);
  execSync('chmod +x .pythonlibs/bin/python');
  
  console.log("Python environment set up.");
} catch (error) {
  console.error("Error setting up Python environment:", error);
}

// Start the application using npm run dev
try {
  console.log("Starting the application...");
  execSync('npm run dev', { stdio: 'inherit' });
} catch (error) {
  console.error("Error starting the application:", error);
  process.exit(1);
}