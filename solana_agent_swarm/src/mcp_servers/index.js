#!/usr/bin/env node
/**
 * Main entry point for Solana Agent Swarm MCP Servers
 * This script launches all three MCP servers:
 * - Solana Data MCP
 * - Pattern Learning MCP
 * - Risk Assessment MCP
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Server paths
const SOLANA_DATA_MCP_PATH = path.join(__dirname, 'solana_data_mcp', 'index.js');
const PATTERN_LEARNING_MCP_PATH = path.join(__dirname, 'pattern_learning_mcp', 'index.js');
const RISK_ASSESSMENT_MCP_PATH = path.join(__dirname, 'risk_assessment_mcp', 'index.js');

// Check if files exist
const checkFile = (filePath) => {
  if (!fs.existsSync(filePath)) {
    console.error(`Error: File not found - ${filePath}`);
    process.exit(1);
  }
};

// Verify server files exist
checkFile(SOLANA_DATA_MCP_PATH);
checkFile(PATTERN_LEARNING_MCP_PATH);
checkFile(RISK_ASSESSMENT_MCP_PATH);

// Configure child process options
const spawnOptions = {
  stdio: ['inherit', 'inherit', 'inherit'],
  env: process.env,
  shell: true
};

// Function to start a server
const startServer = (serverPath, serverName) => {
  console.log(`Starting ${serverName}...`);
  
  // Make the file executable
  try {
    fs.chmodSync(serverPath, '755');
  } catch (error) {
    console.warn(`Warning: Could not make ${serverPath} executable: ${error.message}`);
  }
  
  // Spawn the process
  const server = spawn('node', [serverPath], spawnOptions);
  
  server.on('error', (error) => {
    console.error(`Error starting ${serverName}: ${error.message}`);
  });
  
  server.on('exit', (code, signal) => {
    if (code !== 0) {
      console.error(`${serverName} exited with code ${code} and signal ${signal}`);
    } else {
      console.log(`${serverName} exited normally`);
    }
  });
  
  return server;
};

// Start all servers
console.log('Starting Solana Agent Swarm MCP Servers...');

const solanaDataServer = startServer(SOLANA_DATA_MCP_PATH, 'Solana Data MCP');
const patternLearningServer = startServer(PATTERN_LEARNING_MCP_PATH, 'Pattern Learning MCP');
const riskAssessmentServer = startServer(RISK_ASSESSMENT_MCP_PATH, 'Risk Assessment MCP');

// Handle process termination
process.on('SIGINT', () => {
  console.log('Shutting down all MCP servers...');
  
  // The child processes should handle their own cleanup
  process.exit(0);
});

console.log('All MCP servers started. Press Ctrl+C to exit.');
