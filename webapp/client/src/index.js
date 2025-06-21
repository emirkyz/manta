import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { ThemeProvider } from './components/theme-provider';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProvider defaultTheme="light" storageKey="nmf-ui-theme">
      <App />
    </ThemeProvider>
  </React.StrictMode>
);