import React from 'react';
import { RefreshCw, Folder, Upload, List } from 'lucide-react';
import { Button } from './ui/button';
import { ThemeToggle } from './ui/theme-toggle';

const AppHeader = ({ 
  toggleTreeView, 
  showTreeView, 
  setUploadDialogOpen, 
  loadFolders, 
  hasFolders 
}) => {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4">
        <div className="flex items-center space-x-4">
          <Folder className="h-6 w-6 text-primary" />
          <h1 className="text-xl font-semibold">NMF Analysis Dashboard</h1>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleTreeView}
            className="flex items-center space-x-2"
          >
            <List className="h-4 w-4" />
            <span>{showTreeView ? 'Hide' : 'Browse'} Output</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setUploadDialogOpen(true)}
            className="flex items-center space-x-2"
          >
            <Upload className="h-4 w-4" />
            <span>Upload</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={loadFolders}
            disabled={!hasFolders}
            className="flex items-center space-x-2"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </Button>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
};

export default AppHeader;