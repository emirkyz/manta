import React, { useState, useEffect, useCallback } from 'react';
import { RefreshCw, Folder, Upload, List, AlertCircle } from 'lucide-react';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './components/ui/tabs';
import { Card, CardContent } from './components/ui/card';
import { ThemeToggle } from './components/ui/theme-toggle';
import FolderDashboard from './components/FolderDashboard';
import UploadDialog from './components/UploadDialog';
import FolderTreeView from './components/FolderTreeView';

function App() {
  const [folders, setFolders] = useState([]);
  const [selectedFolder, setSelectedFolder] = useState('');
  const [folderData, setFolderData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [showTreeView, setShowTreeView] = useState(false);

  const loadFolders = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/folders');
      setFolders(response.data);
      if (response.data.length > 0 && !selectedFolder) {
        setSelectedFolder(response.data[0].name);
      }
      setError(null);
    } catch (err) {
      setError('Failed to load folders: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedFolder]);

  const loadFolderData = useCallback(async (folderName) => {
    try {
      setLoading(true);
      const response = await axios.get(`/api/folder/${folderName}`);
      setFolderData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load folder data: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Removed automatic folder loading on startup
  // Users will now manually trigger loading with a button

  useEffect(() => {
    if (selectedFolder) {
      loadFolderData(selectedFolder);
    }
  }, [selectedFolder, loadFolderData]);


  const handleTreeFolderSelect = (folderName) => {
    setSelectedFolder(folderName);
    // Add the selected folder to the folders list if it's not already there
    if (!folders.find(f => f.name === folderName)) {
      setFolders(prevFolders => [...prevFolders, { name: folderName, path: folderName }]);
    }
  };

  const toggleTreeView = () => {
    setShowTreeView(!showTreeView);
    if (!showTreeView) {
      // Load folders when showing tree view for the first time
      loadFolders();
    }
  };

  const handleUploadSuccess = (uploadResults) => {
    // Add only the successfully uploaded folders to the current list
    const successfulUploads = uploadResults.results
      .filter(result => result.status === 'success')
      .map(result => ({
        name: result.name,
        path: result.name // The folder name is used as path reference
      }));
    
    // Add new folders to the existing list
    setFolders(prevFolders => [...prevFolders, ...successfulUploads]);
    
    // Select the first uploaded folder if no folder is currently selected
    if (successfulUploads.length > 0 && !selectedFolder) {
      setSelectedFolder(successfulUploads[0].name);
    }
    
    // Close the upload dialog
    setUploadDialogOpen(false);
  };

  const handleFolderDeleted = (deletedFolderName) => {
    // Remove the deleted folder from the list
    setFolders(prevFolders => 
      prevFolders.filter(folder => folder.name !== deletedFolderName)
    );
    
    // If the deleted folder was selected, switch to the first available folder
    if (selectedFolder === deletedFolderName) {
      const remainingFolders = folders.filter(folder => folder.name !== deletedFolderName);
      if (remainingFolders.length > 0) {
        setSelectedFolder(remainingFolders[0].name);
      } else {
        setSelectedFolder('');
        setFolderData(null);
      }
    }
  };

  if (loading && folders.length === 0) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Modern Header */}
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
              disabled={folders.length === 0}
              className="flex items-center space-x-2"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Refresh</span>
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {error && (
          <div className="mb-4 p-4 border border-destructive/20 rounded-lg bg-destructive/10 text-destructive">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {showTreeView && (
          <div className="mb-6">
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
              <div className="md:col-span-4">
                <FolderTreeView
                  onFolderSelect={handleTreeFolderSelect}
                  selectedFolder={selectedFolder}
                />
              </div>
              <div className="md:col-span-8">
                {selectedFolder && folderData ? (
                  <FolderDashboard
                    folderData={folderData}
                    onRefresh={() => loadFolderData(selectedFolder)}
                    onFolderDeleted={handleFolderDeleted}
                  />
                ) : (
                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center space-x-2 text-muted-foreground">
                        <AlertCircle className="h-4 w-4" />
                        <span>Select an analysis folder from the tree to view its contents.</span>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </div>
        )}

        {!showTreeView && folders.length === 0 && (
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-2 text-muted-foreground">
                <AlertCircle className="h-4 w-4" />
                <span>No analysis folders loaded. Click "Browse Output" to view existing analyses or "Upload" to add new ones.</span>
              </div>
            </CardContent>
          </Card>
        )}

        {!showTreeView && folders.length > 0 && (
          <div>
            <Tabs value={selectedFolder} onValueChange={setSelectedFolder} className="mb-6">
              <TabsList className="grid w-full grid-cols-1 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
                {folders.slice(0, 6).map((folder) => (
                  <TabsTrigger key={folder.name} value={folder.name} className="text-sm">
                    {folder.name.length > 15 ? `${folder.name.substring(0, 15)}...` : folder.name}
                  </TabsTrigger>
                ))}
              </TabsList>
              {folders.map((folder) => (
                <TabsContent key={folder.name} value={folder.name}>
                  {loading ? (
                    <div className="flex justify-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                    </div>
                  ) : (
                    folderData && (
                      <FolderDashboard
                        folderData={folderData}
                        onRefresh={() => loadFolderData(selectedFolder)}
                        onFolderDeleted={handleFolderDeleted}
                      />
                    )
                  )}
                </TabsContent>
              ))}
            </Tabs>
          </div>
        )}
      </main>

      <UploadDialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />
    </div>
  );
}

export default App;