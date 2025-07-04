import React, { useState, useEffect } from 'react';
import { AlertCircle } from 'lucide-react';
import { Card, CardContent } from './components/ui/card';
import AppHeader from './components/AppHeader';
import SearchBar from './components/SearchBar';
import FolderTabs from './components/FolderTabs';
import FolderDashboard from './components/FolderDashboard';
import UploadDialog from './components/UploadDialog';
import FolderTreeView from './components/FolderTreeView';
import { useFolderManagement } from './hooks/useFolderManagement';
import { useSearch } from './hooks/useSearch';

function App() {
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [showTreeView, setShowTreeView] = useState(false);

  // Use custom hooks for folder management and search
  const {
    folders,
    selectedFolder,
    folderData,
    loading,
    error,
    setSelectedFolder,
    setError,
    loadFolders,
    loadFolderData,
    handleTreeFolderSelect,
    handleUploadSuccess,
    handleFolderDeleted,
  } = useFolderManagement();

  const {
    searchTerm,
    setSearchTerm,
    filteredFolders,
    clearSearch,
  } = useSearch(folders, selectedFolder, setSelectedFolder);

  // Load folder data when selected folder changes
  useEffect(() => {
    if (selectedFolder) {
      loadFolderData(selectedFolder);
    }
  }, [selectedFolder, loadFolderData]);

  const toggleTreeView = () => {
    setShowTreeView(!showTreeView);
    // Clear search when toggling views
    clearSearch();
    if (!showTreeView) {
      // Load folders when showing tree view for the first time
      loadFolders();
    }
  };

  const handleUploadSuccessWrapper = (uploadResults) => {
    handleUploadSuccess(uploadResults);
    // Close the upload dialog
    setUploadDialogOpen(false);
  };

  // Loading state for initial load
  if (loading && folders.length === 0) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <AppHeader
        toggleTreeView={toggleTreeView}
        showTreeView={showTreeView}
        setUploadDialogOpen={setUploadDialogOpen}
        loadFolders={loadFolders}
        hasFolders={folders.length > 0}
      />

      <main className="container mx-auto px-4 py-6">
        {/* Error Display */}
        {error && (
          <div className="mb-4 p-4 border border-destructive/20 rounded-lg bg-destructive/10 text-destructive">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Tree View */}
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

        {/* No Folders Message */}
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

        {/* Folder Tabs View */}
        {!showTreeView && folders.length > 0 && (
          <div>
            <SearchBar
              searchTerm={searchTerm}
              setSearchTerm={setSearchTerm}
              clearSearch={clearSearch}
            />

            {filteredFolders.length === 0 ? (
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center space-x-2 text-muted-foreground">
                    <AlertCircle className="h-4 w-4" />
                    <span>No folders found matching "{searchTerm}". Try a different search term.</span>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <FolderTabs
                filteredFolders={filteredFolders}
                selectedFolder={selectedFolder}
                setSelectedFolder={setSelectedFolder}
                folderData={folderData}
                loading={loading}
                loadFolderData={loadFolderData}
                handleFolderDeleted={handleFolderDeleted}
              />
            )}
          </div>
        )}
      </main>

      <UploadDialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        onUploadSuccess={handleUploadSuccessWrapper}
      />
    </div>
  );
}

export default App;