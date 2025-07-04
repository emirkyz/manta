import { useState, useCallback } from 'react';
import axios from 'axios';

export const useFolderManagement = () => {
  const [folders, setFolders] = useState([]);
  const [selectedFolder, setSelectedFolder] = useState('');
  const [folderData, setFolderData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

  const handleTreeFolderSelect = useCallback((folderName) => {
    setSelectedFolder(folderName);
    // Add the selected folder to the folders list if it's not already there
    if (!folders.find(f => f.name === folderName)) {
      setFolders(prevFolders => [...prevFolders, { name: folderName, path: folderName }]);
    }
  }, [folders]);

  const handleUploadSuccess = useCallback((uploadResults) => {
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
  }, [selectedFolder]);

  const handleFolderDeleted = useCallback((deletedFolderName) => {
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
  }, [selectedFolder, folders]);

  return {
    // State
    folders,
    selectedFolder,
    folderData,
    loading,
    error,
    // Actions
    setFolders,
    setSelectedFolder,
    setFolderData,
    setError,
    // Methods
    loadFolders,
    loadFolderData,
    handleTreeFolderSelect,
    handleUploadSuccess,
    handleFolderDeleted,
  };
};