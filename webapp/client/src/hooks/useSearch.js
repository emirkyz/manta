import { useState, useMemo, useEffect } from 'react';

export const useSearch = (folders, selectedFolder, setSelectedFolder) => {
  const [searchTerm, setSearchTerm] = useState('');

  // Filter folders based on search term
  const filteredFolders = useMemo(() => {
    if (!searchTerm.trim()) {
      return folders;
    }
    return folders.filter(folder =>
      folder.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [folders, searchTerm]);

  // Update selected folder if current selection doesn't match search results
  useEffect(() => {
    if (searchTerm && filteredFolders.length > 0) {
      // If current selected folder is not in filtered results, select the first match
      if (!filteredFolders.find(f => f.name === selectedFolder)) {
        setSelectedFolder(filteredFolders[0].name);
      }
    }
  }, [filteredFolders, selectedFolder, searchTerm, setSelectedFolder]);

  const clearSearch = () => {
    setSearchTerm('');
  };

  return {
    searchTerm,
    setSearchTerm,
    filteredFolders,
    clearSearch,
  };
};