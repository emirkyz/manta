import React, { useState, useEffect } from 'react';
import { Folder, FolderOpen, ChevronDown, ChevronRight, FileText, AlertCircle } from 'lucide-react';
import axios from 'axios';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

function FolderTreeView({ onFolderSelect, selectedFolder }) {
  const [outputFolders, setOutputFolders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(true);

  const loadOutputFolders = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get('/api/folders');
      setOutputFolders(response.data);
    } catch (err) {
      setError('Failed to load Output folders: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOutputFolders();
  }, []);

  const handleExpandClick = () => {
    setExpanded(!expanded);
  };

  const handleFolderClick = (folderName) => {
    onFolderSelect(folderName);
  };

  const getMetadataDisplay = (folder) => {
    // Extract metadata from folder name
    const parts = folder.name.split('_');
    const metadata = {
      dataset: parts[0] || 'Unknown',
      algorithm: parts.find(p => p === 'nmf' || p === 'pnmf' || p === 'opnmf') || 'Unknown',
      tokenizer: parts.find(p => p === 'bpe' || p === 'wordpiece' || p === 'w') || 'Unknown',
      topicCount: parts.find(p => /^\d+$/.test(p)) || 'Unknown'
    };

    return (
      <div className="mt-2 flex flex-wrap gap-1">
        <Badge variant="outline" className="text-xs">
          {metadata.algorithm.toUpperCase()}
        </Badge>
        <Badge variant="outline" className="text-xs">
          {metadata.tokenizer}
        </Badge>
        <Badge variant="outline" className="text-xs">
          {metadata.topicCount} topics
        </Badge>
      </div>
    );
  };

  if (loading && outputFolders.length === 0) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
            <span className="text-sm text-muted-foreground">Loading Output folder...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-2 text-destructive">
            <AlertCircle className="h-4 w-4" />
            <span className="text-sm">{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div>
          {/* Main folder header */}
          <Button
            variant="ghost"
            onClick={handleExpandClick}
            className="w-full justify-start p-2 h-auto"
          >
            <div className="flex items-center gap-2 w-full">
              {expanded ? (
                <FolderOpen className="h-4 w-4 text-primary" />
              ) : (
                <Folder className="h-4 w-4 text-primary" />
              )}
              <div className="flex-1 text-left">
                <div className="font-medium">Output Folder</div>
                <div className="text-xs text-muted-foreground">
                  {outputFolders.length} analysis folders
                </div>
              </div>
              {expanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </div>
          </Button>
          
          {/* Folder list */}
          {expanded && (
            <div className="ml-6 mt-2 space-y-1">
              {outputFolders.length === 0 ? (
                <div className="p-3 text-center">
                  <div className="text-sm font-medium text-muted-foreground">
                    No analysis folders found
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Upload new analyses or run the NMF standalone script
                  </div>
                </div>
              ) : (
                outputFolders.map((folder) => (
                  <Button
                    key={folder.name}
                    variant={selectedFolder === folder.name ? "default" : "ghost"}
                    onClick={() => handleFolderClick(folder.name)}
                    className="w-full justify-start p-3 h-auto rounded-md"
                  >
                    <div className="flex items-start gap-3 w-full">
                      <FileText className={`h-4 w-4 mt-0.5 ${
                        selectedFolder === folder.name ? 'text-primary-foreground' : 'text-primary'
                      }`} />
                      <div className="flex-1 text-left">
                        <div className={`font-medium text-sm ${
                          selectedFolder === folder.name ? 'text-primary-foreground' : 'text-foreground'
                        }`}>
                          {folder.name}
                        </div>
                        <div className={`text-xs mt-1 ${
                          selectedFolder === folder.name ? 'text-primary-foreground/80' : 'text-muted-foreground'
                        }`}>
                          {folder.files?.length || 0} files
                        </div>
                        {getMetadataDisplay(folder)}
                      </div>
                    </div>
                  </Button>
                ))
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default FolderTreeView;