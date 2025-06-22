import React from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import FolderDashboard from './FolderDashboard';
import { truncateFolderName } from '../utils/folderUtils';

const FolderTabs = ({ 
  filteredFolders, 
  selectedFolder, 
  setSelectedFolder, 
  folderData, 
  loading, 
  loadFolderData, 
  handleFolderDeleted 
}) => {
  return (
    <Tabs value={selectedFolder} onValueChange={setSelectedFolder} className="mb-6">
      <TabsList className="flex w-full overflow-x-auto">
        {filteredFolders.map((folder) => (
          <TabsTrigger 
            key={folder.name} 
            value={folder.name} 
            className="text-sm whitespace-nowrap flex-shrink-0"
          >
            {truncateFolderName(folder.name)}
          </TabsTrigger>
        ))}
      </TabsList>
      {filteredFolders.map((folder) => (
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
  );
};

export default FolderTabs;