import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { Badge } from './ui/badge';
import OverviewDashboard from './OverviewDashboard';
import TopicVisualization from './TopicVisualization';
import DocumentAnalysis from './DocumentAnalysis';
import CoherenceAnalysis from './CoherenceAnalysis';
import FolderManagement from './FolderManagement';

function FolderDashboard({ folderData, onRefresh, onFolderDeleted }) {
  const [activeTab, setActiveTab] = useState("overview");

  if (!folderData) {
    return (
      <Card>
        <CardContent className="p-6">
          <h3 className="text-lg font-semibold">No data available</h3>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <CardTitle className="text-2xl font-bold">
              {folderData.name}
            </CardTitle>
            <FolderManagement
              folderName={folderData.name}
              onFolderDeleted={onFolderDeleted}
              folderData={folderData}
            />
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <span className="text-sm text-muted-foreground">Dataset: </span>
              <Badge variant="outline" className="ml-1">
                {folderData.metadata?.dataset}
              </Badge>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Algorithm: </span>
              <Badge variant="outline" className="ml-1">
                {folderData.metadata?.algorithm}
              </Badge>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Tokenizer: </span>
              <Badge variant="outline" className="ml-1">
                {folderData.metadata?.tokenizer}
              </Badge>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Topics: </span>
              <Badge variant="outline" className="ml-1">
                {folderData.metadata?.topicCount}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="visualization">Topic Visualization</TabsTrigger>
          <TabsTrigger value="documents">Document Analysis</TabsTrigger>
          <TabsTrigger value="coherence">Coherence Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-6">
          <OverviewDashboard folderData={folderData} />
        </TabsContent>

        <TabsContent value="visualization" className="mt-6">
          <TopicVisualization folderData={folderData} />
        </TabsContent>

        <TabsContent value="documents" className="mt-6">
          <DocumentAnalysis folderData={folderData} />
        </TabsContent>

        <TabsContent value="coherence" className="mt-6">
          <CoherenceAnalysis folderData={folderData} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default FolderDashboard;