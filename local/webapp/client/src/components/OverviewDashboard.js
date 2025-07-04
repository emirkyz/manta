import React from 'react';
import { Download, TrendingUp, Hash } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

function OverviewDashboard({ folderData }) {
  const { coherenceScores, wordcloudScores, topDocs, metadata } = folderData;

  const coherenceData = coherenceScores?.topic_coherences 
    ? Object.entries(coherenceScores.topic_coherences).map(([topic, score]) => ({
        topic: topic.replace('_coherence', ''),
        score: score
      }))
    : [];

  const topicCount = wordcloudScores ? Object.keys(wordcloudScores).length : 0;
  
  const totalDocuments = topDocs 
    ? Object.values(topDocs).reduce((sum, topic) => sum + Object.keys(topic).length, 0)
    : 0;

  const avgCoherence = coherenceScores?.average_coherence || 0;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="flex items-center p-6">
            <Hash className="h-8 w-8 text-primary mr-4" />
            <div>
              <div className="text-2xl font-bold">{topicCount}</div>
              <p className="text-sm text-muted-foreground">Topics</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center p-6">
            <TrendingUp className="h-8 w-8 text-green-500 mr-4" />
            <div>
              <div className="text-2xl font-bold">{avgCoherence.toFixed(3)}</div>
              <p className="text-sm text-muted-foreground">Avg Coherence</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center p-6">
            <Download className="h-8 w-8 text-blue-500 mr-4" />
            <div>
              <div className="text-2xl font-bold">{totalDocuments}</div>
              <p className="text-sm text-muted-foreground">Documents</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Badge variant="outline" className="text-xs">
              Algorithm: {metadata?.algorithm}
            </Badge>
            <Badge variant="outline" className="text-xs">
              Tokenizer: {metadata?.tokenizer}
            </Badge>
            <Badge variant="outline" className="text-xs">
              Dataset: {metadata?.dataset}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {coherenceData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Topic Coherence Scores</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={coherenceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="topic" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="score" fill="hsl(var(--primary))" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {folderData.documentDist && (
        <Card>
          <CardHeader>
            <CardTitle>Document Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center">
              <img 
                src={folderData.documentDist} 
                alt="Document Distribution"
                className="max-w-full h-auto rounded-lg"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {folderData.topics && (
        <Card>
          <CardHeader>
            <CardTitle>Download Files</CardTitle>
          </CardHeader>
          <CardContent>
            <Button
              className="flex items-center space-x-2"
              asChild
            >
              <a href={folderData.topics} download>
                <Download className="h-4 w-4" />
                <span>Download Excel Topics</span>
              </a>
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default OverviewDashboard;