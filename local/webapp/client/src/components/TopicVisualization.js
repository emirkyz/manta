import React, { useState } from 'react';
import { ZoomIn } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

function TopicVisualization({ folderData }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const { wordcloudScores, wordclouds } = folderData;

  const handleImageClick = (imageUrl, topicName) => {
    setSelectedImage({ url: imageUrl, title: topicName });
  };

  const handleCloseDialog = () => {
    setSelectedImage(null);
  };

  if (!wordclouds || wordclouds.length === 0) {
    return (
      <Card>
        <CardContent className="p-6">
          <h3 className="text-lg font-semibold">No word clouds available</h3>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Topic Word Clouds</h2>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {wordclouds.map((wordcloud, index) => {
          const topicKey = `Konu ${index.toString().padStart(2, '0')}`;
          const topicData = wordcloudScores?.[topicKey] || wordcloudScores?.[`Konu ${index}`];
          
          return (
            <Card key={wordcloud.name} className="overflow-hidden">
              <div className="relative">
                <img
                  src={wordcloud.url}
                  alt={`Topic ${index}`}
                  className="w-full h-48 object-cover cursor-pointer transition-transform hover:scale-105"
                  onClick={() => handleImageClick(wordcloud.url, `Topic ${index}`)}
                />
                <Button
                  size="icon"
                  variant="secondary"
                  className="absolute top-2 right-2 h-8 w-8 bg-white/80 hover:bg-white/90"
                  onClick={() => handleImageClick(wordcloud.url, `Topic ${index}`)}
                >
                  <ZoomIn className="h-4 w-4" />
                </Button>
              </div>
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold mb-2">Topic {index}</h3>
                {topicData && (
                  <div>
                    <p className="text-sm text-muted-foreground mb-2">Top Words:</p>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(topicData)
                        .slice(0, 5)
                        .map(([word, score]) => (
                          <Badge
                            key={word}
                            variant="outline"
                            className="text-xs"
                          >
                            {word} ({score.toFixed(2)})
                          </Badge>
                        ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {wordcloudScores && (
        <div>
          <h2 className="text-2xl font-bold mb-4">Topic-Word Scores</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(wordcloudScores).map(([topicName, words]) => (
              <Card key={topicName}>
                <CardHeader>
                  <CardTitle className="text-lg">{topicName}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-sm font-medium border-b pb-2">
                      <div>Word</div>
                      <div className="text-right">Score</div>
                    </div>
                    {Object.entries(words)
                      .slice(0, 10)
                      .map(([word, score]) => (
                        <div key={word} className="grid grid-cols-2 gap-2 text-sm">
                          <div>{word}</div>
                          <div className="text-right text-muted-foreground">{score.toFixed(4)}</div>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      <Dialog open={!!selectedImage} onOpenChange={(open) => !open && handleCloseDialog()}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>{selectedImage?.title}</DialogTitle>
          </DialogHeader>
          {selectedImage && (
            <img
              src={selectedImage.url}
              alt={selectedImage.title}
              className="w-full h-auto rounded-lg"
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default TopicVisualization;