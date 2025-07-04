import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { Card, CardContent } from './ui/card';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';
import { Pagination } from './ui/pagination';

function DocumentAnalysis({ folderData }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedTopic, setSelectedTopic] = useState('all');
  const documentsPerPage = 10;

  const { topDocs } = folderData;

  if (!topDocs) {
    return (
      <Card>
        <CardContent className="p-6">
          <h3 className="text-lg font-semibold">No document data available</h3>
        </CardContent>
      </Card>
    );
  }

  const allDocuments = Object.entries(topDocs).flatMap(([topicName, docs]) =>
    Object.entries(docs).map(([docId, content]) => {
      const [text, score] = content.split(':');
      return {
        topicName,
        docId,
        text: text || content,
        score: parseFloat(score) || 0,
      };
    })
  );

  const filteredDocuments = allDocuments.filter(doc => {
    const matchesSearch = doc.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         doc.topicName.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesTopic = selectedTopic === 'all' || doc.topicName === selectedTopic;
    return matchesSearch && matchesTopic;
  });

  const paginatedDocuments = filteredDocuments.slice(
    (currentPage - 1) * documentsPerPage,
    currentPage * documentsPerPage
  );

  const totalPages = Math.ceil(filteredDocuments.length / documentsPerPage);

  const handlePageChange = (event, value) => {
    setCurrentPage(value);
  };

  const topicNames = Object.keys(topDocs);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Document Analysis</h2>

      <Card>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search documents..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setCurrentPage(1);
                }}
                className="pl-10"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge
                variant={selectedTopic === 'all' ? 'default' : 'outline'}
                className="cursor-pointer"
                onClick={() => {
                  setSelectedTopic('all');
                  setCurrentPage(1);
                }}
              >
                All Topics
              </Badge>
              {topicNames.map(topic => (
                <Badge
                  key={topic}
                  variant={selectedTopic === topic ? 'default' : 'outline'}
                  className="cursor-pointer"
                  onClick={() => {
                    setSelectedTopic(topic);
                    setCurrentPage(1);
                  }}
                >
                  {topic}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <p className="text-sm text-muted-foreground mb-4">
        Showing {filteredDocuments.length} documents
      </p>

      <div className="space-y-4">
        {paginatedDocuments.map((doc, index) => (
          <Card key={`${doc.topicName}-${doc.docId}-${index}`}>
            <CardContent className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div className="space-y-2">
                  <Badge variant="outline">
                    {doc.topicName}
                  </Badge>
                  <p className="text-sm text-muted-foreground">
                    Document ID: {doc.docId}
                  </p>
                </div>
                <Badge variant="secondary">
                  Score: {doc.score.toFixed(4)}
                </Badge>
              </div>
              <p className="text-sm leading-relaxed">
                {doc.text.length > 500 
                  ? `${doc.text.substring(0, 500)}...` 
                  : doc.text
                }
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {totalPages > 1 && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={handlePageChange}
          className="mt-6"
        />
      )}

      <div className="mt-8">
        <h3 className="text-xl font-semibold mb-4">Topic-wise Document Distribution</h3>
        <Accordion type="single" collapsible className="space-y-2">
          {topicNames.map(topicName => {
            const topicDocs = topDocs[topicName];
            const docCount = Object.keys(topicDocs).length;
            
            return (
              <AccordionItem key={topicName} value={topicName}>
                <AccordionTrigger className="text-left">
                  <span className="font-medium">
                    {topicName} ({docCount} documents)
                  </span>
                </AccordionTrigger>
                <AccordionContent>
                  <div className="space-y-3">
                    {Object.entries(topicDocs).slice(0, 5).map(([docId, content]) => {
                      const [text, score] = content.split(':');
                      return (
                        <Card key={docId} variant="outline">
                          <CardContent className="p-4">
                            <div className="flex justify-between items-start mb-2">
                              <p className="text-sm text-muted-foreground">
                                Document {docId}
                              </p>
                              <Badge variant="outline" className="text-xs">
                                Score: {parseFloat(score || 0).toFixed(4)}
                              </Badge>
                            </div>
                            <p className="text-sm">
                              {(text || content).substring(0, 300)}
                              {(text || content).length > 300 && '...'}
                            </p>
                          </CardContent>
                        </Card>
                      );
                    })}
                    {docCount > 5 && (
                      <p className="text-sm text-muted-foreground text-center py-2">
                        ... and {docCount - 5} more documents
                      </p>
                    )}
                  </div>
                </AccordionContent>
              </AccordionItem>
            );
          })}
        </Accordion>
      </div>
    </div>
  );
}

export default DocumentAnalysis;