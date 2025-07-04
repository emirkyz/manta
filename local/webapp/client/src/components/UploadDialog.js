import React, { useState, useCallback } from 'react';
import { CloudUpload, CheckCircle, AlertCircle, AlertTriangle, X } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from './ui/dialog';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';

function UploadDialog({ open, onClose, onUploadSuccess }) {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResults, setUploadResults] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploading(true);
    setUploadProgress(0);
    setError(null);
    setUploadResults(null);

    try {
      const formData = new FormData();
      formData.append('folder', file);

      const response = await axios.post('/api/upload-folder', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
      });

      setUploadResults(response.data);
      if (response.data.successCount > 0) {
        onUploadSuccess(response.data);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [onUploadSuccess]);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'application/zip': ['.zip'],
      'application/x-rar-compressed': ['.rar'],
      'application/x-7z-compressed': ['.7z'],
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024, // 100MB
    disabled: uploading,
  });

  const handleClose = () => {
    if (!uploading) {
      setUploadResults(null);
      setError(null);
      setUploadProgress(0);
      onClose();
    }
  };

  const renderResults = () => {
    if (!uploadResults) return null;

    return (
      <div className="mt-4 space-y-4">
        <div className={`p-4 rounded-lg border ${
          uploadResults.errorCount > 0 
            ? 'bg-yellow-50 border-yellow-200 dark:bg-yellow-950 dark:border-yellow-800' 
            : 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800'
        }`}>
          <div className="flex items-center gap-2">
            {uploadResults.errorCount > 0 ? (
              <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
            ) : (
              <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
            )}
            <span className={`text-sm font-medium ${
              uploadResults.errorCount > 0 
                ? 'text-yellow-800 dark:text-yellow-200' 
                : 'text-green-800 dark:text-green-200'
            }`}>
              {uploadResults.message}
            </span>
          </div>
        </div>

        <div className="space-y-3">
          {uploadResults.results.map((result, index) => (
            <div key={index} className="border rounded-lg p-4">
              <div className="flex items-start gap-3">
                <div className="mt-0.5">
                  {result.status === 'success' ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <AlertCircle className="h-4 w-4 text-red-500" />
                  )}
                </div>
                <div className="flex-1">
                  <div className="font-medium text-sm">
                    {result.originalName || result.name}
                  </div>
                  {result.status === 'success' && (
                    <div className="mt-1">
                      <div className="text-sm text-green-600 dark:text-green-400">
                        Successfully imported as: {result.name}
                      </div>
                      {result.warnings && result.warnings.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {result.warnings.map((warning, i) => (
                            <Badge key={i} variant="outline" className="text-xs text-yellow-600">
                              {warning}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  {result.status === 'error' && (
                    <div className="mt-1">
                      {result.errors && result.errors.map((error, i) => (
                        <div key={i} className="text-sm text-red-600 dark:text-red-400">
                          • {error}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Upload NMF Analysis Folder</DialogTitle>
        </DialogHeader>

        {!uploading && !uploadResults && !error && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Upload a ZIP archive containing NMF analysis folders. Each folder should contain:
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm">*_coherence_scores.json (required)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm">*_wordcloud_scores.json (required)</span>
              </div>
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                <span className="text-sm">wordclouds/ directory with PNG files (recommended)</span>
              </div>
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                <span className="text-sm">top_docs_*.json, *.xlsx, *_document_dist.png (optional)</span>
              </div>
            </div>
          </div>
        )}

        {!uploading && !uploadResults && (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive 
                ? 'border-primary bg-primary/5' 
                : 'border-muted-foreground/25 hover:border-muted-foreground/50 hover:bg-muted/50'
            }`}
          >
            <input {...getInputProps()} />
            <CloudUpload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-lg font-semibold mb-2">
              {isDragActive
                ? 'Drop the ZIP file here...'
                : 'Drag & drop a ZIP file here, or click to select'}
            </h3>
            <p className="text-sm text-muted-foreground">
              Supported formats: ZIP, RAR, 7Z (max 100MB)
            </p>
          </div>
        )}

        {fileRejections.length > 0 && (
          <div className="p-4 rounded-lg border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
              <span className="text-sm font-medium text-red-800 dark:text-red-200">
                {fileRejections[0].errors[0].message}
              </span>
            </div>
          </div>
        )}

        {uploading && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Uploading and processing...</span>
              <span>{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} className="w-full" />
          </div>
        )}

        {error && (
          <div className="p-4 rounded-lg border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
              <span className="text-sm font-medium text-red-800 dark:text-red-200">
                {error}
              </span>
            </div>
          </div>
        )}

        {renderResults()}

        <DialogFooter>
          <Button onClick={handleClose} disabled={uploading} variant="outline">
            {uploadResults ? 'Close' : 'Cancel'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default UploadDialog;