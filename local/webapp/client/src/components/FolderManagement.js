import React, { useState } from 'react';
import { MoreVertical, Trash2, Download, Info } from 'lucide-react';
import axios from 'axios';
import { Button } from './ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from './ui/dropdown-menu';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';

function FolderManagement({ folderName, onFolderDeleted, folderData }) {
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleDeleteClick = () => {
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    setDeleting(true);
    try {
      await axios.delete(`/api/folder/${folderName}`);
      onFolderDeleted(folderName);
      setDeleteDialogOpen(false);
    } catch (error) {
      console.error('Delete failed:', error);
      // Could add error handling here
    } finally {
      setDeleting(false);
    }
  };

  const handleDownload = () => {
    // Download topics as CSV
    if (folderData?.topics) {
      window.open(folderData.topics, '_blank');
    }
  };

  const handleInfo = () => {
    // This would show folder info dialog
    console.log('Show folder info for:', folderName);
  };

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <MoreVertical className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={handleInfo}>
            <Info className="h-4 w-4 mr-2" />
            Folder Info
          </DropdownMenuItem>
          
          <DropdownMenuItem onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            Download as CSV
          </DropdownMenuItem>
          
          <DropdownMenuSeparator />
          
          <DropdownMenuItem 
            onClick={handleDeleteClick}
            className="text-destructive focus:text-destructive"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete Folder
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={deleteDialogOpen} onOpenChange={(open) => !deleting && setDeleteDialogOpen(open)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Folder</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the folder "{folderName}"? 
              This action cannot be undone and will permanently remove all analysis data.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button 
              onClick={() => setDeleteDialogOpen(false)} 
              disabled={deleting}
              variant="outline"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleDeleteConfirm} 
              disabled={deleting}
              variant="destructive"
            >
              {deleting ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default FolderManagement;