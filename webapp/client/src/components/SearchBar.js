import React from 'react';
import { Search, X } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';

const SearchBar = ({ searchTerm, setSearchTerm, clearSearch }) => {
  return (
    <div className="mb-4">
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search analysis folders..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10 pr-10"
        />
        {searchTerm && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearSearch}
            className="absolute right-1 top-1/2 transform -translate-y-1/2 h-7 w-7 p-0"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
};

export default SearchBar;