# NMF Analysis Web Dashboard

A comprehensive web application for visualizing and analyzing NMF (Non-negative Matrix Factorization) topic modeling results.

## Features

- **Multi-Folder Support**: Import and analyze multiple NMF analysis results
- **Folder Upload**: Upload ZIP archives containing NMF analysis folders
- **Tabbed Interface**: Switch between different analyses with browser-style tabs
- **Folder Management**: Delete folders and manage analysis collections
- **Interactive Visualizations**: 
  - Word clouds with zoom functionality
  - Topic coherence charts
  - Document distribution plots
  - Topic-word score tables
- **Document Analysis**: Search and filter documents by content and topic
- **Export Functionality**: Download Excel files and visualizations
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Quick Start

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Navigate to the webapp directory:
```bash
cd webapp
```

2. Install all dependencies:
```bash
npm run install-all
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:7070

### Production Build

```bash
npm run build
npm start
```

## Usage

1. **Automatic Folder Detection**: The app automatically scans the `Output/` directory for valid NMF analysis folders
2. **Upload New Folders**: Click the "Upload" button to upload ZIP archives containing NMF analysis folders
3. **Tab Navigation**: Click on folder tabs at the top to switch between different analyses
4. **Folder Management**: Use the menu button (⋮) next to folder names to delete folders
5. **Dashboard Views**: Each folder has four main views:
   - **Overview**: Summary statistics, coherence charts, and metadata
   - **Topic Visualization**: Word clouds and topic-word score tables
   - **Document Analysis**: Search documents and view top documents per topic
   - **Coherence Analysis**: Detailed coherence score analysis and interpretation

### Uploading Analysis Folders

1. Click the "Upload" button in the top toolbar
2. Drag and drop a ZIP file or click to select one
3. The ZIP should contain folders with NMF analysis results
4. Required files per folder:
   - `*_coherence_scores.json`
   - `*_wordcloud_scores.json`
5. Optional but recommended:
   - `wordclouds/` directory with PNG files
   - `top_docs_*.json`
   - `*.xlsx` files
   - `*_document_dist.png`

## Supported File Formats

The app automatically detects and processes these files from each analysis folder:

- `*_coherence_scores.json` - Topic coherence data
- `*_wordcloud_scores.json` - Word importance scores per topic
- `top_docs_*.json` - Top documents for each topic
- `*_document_dist.png` - Document distribution visualization
- `*.xlsx` - Excel files with topic data
- `wordclouds/*.png` - Individual topic word cloud images

## API Endpoints

- `GET /api/folders` - List all available analysis folders
- `GET /api/folder/:folderName` - Get detailed data for a specific folder
- `POST /api/upload-folder` - Upload ZIP archive containing analysis folders
- `DELETE /api/folder/:folderName` - Delete a specific analysis folder
- `GET /api/image/:folderName/*` - Serve image files
- `GET /api/excel/:folderName/:fileName` - Download Excel files

## Project Structure

```
webapp/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── App.js         # Main app component
│   │   └── index.js       # Entry point
│   └── package.json
├── server/                 # Express backend
│   └── index.js           # API server
├── package.json           # Root package.json
└── README.md
```

## Development

### Adding New Visualizations

1. Create a new component in `client/src/components/`
2. Import and add it to `FolderDashboard.js`
3. Update the backend API if needed in `server/index.js`

### Adding New File Types

1. Update the file detection logic in `server/index.js`
2. Add parsing logic for the new file format
3. Create corresponding frontend components

## Troubleshooting

### Common Issues

1. **No folders detected**: Ensure your NMF analyses have generated the required JSON files
2. **Images not loading**: Check that the Output directory structure matches the expected format
3. **Port conflicts**: Change the ports in package.json if 3000 or 5000 are in use

### Debug Mode

Set `NODE_ENV=development` for additional logging and error details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see the main project license for details.