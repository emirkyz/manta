const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs-extra');
const multer = require('multer');
const XLSX = require('xlsx');
const AdmZip = require('adm-zip');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 7070;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../client/build')));

const outputPath = path.join(__dirname, '../../Output');
const uploadsPath = path.join(__dirname, '../uploads');

// Ensure uploads directory exists
fs.ensureDirSync(uploadsPath);

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsPath);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB limit
  },
  fileFilter: (req, file, cb) => {
    // Accept ZIP files and common archive formats
    const allowedTypes = ['.zip', '.rar', '.7z'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Only ZIP archives are allowed'), false);
    }
  }
});

app.get('/api/folders', async (req, res) => {
  try {
    console.log(`[${new Date().toISOString()}] GET /api/folders - Reading from: ${outputPath}`);
    
    if (!await fs.pathExists(outputPath)) {
      console.error(`[${new Date().toISOString()}] Output path does not exist: ${outputPath}`);
      return res.status(404).json({ error: 'Output directory not found' });
    }
    
    const folders = await fs.readdir(outputPath);
    console.log(`[${new Date().toISOString()}] Found ${folders.length} items in output directory`);
    
    const validFolders = [];
    
    for (const folder of folders) {
      const folderPath = path.join(outputPath, folder);
      const stat = await fs.stat(folderPath);
      
      if (stat.isDirectory() && !folder.startsWith('.')) {
        const files = await fs.readdir(folderPath);
        const hasRequiredFiles = files.some(file => 
          file.includes('wordcloud_scores.json') || 
          file.includes('coherence_scores.json')
        );
        
        if (hasRequiredFiles) {
          validFolders.push({
            name: folder,
            path: folderPath,
            files: files
          });
        } else {
          console.log(`[${new Date().toISOString()}] Skipping folder ${folder} - missing required files`);
        }
      }
    }
    
    console.log(`[${new Date().toISOString()}] Returning ${validFolders.length} valid folders`);
    res.json(validFolders);
  } catch (error) {
    console.error(`[${new Date().toISOString()}] Error reading folders:`, error);
    res.status(500).json({ 
      error: 'Failed to read output folders',
      details: error.message 
    });
  }
});

app.get('/api/folder/:folderName', async (req, res) => {
  try {
    const { folderName } = req.params;
    const folderPath = path.join(outputPath, folderName);
    
    if (!await fs.pathExists(folderPath)) {
      return res.status(404).json({ error: 'Folder not found' });
    }
    
    const files = await fs.readdir(folderPath);
    const folderData = {
      name: folderName,
      coherenceScores: null,
      wordcloudScores: null,
      topDocs: null,
      documentDist: null,
      topics: null,
      wordclouds: []
    };
    
    for (const file of files) {
      const filePath = path.join(folderPath, file);
      
      if (file.includes('coherence_scores.json')) {
        folderData.coherenceScores = await fs.readJson(filePath);
      } else if (file.includes('wordcloud_scores.json')) {
        folderData.wordcloudScores = await fs.readJson(filePath);
      } else if (file.includes('top_docs_')) {
        folderData.topDocs = await fs.readJson(filePath);
      } else if (file.includes('document_dist.png')) {
        folderData.documentDist = `/api/image/${folderName}/${file}`;
      } else if (file.includes('topics.xlsx')) {
        folderData.topics = `/api/csv/${folderName}/${file}`;
      } else if (file === 'wordclouds') {
        const wordcloudPath = path.join(folderPath, 'wordclouds');
        if (await fs.pathExists(wordcloudPath)) {
          const wordcloudFiles = await fs.readdir(wordcloudPath);
          folderData.wordclouds = wordcloudFiles
            .filter(f => f.endsWith('.png'))
            .map(f => ({
              name: f,
              url: `/api/image/${folderName}/wordclouds/${f}`
            }));
        }
      }
    }
    
    const metadata = extractMetadata(folderName);
    folderData.metadata = metadata;
    
    res.json(folderData);
  } catch (error) {
    console.error('Error reading folder data:', error);
    res.status(500).json({ error: 'Failed to read folder data' });
  }
});

app.get('/api/image/:folderName/*', (req, res) => {
  const folderName = req.params.folderName;
  const imagePath = req.params[0];
  const fullPath = path.join(outputPath, folderName, imagePath);
  
  if (fs.existsSync(fullPath)) {
    res.sendFile(fullPath);
  } else {
    res.status(404).json({ error: 'Image not found' });
  }
});

app.get('/api/csv/:folderName/:fileName', async (req, res) => {
  try {
    const { folderName, fileName } = req.params;
    const folderPath = path.join(outputPath, folderName);
    
    // Check if folder exists
    if (!await fs.pathExists(folderPath)) {
      return res.status(404).json({ error: 'Folder not found' });
    }
    
    // Try to find wordcloud scores JSON file to convert to CSV
    const files = await fs.readdir(folderPath);
    const wordcloudScoresFile = files.find(file => file.includes('wordcloud_scores.json'));
    
    if (!wordcloudScoresFile) {
      return res.status(404).json({ error: 'No topic data available for CSV export' });
    }
    
    // Read the wordcloud scores JSON
    const wordcloudScoresPath = path.join(folderPath, wordcloudScoresFile);
    const wordcloudData = await fs.readJson(wordcloudScoresPath);
    
    // Convert JSON to CSV format
    let csvContent = 'Topic,Word,Score\n';
    
    Object.entries(wordcloudData).forEach(([topicName, words]) => {
      Object.entries(words).forEach(([word, score]) => {
        // Escape commas and quotes in data
        const escapedWord = word.replace(/"/g, '""');
        const escapedTopic = topicName.replace(/"/g, '""');
        csvContent += `"${escapedTopic}","${escapedWord}",${score}\n`;
      });
    });
    
    // Generate CSV filename
    const csvFileName = fileName.replace('.xlsx', '.csv').replace(/\.json$/, '.csv');
    
    // Set proper headers for CSV download
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="${csvFileName}"`);
    res.setHeader('Cache-Control', 'no-cache');
    
    // Send CSV content
    res.send(csvContent);
    
  } catch (error) {
    console.error('CSV generation error:', error);
    res.status(500).json({ error: 'Failed to generate CSV file' });
  }
});

// Utility function to recursively clean macOS artifacts
function cleanMacOSArtifacts(dirPath) {
  try {
    const items = fs.readdirSync(dirPath);
    
    for (const item of items) {
      const itemPath = path.join(dirPath, item);
      
      // Remove macOS artifacts
      if (item === '__MACOSX' || item.startsWith('.DS_Store') || item.startsWith('._')) {
        console.log(`[${new Date().toISOString()}] Removing macOS artifact: ${itemPath}`);
        fs.removeSync(itemPath);
        continue;
      }
      
      // Recursively clean subdirectories
      if (fs.statSync(itemPath).isDirectory()) {
        cleanMacOSArtifacts(itemPath);
      }
    }
  } catch (error) {
    console.warn(`[${new Date().toISOString()}] Warning: Could not clean macOS artifacts in ${dirPath}: ${error.message}`);
  }
}

// Validation functions
function validateNMFFolder(folderPath) {
  const requiredFiles = [
    'wordcloud_scores.json',
    'coherence_scores.json'
  ];
  
  const optionalFiles = [
    'top_docs_',
    'document_dist.png',
    'topics.xlsx'
  ];

  const errors = [];
  const warnings = [];

  try {
    // Clean macOS artifacts recursively before validation
    cleanMacOSArtifacts(folderPath);
    
    const files = fs.readdirSync(folderPath);
    
    // Filter out system files
    const filteredFiles = files.filter(file => 
      !file.startsWith('.DS_Store') && 
      !file.startsWith('._') && 
      !file.startsWith('~$')
    );
    
    // Check for required files
    for (const required of requiredFiles) {
      const hasFile = filteredFiles.some(file => file.includes(required));
      if (!hasFile) {
        errors.push(`Missing required file: ${required}`);
      }
    }
    
    // Check for wordclouds directory
    const wordcloudsPath = path.join(folderPath, 'wordclouds');
    if (!fs.existsSync(wordcloudsPath)) {
      warnings.push('Missing wordclouds directory');
    } else {
      const wordcloudFiles = fs.readdirSync(wordcloudsPath);
      const pngFiles = wordcloudFiles.filter(f => f.endsWith('.png'));
      if (pngFiles.length === 0) {
        warnings.push('No PNG files found in wordclouds directory');
      }
    }
    
    // Validate JSON files (more lenient approach)
    for (const file of filteredFiles) {
      if (file.endsWith('.json')) {
        try {
          const filePath = path.join(folderPath, file);
          let content = fs.readFileSync(filePath, 'utf8');
          
          // Remove common comment patterns that might be in JSON files
          content = content.replace(/\s*=\s*\d+/g, ''); // Remove "= 0" style comments
          content = content.replace(/\/\/.*$/gm, ''); // Remove // comments
          content = content.replace(/\/\*[\s\S]*?\*\//g, ''); // Remove /* */ comments
          
          JSON.parse(content);
        } catch (jsonError) {
          // Try reading as-is first, then warn if it fails
          try {
            fs.readJsonSync(filePath);
          } catch (secondError) {
            warnings.push(`JSON file may have formatting issues: ${file}`);
          }
        }
      }
    }
    
    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      fileCount: filteredFiles.length
    };
  } catch (error) {
    return {
      isValid: false,
      errors: [`Failed to read folder: ${error.message}`],
      warnings: [],
      fileCount: 0
    };
  }
}

function generateUniqueFolderName(baseName) {
  let counter = 1;
  let folderName = baseName;
  
  while (fs.existsSync(path.join(outputPath, folderName))) {
    folderName = `${baseName}_${counter}`;
    counter++;
  }
  
  return folderName;
}

function extractMetadata(folderName) {
  const parts = folderName.split('_');
  return {
    dataset: parts[0] || 'Unknown',
    algorithm: parts.find(p => p === 'nmf' || p === 'pnmf' || p === 'opnmf') || 'Unknown',
    tokenizer: parts.find(p => p === 'bpe' || p === 'wordpiece' || p === 'w') || 'Unknown',
    topicCount: parts.find(p => /^\d+$/.test(p)) || 'Unknown'
  };
}

// Upload endpoint
app.post('/api/upload-folder', upload.single('folder'), async (req, res) => {
  let tempExtractPath = null;
  
  try {
    console.log(`[${new Date().toISOString()}] POST /api/upload-folder - Upload request received`);
    
    if (!req.file) {
      console.error(`[${new Date().toISOString()}] No file uploaded`);
      return res.status(400).json({ error: 'No file uploaded' });
    }

    console.log(`[${new Date().toISOString()}] File uploaded: ${req.file.filename}, size: ${req.file.size} bytes`);

    const uploadedFile = req.file;
    const tempId = uuidv4();
    tempExtractPath = path.join(uploadsPath, `temp_${tempId}`);
    
    // Extract ZIP file
    console.log('Extracting ZIP file:', uploadedFile.filename);
    const zip = new AdmZip(uploadedFile.path);
    zip.extractAllTo(tempExtractPath, true);
    
    // Find folders in extracted content
    const extractedItems = await fs.readdir(tempExtractPath);
    const folders = [];
    
    for (const item of extractedItems) {
      // Skip macOS-created files and folders
      if (item === '__MACOSX' || item.startsWith('.DS_Store') || item.startsWith('._')) {
        console.log(`[${new Date().toISOString()}] Skipping macOS artifact: ${item}`);
        continue;
      }
      
      const itemPath = path.join(tempExtractPath, item);
      const stat = await fs.stat(itemPath);
      
      if (stat.isDirectory()) {
        folders.push({ name: item, path: itemPath });
      }
    }
    
    if (folders.length === 0) {
      return res.status(400).json({ 
        error: 'No folders found in the uploaded archive'
      });
    }
    
    const results = [];
    
    // Process each folder
    for (const folder of folders) {
      const validation = validateNMFFolder(folder.path);
      
      if (validation.isValid) {
        // Generate unique folder name
        const uniqueName = generateUniqueFolderName(folder.name);
        const destPath = path.join(outputPath, uniqueName);
        
        // Copy folder to Output directory
        await fs.copy(folder.path, destPath);
        
        results.push({
          name: uniqueName,
          originalName: folder.name,
          status: 'success',
          warnings: validation.warnings,
          fileCount: validation.fileCount
        });
      } else {
        results.push({
          name: folder.name,
          status: 'error',
          errors: validation.errors,
          warnings: validation.warnings
        });
      }
    }
    
    // Clean up
    await fs.remove(uploadedFile.path);
    await fs.remove(tempExtractPath);
    
    const successCount = results.filter(r => r.status === 'success').length;
    const errorCount = results.filter(r => r.status === 'error').length;
    
    res.json({
      message: `Upload completed: ${successCount} successful, ${errorCount} failed`,
      results: results,
      successCount,
      errorCount
    });
    
  } catch (error) {
    console.error('Upload error:', error);
    
    // Clean up on error
    if (req.file) {
      await fs.remove(req.file.path).catch(() => {});
    }
    if (tempExtractPath) {
      await fs.remove(tempExtractPath).catch(() => {});
    }
    
    res.status(500).json({ 
      error: 'Upload failed: ' + error.message 
    });
  }
});

// Delete folder endpoint
app.delete('/api/folder/:folderName', async (req, res) => {
  try {
    const { folderName } = req.params;
    const folderPath = path.join(outputPath, folderName);
    
    if (!await fs.pathExists(folderPath)) {
      return res.status(404).json({ error: 'Folder not found' });
    }
    
    await fs.remove(folderPath);
    res.json({ message: 'Folder deleted successfully' });
  } catch (error) {
    console.error('Delete error:', error);
    res.status(500).json({ error: 'Failed to delete folder' });
  }
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../client/build/index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Output directory: ${outputPath}`);
});