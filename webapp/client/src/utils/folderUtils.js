/**
 * Truncates folder name for display in tabs
 * @param {string} folderName - The folder name to truncate
 * @param {number} maxLength - Maximum length before truncation (default: 15)
 * @returns {string} - Truncated folder name
 */
export const truncateFolderName = (folderName, maxLength = 15) => {
  return folderName.length > maxLength 
    ? `${folderName.substring(0, maxLength)}...` 
    : folderName;
};

/**
 * Checks if a folder exists in the folders array
 * @param {Array} folders - Array of folder objects
 * @param {string} folderName - Name of the folder to check
 * @returns {boolean} - True if folder exists
 */
export const folderExists = (folders, folderName) => {
  return folders.some(folder => folder.name === folderName);
};

/**
 * Gets the first available folder from an array
 * @param {Array} folders - Array of folder objects
 * @returns {string|null} - Name of the first folder or null if empty
 */
export const getFirstFolder = (folders) => {
  return folders.length > 0 ? folders[0].name : null;
};