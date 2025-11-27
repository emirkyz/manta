# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - N-gram Implementation

### Added
- **N-gram Support**: Enhanced NMF and NMTF functions with flexible n-gram processing capabilities
- **t-SNE Visualization**: Optimized t-SNE visualization module for large datasets with PCA preprocessing and multi-threading support
- **LDAvis Integration**: Support for pyLDAvis-style interactive topic visualizations via `create_manta_ldavis` function

### Changed
- **CSR Format Support**: Updated NMF and NMTF functions to use Compressed Sparse Row (CSR) format for improved performance and memory efficiency
- **Enhanced S Matrix**: Improved S matrix support in NMF and NMTF functions for better topic relationship modeling
- **Topic Extraction**: Updated `extract_topics` function to support CSR format and optional S matrix parameter
- **Informative Logging**: Added comprehensive logging throughout NMF and NMTF processing pipeline

### Improved
- **Performance**: Significant performance improvements through CSR sparse matrix format adoption
- **Memory Efficiency**: Reduced memory footprint for large-scale topic modeling tasks
- **Topic Quality**: Enhanced topic extraction algorithms for more coherent results

## [0.3.5] - 2025-01-XX

### Changed
- **BREAKING**: Package renamed from "nmf-standalone" to "MANTA" (Multi-lingual Advanced NMF-based Topic Analysis)
- CLI command changed from `nmf-standalone` to `manta`
- Python import changed from `nmf_standalone` to `manta`
- Updated all documentation and examples to reflect new branding
- Updated GitHub repository URLs and references

## [0.1.6] - 2024-12-XX

### Fixed
- Resolved PyPI version conflict by incrementing version number
- Package republication after initial PyPI upload issue

## [0.1.0] - 2024-12-XX

### Added
- Initial release of MANTA package
- Support for Turkish and English text processing
- Multiple NMF algorithm variants (standard NMF and OPNMF)
- Advanced tokenization for Turkish (BPE and WordPiece)
- Traditional preprocessing for English with optional lemmatization
- Word cloud generation and topic visualization
- Excel export functionality
- Topic coherence scoring
- Command-line interface with comprehensive options
- Python API for programmatic access
- Comprehensive documentation and examples
- SQLite database storage for results
- Topic distribution plotting
- Word co-occurrence analysis
- Emoji processing for Turkish texts

### Features
- **Multi-language Support**: Native Turkish and English processing
- **Advanced NMF**: Standard and projective NMF implementations
- **Modern Tokenization**: BPE/WordPiece for Turkish, traditional for English
- **Rich Visualizations**: Word clouds, distribution plots, heatmaps
- **Multiple Exports**: Excel, JSON, database storage
- **CLI and API**: Both command-line and programmatic interfaces
- **Evaluation Metrics**: Built-in coherence scoring

## [Unreleased]

### Planned
- Additional language support
- More NMF algorithm variants
- Enhanced visualization options
- Performance optimizations
- Extended documentation