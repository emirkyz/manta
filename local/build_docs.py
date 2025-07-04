#!/usr/bin/env python3
"""
Build script for Sphinx documentation.

This script builds the Sphinx documentation and provides helpful output.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Build the Sphinx documentation."""
    print("Building NMF Standalone Documentation with Sphinx")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent
    docs_dir = project_root / "docs"
    output_dir = docs_dir / "_build" / "html"
    
    print(f"Project root: {project_root}")
    print(f"Docs directory: {docs_dir}")
    print(f"Output directory: {output_dir}")
    
    if not docs_dir.exists():
        print("Error: docs/ directory not found!")
        sys.exit(1)
    
    if not (docs_dir / "conf.py").exists():
        print("Error: conf.py not found in docs/ directory!")
        sys.exit(1)
    
    # First, regenerate API documentation
    print("\nRegenerating API documentation...")
    try:
        api_result = subprocess.run([
            "uv", "run", "sphinx-apidoc",
            "-o", str(docs_dir),
            str(project_root),
            "--separate", "--force"
        ], cwd=project_root, capture_output=True, text=True)
        
        if api_result.returncode == 0:
            print("✓ API documentation regenerated")
        else:
            print("⚠ API documentation generation had issues, continuing...")
    except Exception as e:
        print(f"⚠ API documentation generation failed: {e}, continuing...")
    
    # Build documentation
    print("\nBuilding Sphinx documentation...")
    try:
        result = subprocess.run([
            "uv", "run", "sphinx-build",
            "-b", "html",
            "-E",  # Don't use cached environment
            str(docs_dir),
            str(output_dir)
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Documentation built successfully!")
            print(f"\nDocumentation available at: file://{output_dir}/index.html")
            
            # Check if we can open the docs
            index_file = output_dir / "index.html"
            if index_file.exists():
                print(f"\nTo view the documentation:")
                print(f"  open {index_file}")
                print(f"  # Or visit: file://{index_file}")
                
                # Count generated files
                html_files = list(output_dir.rglob("*.html"))
                print(f"\nGenerated {len(html_files)} HTML files with modern Sphinx theme")
        else:
            print("✗ Documentation build failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            print("\nNote: Some warnings are expected for duplicate references")
            sys.exit(1)
            
    except FileNotFoundError:
        print("Error: sphinx-build not found. Make sure Sphinx is installed:")
        print("  uv add sphinx sphinx-rtd-theme sphinx-autodoc-typehints")
        sys.exit(1)
    except Exception as e:
        print(f"Error building documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()