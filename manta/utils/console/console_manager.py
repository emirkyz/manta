"""
Console Manager for MANTA Topic Analysis

Provides enhanced console output with rich formatting, progress bars, and structured displays.
"""

import time
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.columns import Columns
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ConsoleManager:
    """
    Manages enhanced console output for MANTA topic analysis.
    
    Provides beautiful, structured console output with progress tracking,
    configuration displays, and professional status messages.
    """
    
    def __init__(self, use_rich: bool = False):
        """
        Initialize the console manager.
        
        Args:
            use_rich: Whether to use Rich library for enhanced output.
                     Falls back to plain text if Rich is unavailable.
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        self.progress = None
        self.current_tasks = {}
        self.start_time = None
        self.stage_times = {}
        
    def print_header(self, title: str, subtitle: Optional[str] = None):
        """Print a formatted header."""
        if self.use_rich:
            header_text = Text(title, style="bold blue")
            if subtitle:
                header_text.append(f"\n{subtitle}", style="dim")
            
            panel = Panel(
                header_text,
                box=box.DOUBLE,
                padding=(1, 2),
                style="blue"
            )
            self.console.print(panel)
        else:
            print(f"\n{title}")
            if subtitle:
                print(f"{subtitle}")
            print()
            
    def display_config(self, options: Dict[str, Any], filepath: str, column: str, table_name: str):
        """
        Display analysis configuration in a structured format.
        
        Args:
            options: Configuration options dictionary
            filepath: Input file path
            column: Column name being analyzed
            table_name: Output table/directory name
        """
        if self.use_rich:
            self._display_config_rich(options, filepath, column, table_name)
        else:
            self._display_config_plain(options, filepath, column, table_name)
    
    def _display_config_rich(self, options: Dict[str, Any], filepath: str, column: str, table_name: str):
        """Display configuration using Rich formatting."""
        # Create main configuration table
        config_table = Table(title="Analysis Configuration", box=box.ROUNDED)
        config_table.add_column("Setting", style="cyan", no_wrap=True, width=25)
        config_table.add_column("Value", style="white", width=35)
        config_table.add_column("Description", style="dim", width=30)
        
        # Input settings
        config_table.add_row(
            "Input File", 
            str(filepath).split('/')[-1], 
            "Source data file"
        )
        config_table.add_row(
            "Text Column", 
            str(column), 
            "Column containing text to analyze"
        )
        config_table.add_row(
            "Output Name", 
            str(table_name), 
            "Name for output files and directories"
        )
        
        # Processing settings
        config_table.add_row("", "", "")  # Separator
        config_table.add_row(
            "Language", 
            f"[bold]{options.get('LANGUAGE', 'EN')}[/bold]", 
            "Text language (TR/EN)"
        )
        config_table.add_row(
            "Topics", 
            f"[bold green]{options.get('DESIRED_TOPIC_COUNT', 5)}[/bold green]", 
            "Number of topics to extract"
        )
        config_table.add_row(
            "Words per Topic", 
            str(options.get('N_TOPICS', 15)), 
            "Top words shown per topic"
        )
        
        if options.get('LANGUAGE') == 'TR':
            config_table.add_row(
                "Tokenizer", 
                f"[yellow]{options.get('tokenizer_type', 'bpe').upper()}[/yellow]", 
                "Turkish tokenization method"
            )
        elif options.get('LANGUAGE') == 'EN':
            config_table.add_row(
                "Lemmatization", 
                "[green]Yes[/green]" if options.get('LEMMATIZE') else "[red]No[/red]", 
                "Reduce words to base forms"
            )
        
        config_table.add_row(
            "NMF Method", 
            f"[magenta]{options.get('nmf_type', 'nmf').upper()}[/magenta]", 
            "Matrix factorization algorithm"
        )
        
        # Output settings
        config_table.add_row("", "", "")  # Separator
        config_table.add_row(
            "Word Clouds",
            "[green]Yes[/green]" if options.get('gen_cloud') else "[red]No[/red]",
            "Generate topic word clouds"
        )
        config_table.add_row(
            "Excel Export",
            "[green]Yes[/green]" if options.get('save_excel') else "[red]No[/red]",
            "Export results to Excel"
        )
        config_table.add_row(
            "Topic Distribution",
            "[green]Yes[/green]" if options.get('gen_topic_distribution') else "[red]No[/red]",
            "Create distribution plots"
        )
        config_table.add_row(
            "Database Storage",
            "[green]Yes[/green]" if options.get('save_to_db') else "[red]No[/red]",
            "Save to database"
        )
        config_table.add_row(
            "Emoji Processing",
            "[green]Yes[/green]" if options.get('emoji_map') else "[red]No[/red]",
            "Process emojis in text"
        )
        
        # Filter settings if applicable
        if options.get('filter_app'):
            filter_opts = options.get('data_filter_options', {})
            config_table.add_row("", "", "")  # Separator
            if filter_opts.get('filter_app_name'):
                config_table.add_row(
                    "App Filter", 
                    f"[cyan]{filter_opts['filter_app_name']}[/cyan]", 
                    f"Column: {filter_opts.get('filter_app_column', 'N/A')}"
                )
            if filter_opts.get('filter_app_country'):
                config_table.add_row(
                    "Country Filter", 
                    f"[cyan]{filter_opts['filter_app_country']}[/cyan]", 
                    f"Column: {filter_opts.get('filter_app_country_column', 'N/A')}"
                )
        
        self.console.print(config_table)
        self.console.print()
        
    def _display_config_plain(self, options: Dict[str, Any], filepath: str, column: str, table_name: str):
        """Display configuration using plain text formatting."""
        print("\nAnalysis Configuration")
        print(f"Input File: {str(filepath).split('/')[-1]}")
        print(f"Text Column: {column}")
        print(f"Output Name: {table_name}")
        print()
        print(f"Language: {options.get('LANGUAGE', 'EN')}")
        print(f"Topics: {options.get('DESIRED_TOPIC_COUNT', 5)}")
        print(f"Words per Topic: {options.get('N_TOPICS', 15)}")
        
        if options.get('LANGUAGE') == 'TR':
            print(f"Tokenizer: {options.get('tokenizer_type', 'bpe').upper()}")
        elif options.get('LANGUAGE') == 'EN':
            print(f"Lemmatization: {'Enabled' if options.get('LEMMATIZE') else 'Disabled'}")
            
        print(f"NMF Method: {options.get('nmf_type', 'nmf').upper()}")
        print()
        print("Output Options:")
        print(f"  Word Clouds: {'Enabled' if options.get('gen_cloud') else 'Disabled'}")
        print(f"  Excel Export: {'Enabled' if options.get('save_excel') else 'Disabled'}")
        print(f"  Distribution Plots: {'Enabled' if options.get('gen_topic_distribution') else 'Disabled'}")
        print(f"  Database Storage: {'Enabled' if options.get('save_to_db') else 'Disabled'}")
        print(f"  Emoji Processing: {'Enabled' if options.get('emoji_map') else 'Disabled'}")
        print()
        print("Analysis Options:")
        n_grams = options.get('n_grams_to_discover')
        if n_grams == "auto":
            print(f"  N-gram Discovery: Auto (k={options.get('ngram_auto_k', 0.5)})")
        elif n_grams:
            print(f"  N-gram Discovery: {n_grams} n-grams")
        else:
            print(f"  N-gram Discovery: Disabled")
        print(f"  Keep Numbers: {'Enabled' if options.get('keep_numbers') else 'Disabled'}")
        if options.get('keep_numbers'):
            print(f"  PMI Scoring: {'Enabled' if options.get('use_pmi', True) else 'Disabled'}")
        print()
        
    @contextmanager
    def progress_context(self, title: str = "Processing..."):
        """
        Context manager for progress tracking.
        
        Args:
            title: Title for the progress display
            
        Yields:
            ConsoleManager instance for adding tasks
        """
        if self.use_rich:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=self.console,
                transient=False
            ) as progress:
                self.progress = progress
                self.current_tasks = {}
                yield self
                self.progress = None
        else:
            print(f"\n{title}")
            self.progress = None
            self.current_tasks = {}
            yield self
            
    def add_task(self, description: str, total: Optional[int] = None) -> TaskID:
        """
        Add a new progress task.
        
        Args:
            description: Task description
            total: Total number of steps (None for spinner)
            
        Returns:
            Task ID for updating progress
        """
        if self.use_rich and self.progress:
            task_id = self.progress.add_task(description, total=total)
            self.current_tasks[description] = task_id
            return task_id
        else:
            print(f"  {description}...")
            return description
            
    def update_task(self, task_id: Union[TaskID, str], advance: int = 1, description: Optional[str] = None):
        """
        Update task progress.
        
        Args:
            task_id: Task ID or description
            advance: Number of steps to advance
            description: New task description (optional)
        """
        if self.use_rich and self.progress and isinstance(task_id, int):
            update_kwargs = {"advance": advance}
            if description:
                update_kwargs["description"] = description
            self.progress.update(task_id, **update_kwargs)
        elif not self.use_rich and description:
            print(f"  {description}...")
            
    def complete_task(self, task_id: Union[TaskID, str], description: Optional[str] = None):
        """
        Complete a task.
        
        Args:
            task_id: Task ID or description
            description: Completion message (optional)
        """
        if self.use_rich and self.progress and isinstance(task_id, int):
            if description:
                self.progress.update(task_id, description=f"[green][OK][/green] {description}")
            else:
                # Find original description and mark as complete
                for desc, tid in self.current_tasks.items():
                    if tid == task_id:
                        self.progress.update(task_id, description=f"[green][OK][/green] {desc}")
                        break
        elif not self.use_rich:
            if description:
                print(f"  {description} - Complete")
                
    def print_status(self, message: str, status: str = "info", timestamp: bool = True):
        """
        Print a status message with formatting.
        
        Args:
            message: Status message
            status: Message type (info, success, warning, error)
            timestamp: Whether to include timestamp
        """
        if timestamp:
            current_time = time.strftime("%H:%M:%S")
            time_prefix = f"[{current_time}] " if not self.use_rich else f"[dim][{current_time}][/dim] "
        else:
            time_prefix = ""
            
        if self.use_rich:
            icons = {
                "info": "[INFO]",
                "success": "[OK]",
                "warning": "[WARN]",
                "error": "[ERR]",
                "processing": ""
            }

            styles = {
                "info": "blue",
                "success": "green",
                "warning": "yellow",
                "error": "red",
                "processing": "cyan"
            }

            icon = icons.get(status, "")
            style = styles.get(status, "white")

            self.console.print(f"{time_prefix}[{style}]{icon}[/{style}] {message}")
        else:
            prefixes = {
                "info": "INFO:",
                "success": "SUCCESS:",
                "warning": "WARNING:", 
                "error": "ERROR:",
                "processing": "PROCESSING:"
            }
            prefix = prefixes.get(status, "")
            print(f"{time_prefix}{prefix} {message}" if prefix else f"{time_prefix}{message}")

    def print_warning(self, message: str, tag: Optional[str] = None, timestamp: bool = True):
        """
        Print a warning message.

        Args:
            message: Warning message
            tag: Optional tag prefix (e.g., "DATA LOADING", "PREPROCESSING")
            timestamp: Whether to include timestamp
        """
        formatted_msg = f"[{tag}] {message}" if tag else message
        self.print_status(formatted_msg, status="warning", timestamp=timestamp)

    def print_error(self, message: str, tag: Optional[str] = None, timestamp: bool = True):
        """
        Print an error message.

        Args:
            message: Error message
            tag: Optional tag prefix
            timestamp: Whether to include timestamp
        """
        formatted_msg = f"[{tag}] {message}" if tag else message
        self.print_status(formatted_msg, status="error", timestamp=timestamp)

    def print_debug(self, message: str, tag: Optional[str] = None, timestamp: bool = False):
        """
        Print a debug/tracking message. These are detailed process tracking messages.

        Args:
            message: Debug message
            tag: Optional tag prefix (e.g., "DATA LOADING", "YEAR FILTER")
            timestamp: Whether to include timestamp (default False for debug)
        """
        formatted_msg = f"[{tag}] {message}" if tag else message
        if self.use_rich:
            self.console.print(f"[dim]{formatted_msg}[/dim]")
        else:
            print(formatted_msg)

    def print_timing_summary(self, stage_times: Dict[str, float], total_time: float):
        """
        Print timing summary for analysis stages.
        
        Args:
            stage_times: Dictionary mapping stage names to execution times
            total_time: Total execution time
        """
        if self.use_rich:
            timing_table = Table(title="Timing Summary", box=box.SIMPLE)
            timing_table.add_column("Stage", style="cyan")
            timing_table.add_column("Time", style="green", justify="right")
            timing_table.add_column("Percentage", style="dim", justify="right")
            
            for stage, time_taken in stage_times.items():
                percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
                timing_table.add_row(
                    stage,
                    f"{time_taken:.2f}s",
                    f"{percentage:.1f}%"
                )
            
            timing_table.add_row("", "", "")  # Separator
            timing_table.add_row(
                "[bold]Total Time[/bold]",
                f"[bold green]{total_time:.2f}s[/bold green]",
                "[bold]100.0%[/bold]"
            )
            
            self.console.print(timing_table)
        else:
            print("\nTiming Summary")
            for stage, time_taken in stage_times.items():
                percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
                print(f"{stage}: {time_taken:.2f}s ({percentage:.1f}%)")
            print(f"Total Time: {total_time:.2f}s (100.0%)")
            
    def print_analysis_summary(self, result: Dict[str, Any], stage_times: Dict[str, float], total_time: float):
        """
        Print comprehensive analysis summary.
        
        Args:
            result: Analysis result dictionary
            stage_times: Timing information for each stage
            total_time: Total analysis time
        """
        if result["state"] == "SUCCESS":
            if self.use_rich:
                self._print_success_summary_rich(result, stage_times, total_time)
            else:
                self._print_success_summary_plain(result, stage_times, total_time)
        else:
            self.print_status(f"Analysis failed: {result['message']}", "error")
            
    def _print_success_summary_rich(self, result: Dict[str, Any], stage_times: Dict[str, float], total_time: float):
        """Print success summary using Rich formatting."""
        # Main success panel
        success_text = Text("Analysis Completed Successfully!", style="bold green", justify="center")
        success_panel = Panel(
            success_text,
            box=box.DOUBLE,
            style="green",
            padding=(1, 2)
        )
        self.console.print(success_panel)

        # Results summary table
        summary_table = Table(title="Results Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Dataset", result.get('data_name', 'Unknown'))
        
        topic_count = len(result.get('topic_word_scores', {}))
        summary_table.add_row("Topics Found", f"[bold green]{topic_count}[/bold green]")
        
        if result.get('coherence_scores'):
            # Handle nested coherence score structure
            coherence_data = result['coherence_scores']
            avg_coherence = None
            
            # Try to extract average coherence from different possible structures
            if isinstance(coherence_data, dict):
                if 'gensim' in coherence_data and 'c_v_average' in coherence_data['gensim']:
                    avg_coherence = coherence_data['gensim']['c_v_average']
                elif 'class_based' in coherence_data and 'average_coherence' in coherence_data['class_based']:
                    avg_coherence = coherence_data['class_based']['average_coherence']
                
            if avg_coherence is not None:
                summary_table.add_row("Avg. Coherence", f"{avg_coherence:.4f}")
        
        # Output files information
        summary_table.add_row("", "")  # Separator
        summary_table.add_row("Output Location", f"Output/{result.get('data_name', 'Unknown')}/")
        
        generated_files = []
        if result.get('topic_word_scores'):
            generated_files.append("Topic-word scores (JSON/Excel)")
        if result.get('topic_doc_scores'):
            generated_files.append("Document-topic scores")
        if result.get('topic_dist_img'):
            generated_files.append("Topic distribution plot")
        if result.get('coherence_scores'):
            generated_files.append("Coherence scores")
            
        for i, file_desc in enumerate(generated_files):
            label = "Generated Files" if i == 0 else ""
            summary_table.add_row(label, file_desc)
            
        self.console.print(summary_table)
        
        # Timing summary
        self.print_timing_summary(stage_times, total_time)
        
    def _print_success_summary_plain(self, result: Dict[str, Any], stage_times: Dict[str, float], total_time: float):
        """Print success summary using plain text."""
        print("\nAnalysis Completed Successfully!")
        print(f"Dataset: {result.get('data_name', 'Unknown')}")
        
        topic_count = len(result.get('topic_word_scores', {}))
        print(f"Topics Found: {topic_count}")
        
        if result.get('coherence_scores'):
            # Handle nested coherence score structure  
            coherence_data = result['coherence_scores']
            avg_coherence = None
            
            # Try to extract average coherence from different possible structures
            if isinstance(coherence_data, dict):
                if 'gensim' in coherence_data and 'c_v_average' in coherence_data['gensim']:
                    avg_coherence = coherence_data['gensim']['c_v_average']
                elif 'class_based' in coherence_data and 'average_coherence' in coherence_data['class_based']:
                    avg_coherence = coherence_data['class_based']['average_coherence']
                    
            if avg_coherence is not None:
                print(f"Average Coherence: {avg_coherence:.4f}")
            
        print(f"Results saved in: Output/{result.get('data_name', 'Unknown')}/")
        
        print("\nGenerated Files:")
        if result.get('topic_word_scores'):
            print("  Topic-word scores (JSON/Excel)")
        if result.get('topic_doc_scores'):
            print("  Document-topic scores")  
        if result.get('topic_dist_img'):
            print("  Topic distribution plot")
        if result.get('coherence_scores'):
            print("  Coherence scores")
            
        self.print_timing_summary(stage_times, total_time)
        
    def start_timing(self):
        """Start timing the analysis."""
        self.start_time = time.time()
        self.stage_times = {}
        
    def record_stage_time(self, stage_name: str, start_time: float):
        """Record the time taken for a specific stage."""
        self.stage_times[stage_name] = time.time() - start_time
        
    def get_total_time(self) -> float:
        """Get total elapsed time since start_timing() was called."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
        
    def print_section_header(self, title: str, icon: str = ""):
        """Print a section header."""
        if self.use_rich:
            header_text = f"{icon} {title}" if icon else title
            header = Text(header_text, style="bold cyan")
            self.console.print(Panel(header, style="cyan", box=box.SIMPLE))
        else:
            print(f"\n{title}")
            print()


# Global console manager instance management
_global_console: Optional[ConsoleManager] = None


def get_console(use_rich: bool = False) -> ConsoleManager:
    """
    Get the global ConsoleManager singleton.
    Creates one if it doesn't exist.

    Args:
        use_rich: Whether to use Rich library for enhanced output (default: False)
    """
    global _global_console
    if _global_console is None:
        _global_console = ConsoleManager(use_rich=use_rich)
    return _global_console


def set_console(console: ConsoleManager) -> None:
    """
    Set the global ConsoleManager instance.
    Useful for testing or custom configurations.
    """
    global _global_console
    _global_console = console


# Backwards compatibility - expose global instance
console_manager = get_console()