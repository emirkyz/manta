"""Output generation pipeline for MANTA topic analysis."""

from pathlib import Path
from typing import Any, Dict, Optional

from ..config import TopicAnalysisConfig
from ..utils.visualization.visualizer import create_visualization
from ..utils.export.json_to_excel import convert_json_to_excel
from ..utils.processing_utils import CachedData, PipelineContext


class OutputPipeline:
    """Handles visualization and output file generation."""

    @staticmethod
    def generate_outputs(
        cached_data: CachedData,
        nmf_output: Dict,
        word_result: Any,
        topic_word_scores: Dict,
        topic_doc_scores: Dict,
        config: TopicAnalysisConfig,
        ctx: PipelineContext,
        table_output_dir: Path,
    ) -> Any:
        """Generate visualizations and output files.

        Args:
            cached_data: Text arrays, vocab, and datetime series from the data stage
            nmf_output: W, H (and optionally S) matrices from NMF
            word_result: Per-topic word scores from topic extraction
            topic_word_scores: Formatted word scores for export
            topic_doc_scores: Formatted document scores for export
            config: Typed analysis configuration
            ctx: Pipeline context with paths, db config, console, and runtime state
            table_output_dir: Directory for writing output files

        Returns:
            Visual returns from visualization generation
        """
        table_name = ctx.paths.table_name

        if ctx.console:
            ctx.console.print_status("Generating visualizations and exports...", "processing")

        # Build a minimal options dict for the visualizer (which still uses the old interface)
        viz_options = config.to_run_options()

        visual_returns = create_visualization(
            nmf_output,
            cached_data.vocab,
            table_output_dir,
            table_name,
            viz_options,
            word_result,
            topic_word_scores,
            cached_data.text_array,
            ctx.db_config.topics_db_engine,
            ctx.emoji_map,
            ctx.db_config.program_output_dir,
            ctx.db_config.output_dir,
            datetime_series=cached_data.datetime_series,
        )

        if ctx.console:
            ctx.console.print_status("Exporting results to Excel...", "processing")

        convert_json_to_excel(
            word_json_data=topic_word_scores,
            doc_json_data=topic_doc_scores,
            output_dir=table_output_dir,
            data_frame_name=table_name,
            total_docs_count=len(cached_data.text_array),
        )

        if ctx.console:
            ctx.console.print_status("Output generation completed", "success")

        return visual_returns
