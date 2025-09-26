"""
Output generation pipeline for MANTA topic analysis.
"""

from typing import Dict, Any, Optional

from ..utils.visualization.visualizer import create_visualization
from ..utils.export.json_to_excel import convert_json_to_excel
from ..utils.console.console_manager import ConsoleManager

class OutputPipeline:
    """Handles visualization and output file generation."""
    
    @staticmethod
    def generate_outputs(
        nmf_output, 
        vocab, 
        table_output_dir, 
        table_name: str, 
        options: Dict[str, Any], 
        word_result, 
        topic_word_scores, 
        text_array, 
        topics_db_eng, 
        program_output_dir, 
        output_dir, 
        topic_doc_scores, 
        console: Optional[ConsoleManager] = None
    ):
        """
        Generate visualizations and output files.
        
        Args:
            console: Console manager for status messages
        
        Returns:
            Visual returns from visualization generation
        """
        if console:
            console.print_status("Generating visualizations and exports...", "processing")
        else:
            print("Generating visual outputs.")
            
        visual_returns = create_visualization(
            nmf_output["W"],
            nmf_output["H"],
            vocab,
            table_output_dir,
            table_name,
            options,
            word_result,
            topic_word_scores,
            text_array,
            topics_db_eng,
            options["emoji_map"],
            program_output_dir,
            output_dir,
        )

        save_to_excel = True
        if save_to_excel:
            if console:
                console.print_status("Exporting results to Excel...", "processing")
            # Save jsons to excel format
            convert_json_to_excel(
                word_json_data=topic_word_scores,
                doc_json_data=topic_doc_scores,
                output_dir=table_output_dir,
                data_frame_name=table_name,
                total_docs_count=len(text_array),
            )


        





        if console:
            console.print_status("Output generation completed", "success")
        
        return visual_returns