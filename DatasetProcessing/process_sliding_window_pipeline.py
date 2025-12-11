#!/usr/bin/env python3
"""
Combined pipeline script that processes MineRL episodes through the complete sliding window pipeline:
1. Creates overlapping sliding window chunks
2. Generates embeddings using MineCLIP
3. Generates natural language descriptions for each window
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_sliding_window_processor(
    data_dir: Path,
    output_dir: Path,
    window_size: int = 16,
    stride: int = 8,
    episodes: Optional[int] = None
) -> bool:
    """Run the sliding window processor."""
    logger.info("=" * 50)
    logger.info("STEP 1: Creating sliding window chunks")
    logger.info("=" * 50)
    
    try:
        # Import here to avoid loading dependencies if not needed
        sys.path.append(str(Path(__file__).parent))
        from sliding_window_processor import SlidingWindowProcessor
        
        processor = SlidingWindowProcessor(
            data_dir=data_dir,
            output_dir=output_dir,
            window_size=window_size,
            stride=stride
        )
        
        if episodes:
            episode_dirs = processor.get_episode_directories()[:episodes]
            total_windows = 0
            for episode_dir in episode_dirs:
                try:
                    windows_created = processor.process_episode(episode_dir)
                    total_windows += windows_created
                except Exception as e:
                    logger.error(f"Error processing {episode_dir}: {e}")
                    continue
            logger.info(f"Created {total_windows} windows from {episodes} episodes")
        else:
            processor.process_all_episodes()
        
        logger.info("✓ Sliding window processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Sliding window processing failed: {e}")
        return False


def run_embedding_generation(
    data_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 16
) -> bool:
    """Run video embedding generation."""
    logger.info("=" * 50)
    logger.info("STEP 2: Generating video embeddings")
    logger.info("=" * 50)
    
    try:
        # Import here to avoid loading dependencies if not needed
        sys.path.append(str(Path(__file__).parent))
        from embed_videos import load_model, get_window_dirs, process_windows_in_batches
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        logger.info("Loading MineCLIP model...")
        model = load_model(str(checkpoint_path), device)
        logger.info("Model loaded successfully")
        
        window_dirs = get_window_dirs(data_dir)
        logger.info(f"Found {len(window_dirs)} windows to process")
        
        if not window_dirs:
            logger.warning(f"No windows found in {data_dir}")
            return True
        
        process_windows_in_batches(
            window_dirs, model, device,
            batch_size=batch_size
        )
        
        logger.info("✓ Video embedding generation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Video embedding generation failed: {e}")
        return False


def run_description_generation(
    data_dir: Path,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "auto",
    resume: bool = True,
    start_window: int = 0,
    end_window: Optional[int] = None
) -> bool:
    """Run description generation."""
    logger.info("=" * 50)
    logger.info("STEP 3: Generating window descriptions")
    logger.info("=" * 50)
    
    try:
        # Import here to avoid loading dependencies if not needed
        sys.path.append(str(Path(__file__).parent))
        from generate_descriptions import DescriptionGenerator, process_windows
        
        generator = DescriptionGenerator(model_id=model_id, device=device)
        generator.load_model()
        
        process_windows(
            data_dir=data_dir,
            generator=generator,
            resume=resume,
            start_window=start_window,
            end_window=end_window
        )
        
        logger.info("✓ Description generation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Description generation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete sliding window processing pipeline for MineRL dataset"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".data/MineRLTreechop-v0",
        help="Path to input MineRL dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sliding_window_dataset_complete",
        help="Output directory for processed dataset"
    )
    
    # Sliding window parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Number of frames per window"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Number of frames to advance between windows"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Limit number of episodes to process"
    )
    
    # Embedding parameters
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=".ckpts/attn.pth",
        help="Path to MineCLIP checkpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation"
    )
    
    # Description parameters
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model ID for the VLM"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for description generation (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="Start from this window index for description generation"
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=None,
        help="End at this window index for description generation"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip windows that already have descriptions"
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip-windowing",
        action="store_true",
        help="Skip sliding window processing step"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation step"
    )
    parser.add_argument(
        "--skip-descriptions",
        action="store_true",
        help="Skip description generation step"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    if not args.skip_embeddings and not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    logger.info("Pipeline Configuration:")
    logger.info(f"  Input directory: {input_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Window size: {args.window_size}")
    logger.info(f"  Stride: {args.stride}")
    logger.info(f"  Episodes limit: {args.episodes or 'All'}")
    logger.info(f"  MineCLIP checkpoint: {checkpoint_path}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  VLM model: {args.vlm_model}")
    logger.info(f"  Description device: {args.device}")
    logger.info(f"  Skip steps: windowing={args.skip_windowing}, embeddings={args.skip_embeddings}, descriptions={args.skip_descriptions}")
    
    start_time = time.time()
    
    # Step 1: Sliding window processing
    if not args.skip_windowing:
        success = run_sliding_window_processor(
            data_dir=input_dir,
            output_dir=output_dir,
            window_size=args.window_size,
            stride=args.stride,
            episodes=args.episodes
        )
        if not success:
            logger.error("Pipeline failed at sliding window processing step")
            sys.exit(1)
    else:
        logger.info("Skipping sliding window processing step")
    
    # Step 2: Embedding generation
    if not args.skip_embeddings:
        success = run_embedding_generation(
            data_dir=output_dir,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size
        )
        if not success:
            logger.error("Pipeline failed at embedding generation step")
            sys.exit(1)
    else:
        logger.info("Skipping embedding generation step")
    
    # Step 3: Description generation
    if not args.skip_descriptions:
        success = run_description_generation(
            data_dir=output_dir,
            model_id=args.vlm_model,
            device=args.device,
            resume=not args.no_resume,
            start_window=args.start_window,
            end_window=args.end_window
        )
        if not success:
            logger.error("Pipeline failed at description generation step")
            sys.exit(1)
    else:
        logger.info("Skipping description generation step")
    
    # Pipeline completed
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.info("=" * 50)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)
    logger.info(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info(f"Output dataset: {output_dir}")
    
    # Print final summary
    try:
        summary_path = output_dir / "dataset_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            logger.info(f"Final dataset summary:")
            logger.info(f"  Episodes processed: {summary.get('total_episodes', 'Unknown')}")
            logger.info(f"  Windows created: {summary.get('total_windows', 'Unknown')}")
            logger.info(f"  Overlap percentage: {summary.get('overlap_percentage', 'Unknown')}%")
    except Exception:
        pass


if __name__ == "__main__":
    main()