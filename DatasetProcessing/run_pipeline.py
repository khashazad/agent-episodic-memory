#!/usr/bin/env python3
"""
6-Step Dataset Processing Pipeline Orchestrator

Processes MineRL episodes through the complete pipeline:
1. Creating sliding window chunks
2. Embedding videos using MineCLIP encoder
3. Generating LLM-derived descriptions (Qwen VLM)
4. Encoding descriptions using MineCLIP text encoder
5. Generating LLM-derived actions (Qwen VLM)
6. Creating CSV with fused embeddings for vectorDB

Usage:
    python run_pipeline.py --input-dir .data/MineRLTreechop-v0 --output-dir .data/pipeline_output

To skip specific steps:
    python run_pipeline.py --skip-step1 --skip-step2  # Skip windowing and video embedding
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(args):
    """Run the complete 6-step pipeline."""

    start_time = time.time()

    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)
    output_csv = Path(args.output_csv)

    if not args.skip_step1 and not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False

    if not args.skip_step2 and not args.skip_step4 and not checkpoint_path.exists():
        logger.error(f"MineCLIP checkpoint not found: {checkpoint_path}")
        return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    logger.info("=" * 60)
    logger.info("6-STEP DATASET PROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"MineCLIP checkpoint: {checkpoint_path}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"Window size: {args.window_size}")
    logger.info(f"Stride: {args.stride}")
    logger.info(f"VLM model: {args.vlm_model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Skip steps: 1={args.skip_step1}, 2={args.skip_step2}, 3={args.skip_step3}, "
                f"4={args.skip_step4}, 5={args.skip_step5}, 6={args.skip_step6}")
    logger.info("=" * 60)

    # Import pipeline steps
    from pipeline.step1_sliding_window import run_step1
    from pipeline.step2_video_embedding import run_step2
    from pipeline.step3_generate_descriptions import run_step3
    from pipeline.step4_text_embedding import run_step4
    from pipeline.step5_generate_actions import run_step5
    from pipeline.step6_export_csv import run_step6

    # Step 1: Sliding window chunking
    if not args.skip_step1:
        success = run_step1(
            input_dir=input_dir,
            output_dir=output_dir,
            window_size=args.window_size,
            stride=args.stride,
            max_episodes=args.max_episodes,
            force_recompute=args.force_recompute
        )
        if not success:
            logger.error("Pipeline failed at Step 1: Sliding window chunking")
            return False
    else:
        logger.info("Skipping Step 1: Sliding window chunking")

    # Step 2: Video embedding
    if not args.skip_step2:
        success = run_step2(
            data_dir=output_dir,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
            force_recompute=args.force_recompute
        )
        if not success:
            logger.error("Pipeline failed at Step 2: Video embedding")
            return False
    else:
        logger.info("Skipping Step 2: Video embedding")

    # Step 3: Generate descriptions
    if not args.skip_step3:
        success = run_step3(
            data_dir=output_dir,
            model_id=args.vlm_model,
            device=args.device,
            resume=not args.no_resume,
            start_window=args.start_window,
            end_window=args.end_window
        )
        if not success:
            logger.error("Pipeline failed at Step 3: Generate descriptions")
            return False
    else:
        logger.info("Skipping Step 3: Generate descriptions")

    # Step 4: Text embedding
    if not args.skip_step4:
        success = run_step4(
            data_dir=output_dir,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
            force_recompute=args.force_recompute
        )
        if not success:
            logger.error("Pipeline failed at Step 4: Text embedding")
            return False
    else:
        logger.info("Skipping Step 4: Text embedding")

    # Step 5: Generate actions
    if not args.skip_step5:
        success = run_step5(
            data_dir=output_dir,
            model_id=args.vlm_model,
            device=args.device,
            resume=not args.no_resume,
            start_window=args.start_window,
            end_window=args.end_window,
            use_descriptions=True
        )
        if not success:
            logger.error("Pipeline failed at Step 5: Generate actions")
            return False
    else:
        logger.info("Skipping Step 5: Generate actions")

    # Step 6: Export CSV
    if not args.skip_step6:
        success = run_step6(
            data_dir=output_dir,
            output_csv=output_csv,
            resume=not args.no_resume,
            batch_size=args.csv_batch_size
        )
        if not success:
            logger.error("Pipeline failed at Step 6: Export CSV")
            return False
    else:
        logger.info("Skipping Step 6: Export CSV")

    # Pipeline completed
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output CSV: {output_csv}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="6-Step Dataset Processing Pipeline for MineRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. Sliding Window Chunking - Create overlapping windows from episodes
  2. Video Embedding - Encode windows using MineCLIP video encoder
  3. Generate Descriptions - Use Qwen VLM to describe each window
  4. Text Embedding - Encode descriptions using MineCLIP text encoder
  5. Generate Actions - Use Qwen VLM to predict next best actions
  6. Export CSV - Create fused embeddings CSV for vectorDB

Examples:
  # Run full pipeline
  python run_pipeline.py --input-dir .data/MineRLTreechop-v0

  # Process only first 5 episodes
  python run_pipeline.py --input-dir .data/MineRLTreechop-v0 --max-episodes 5

  # Skip VLM steps (steps 3 and 5)
  python run_pipeline.py --skip-step3 --skip-step5

  # Resume interrupted pipeline
  python run_pipeline.py --skip-step1 --skip-step2
"""
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
        default=".data/pipeline_output",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=".data/vectordb_data.csv",
        help="Output CSV file path"
    )

    # Sliding window parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Number of frames per window (default: 16)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Number of frames to advance between windows (default: 8)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes to process"
    )

    # MineCLIP parameters
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

    # VLM parameters
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID for Qwen VLM"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for VLM (auto, cuda, cpu)"
    )

    # Window range for VLM steps
    parser.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="Start from this window index for VLM steps"
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=None,
        help="End at this window index for VLM steps"
    )

    # CSV parameters
    parser.add_argument(
        "--csv-batch-size",
        type=int,
        default=1000,
        help="Batch size for CSV writing"
    )

    # Pipeline control
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="Skip Step 1: Sliding window chunking"
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="Skip Step 2: Video embedding"
    )
    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="Skip Step 3: Generate descriptions"
    )
    parser.add_argument(
        "--skip-step4",
        action="store_true",
        help="Skip Step 4: Text embedding"
    )
    parser.add_argument(
        "--skip-step5",
        action="store_true",
        help="Skip Step 5: Generate actions"
    )
    parser.add_argument(
        "--skip-step6",
        action="store_true",
        help="Skip Step 6: Export CSV"
    )

    # Resume/recompute options
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already processed windows"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute embeddings even if they exist"
    )

    args = parser.parse_args()

    success = run_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
