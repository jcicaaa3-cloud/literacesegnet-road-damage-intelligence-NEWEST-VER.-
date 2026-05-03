This folder is generated locally by the evidence scripts.

Expected generated structure:

final_evidence/
 01_checkpoints/
  checkpoint_manifest.md
 02_metrics_and_compare_cpu/
  model_compare_summary.csv
  model_compare_summary.json
 02_metrics_and_compare_gpu/
  model_compare_summary.csv
  model_compare_summary.json
 03_literace_overlays/
  *_service_overlay.png
  *_service_mask.png
  *_service_card.png
  *_service_boundary.png
 04_segformer_overlays/
  *_overlay.png
  *_mask_color.png
  *_pred_class.png
 05_llm_service_example/
  literace_service_batch_summary.json
  literace_service_batch_summary.csv
  literace_llm_service_example.md
 06_report_ready/
  final_comparison_table.md
  capstone_result_summary.md

Generated files are ignored by Git unless they are placeholder README/.gitkeep files.
Do not commit private dataset outputs, restricted images, checkpoints, CSV/JSON logs, or model weights.
