Put trained model checkpoint here if you have one.

Expected path:
 seg/runs/literace_boundary_degradation/best.pth

If best.pth exists, run_batch_infer_service.bat uses real model inference.
If best.pth does not exist, the service automatically uses CV demo mode for UI/flow demonstration only.
