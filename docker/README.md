# Docker Layout

Planned files:

- `Dockerfile.cpu`
- `Dockerfile.gpu`
- root Compose files for common, CPU, and GPU services.

The CPU image will support preprocessing/evaluation/storage work. The GPU image will use a validated pinned NVIDIA PyTorch base for H200 training. Large datasets remain on mounted local scratch/object storage rather than inside images.

Docker Compose is the intended orchestrator for this single-host workflow; Docker Swarm is not currently planned.
