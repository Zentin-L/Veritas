# Installation note

Install with Python 3.11:

```powershell
python -m pip install -r requirements.txt
```

The requirements file includes the official PyTorch CPU wheel index. This keeps the local embedding and optional reranker stack CPU-only and avoids CUDA downloads. All retrieval and indexing dependencies are open source and run locally.
