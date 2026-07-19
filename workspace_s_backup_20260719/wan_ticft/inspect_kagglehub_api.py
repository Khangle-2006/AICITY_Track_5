import inspect
import kagglehub

print("version", getattr(kagglehub, "__version__", "?"))
print([x for x in dir(kagglehub) if "dataset" in x.lower() or "file" in x.lower()])
print("dataset_download", inspect.signature(kagglehub.dataset_download))
print("datasets module", [x for x in dir(kagglehub.datasets) if not x.startswith("_")][:80])
