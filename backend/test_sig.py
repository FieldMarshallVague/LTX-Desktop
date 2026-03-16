import inspect
try:
    from ltx_core.loader.module_ops import ModuleOps
    print("ModuleOps:", inspect.signature(ModuleOps))
except Exception as e:
    print("Error:", e)
