# pyright: reportUnusedImport=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import os
import sys

def check_gguf_support():
    try:
        from ltx_core.text_encoders import gemma
        import inspect
        src = inspect.getsource(gemma.load_text_encoder_and_tokenizer)  # type: ignore
        if "gguf" in src.lower():
            print("GGUF support found in load_text_encoder_and_tokenizer!")
            return True
        # Check DistilledPipeline
        from ltx_pipelines.distilled import DistilledPipeline
        init_src = inspect.getsource(DistilledPipeline.__init__)
        if "gguf" in init_src.lower():
            print("GGUF support found in DistilledPipeline init!")
            return True
        print("No explicit GGUF support strings found, but it might use transformers natively.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

check_gguf_support()
