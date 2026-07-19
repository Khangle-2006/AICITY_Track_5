from __future__ import annotations


def main() -> None:
    import diffusers

    print(f"diffusers={diffusers.__version__}")
    print(f"HunyuanVideoFramepackPipeline={hasattr(diffusers, 'HunyuanVideoFramepackPipeline')}")
    print(f"HunyuanVideoFramepackTransformer3DModel={hasattr(diffusers, 'HunyuanVideoFramepackTransformer3DModel')}")


if __name__ == "__main__":
    main()
