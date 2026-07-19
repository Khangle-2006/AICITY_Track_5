from runpy import run_path

globals().update(
    {
        key: value
        for key, value in run_path(
            "/workspace/wan_ticft/evaluation/epona_one_frame_config.py"
        ).items()
        if not key.startswith("__")
    }
)
test_video_frames = 4
del run_path
