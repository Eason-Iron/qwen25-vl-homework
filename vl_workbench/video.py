import math

import av


def make_timed_frame_messages(frames, sample_fps, duration, prompt):
    if not frames or sample_fps <= 0 or duration <= 0:
        raise ValueError("视频帧或时间参数无效。")
    content = [{"type": "text", "text": f"以下图片是从同一段 {duration:.1f} 秒的视频中抽取的帧。每张图前标注原视频时间。"}]
    for index, frame in enumerate(frames):
        content.extend([
            {"type": "text", "text": f"视频时间 {index / sample_fps:.1f} 秒："},
            {"type": "image", "image": frame, "max_pixels": 128 * 28 * 28},
        ])
    content.append({"type": "text", "text": prompt + "\n请根据标注的时间估计事件发生的时间，不要超出视频总时长。"})
    return [{"role": "user", "content": content}]


def sample_video(path, max_frames=16, desired_fps=0.5):
    if max_frames < 2 or desired_fps <= 0:
        raise ValueError("视频抽帧参数无效。")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("文件中没有视频轨道。")
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            raise ValueError("无法确定视频时长，请使用常见的 MP4 视频。")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("视频时长无效。")

        frame_count = min(max_frames, max(4, int(duration * desired_fps)))
        frame_count -= frame_count % 2
        frames = []
        start_pts = stream.start_time or 0
        for index in range(frame_count):
            target_pts = start_pts + int((index * duration / frame_count) / float(stream.time_base))
            container.seek(target_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts >= target_pts:
                    image = frame.to_image().convert("RGB")
                    image.thumbnail((640, 640))
                    frames.append(image)
                    break
            else:
                raise ValueError("读取视频帧失败，请尝试其他视频。")
    return frames, frame_count / duration, duration
