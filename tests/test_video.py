import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from vl_workbench.video import make_timed_frame_messages, sample_video


class VideoSamplingTests(unittest.TestCase):
    def test_sampled_frames_have_their_original_video_timestamps(self):
        frames = [object(), object(), object()]
        content = make_timed_frame_messages(frames, 0.4, 7.5, "找到兔子")[0]["content"]
        self.assertEqual([item["text"] for item in content if item["type"] == "text"][1:4], [
            "视频时间 0.0 秒：", "视频时间 2.5 秒：", "视频时间 5.0 秒："
        ])
        self.assertEqual([item["image"] for item in content if item["type"] == "image"], frames)
        self.assertIn("找到兔子", content[-1]["text"])

    def test_samples_evenly_without_loading_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            container = av.open(str(path), mode="w")
            stream = container.add_stream("mpeg4", rate=4)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            for index in range(32):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index * 7, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            frames, fps, duration = sample_video(path, max_frames=8)
            self.assertEqual(len(frames), 4)
            self.assertAlmostEqual(duration, 8.0, places=1)
            self.assertAlmostEqual(fps, 0.5, places=2)
            brightness = [np.asarray(frame).mean() for frame in frames]
            self.assertEqual(brightness, sorted(brightness))


if __name__ == "__main__":
    unittest.main()
