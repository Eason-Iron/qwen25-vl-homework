"""Create a short silent teaching clip from the official Big Buck Bunny release."""

from pathlib import Path

import av


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "downloads" / "BigBuckBunny_320x180.mp4"
DESTINATION = ROOT / "examples" / "bunny-clip.mp4"


def main():
    start, duration, fps = 38.0, 20.0, 2
    with av.open(str(SOURCE)) as input_file, av.open(str(DESTINATION), mode="w") as output_file:
        input_stream = input_file.streams.video[0]
        output_stream = output_file.add_stream("libx264", rate=fps)
        output_stream.width, output_stream.height = 320, 180
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"preset": "fast", "crf": "23"}
        input_file.seek(int(start / float(input_stream.time_base)), stream=input_stream, backward=True)
        next_time = start
        for frame in input_file.decode(input_stream):
            if frame.time is None or frame.time < next_time:
                continue
            if frame.time >= start + duration:
                break
            clean = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
            for packet in output_stream.encode(clean):
                output_file.mux(packet)
            next_time += 1 / fps
        for packet in output_stream.encode():
            output_file.mux(packet)
    print(DESTINATION)


if __name__ == "__main__":
    main()
