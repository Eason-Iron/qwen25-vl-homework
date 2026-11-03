from pathlib import Path

import av


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "examples" / "qwen-official-demo.mp4"
    destination = ROOT / "examples" / "space-clip.mp4"
    start, duration, fps = 45.0, 30.0, 2
    with av.open(str(source)) as input_file, av.open(str(destination), mode="w") as output_file:
        input_stream = input_file.streams.video[0]
        output_stream = output_file.add_stream("libx264", rate=fps)
        output_stream.width, output_stream.height = 640, 360
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"preset": "fast", "crf": "23"}
        input_file.seek(int(start / float(input_stream.time_base)), stream=input_stream, backward=True)
        next_time = start
        for frame in input_file.decode(input_stream):
            if frame.time is None or frame.time < next_time:
                continue
            if frame.time >= start + duration:
                break
            # A fresh frame lets the encoder assign timestamps in the output time base.
            clean_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
            clean_frame = clean_frame.reformat(width=640, height=360, format="yuv420p")
            for packet in output_stream.encode(clean_frame):
                output_file.mux(packet)
            next_time += 1 / fps
        for packet in output_stream.encode():
            output_file.mux(packet)
    print(destination)


if __name__ == "__main__":
    main()
