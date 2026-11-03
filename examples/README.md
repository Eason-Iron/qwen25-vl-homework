# 示例素材

`ocr-example.png` 是运行 `python scripts/create_ocr_example.py` 自行生成的虚构票据，可用于 OCR 和图片阅读理解；它不包含真实个人信息。

医学训练数据使用 [MedMNIST PneumoniaMNIST](https://github.com/MedMNIST/MedMNIST)，请参阅项目根目录说明。

`qwen-official-demo.mp4` 是从 [Qwen 官方模型示例](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)下载的视频，当前仅用于本地测试；由于尚未核实该视频的单独分发许可，它默认不加入仓库。

`space-clip.mp4` 是从该视频第 45–75 秒生成的本地短片，运行 `python scripts/create_video_clip.py` 可以重建；视频时间从短片开始重新计时。

`bunny-clip.mp4` 是从官方影片第 38–58 秒截取的无声片段，采样为 2 帧/秒；可从 [Blender 官方下载页](https://peach.blender.org/download/)取得原片，再运行 `python scripts/create_open_video_clip.py` 重建。© 2008 Blender Foundation / www.bigbuckbunny.org，使用 [CC BY 3.0 许可](https://peach.blender.org/about/)，本片段经过裁剪、降帧并移除音频。

`astronaut.png` 来自 scikit-image 的 [astronaut 示例](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)，照片人物是 NASA 宇航员 Eileen Collins；该来源说明照片为公共领域。

`public-figure.jpg` 是白宫官方人物照片，由 [美国国会图书馆](https://www.loc.gov/pictures/item/2025163294/)提供，作品为美国联邦政府公开内容；图像内容可与来源页面对照核实。

`scenery.jpg` 是约 1901 年的优胜美地山谷照片，来源为[美国国会图书馆](https://www.loc.gov/pictures/item/2002712508/)，无已知出版限制。

`qwen-official-image.jpg` 来自 [Qwen 官方示例](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)，用于本地海滩场景识别；因尚未核实该图片的单独分发许可，默认不加入仓库。
