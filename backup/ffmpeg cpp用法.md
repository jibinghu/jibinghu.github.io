opencv 拉流并无损输出：

``` cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>

int main() {
    std::string input_url = "rtmp://101.126.68.169/live/mac";
    std::string output_url = "rtmp://101.126.68.169/live/processed";

    // 打开输入流
    cv::VideoCapture cap(input_url);
    if (!cap.isOpened()) {
        std::cerr << "无法打开输入流: " << input_url << std::endl;
        return -1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps    = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (fps == 0) fps = 25; // 有些流无法正确获取fps，给个默认值

    // 准备 ffmpeg 命令
    std::string ffmpeg_cmd =
        "ffmpeg -y -f rawvideo -pix_fmt bgr24 -s " +
        std::to_string(width) + "x" + std::to_string(height) +
        " -r " + std::to_string(fps) +
        " -i - -c:v libx264 -pix_fmt yuv420p -preset ultrafast -f flv " +
        output_url;

#ifdef _WIN32
    FILE* pipe = _popen(ffmpeg_cmd.c_str(), "wb");
#else
    FILE* pipe = popen(ffmpeg_cmd.c_str(), "w");
#endif

    if (!pipe) {
        std::cerr << "无法启动 ffmpeg 进程" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "读取帧失败，结束" << std::endl;
            break;
        }

        // 将帧写入 ffmpeg stdin
        fwrite(frame.data, 1, frame.total() * frame.elemSize(), pipe);
    }

#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    cap.release();
    return 0;
}
```

`g++ test.cpp -o test `pkg-config --cflags --libs opencv4`
