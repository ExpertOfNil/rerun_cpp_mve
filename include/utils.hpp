#ifndef UTILS_HPP
#define UTILS_HPP

#include <string.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rerun.hpp>
#include <rerun_helpers.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

/* Build URL string from user input IP:PORT.  Use only with Rerun v0.23+ */
std::string build_url(const char* ip_str);
/* Check for image file types */
bool is_image_file(const fs::path& img_path);
/* Load a folder of images into memory */
int load_saved_images(
    std::string rr_path,
    const fs::path img_path,
    std::vector<cv::Mat>& images,
    const rerun::RecordingStream& rec
);

enum RETURN_STATUS {
    OK,
    ERROR,
    EARLY_OUT,
};

struct Cli {
    std::string path;
    bool enable_rerun;
    std::string viewer_addr;
    size_t threads;
};

/* Help text for CLI */
void help();
/* Parse user input and populate `Cli` */
std::pair<Cli, RETURN_STATUS> parseArgs(int argc, char** argv);

struct ImageBuffer {
    std::vector<cv::Mat> images;
    std::vector<fs::path> image_paths;
    uint32_t buffer_size;

    std::atomic<uint32_t> head_idx;
    std::atomic<uint32_t> tail_idx;
    std::atomic<uint32_t> path_idx;
    std::atomic<uint32_t> loaded_count;

    std::atomic<bool> shutdown;
    std::atomic<uint32_t> loader_threads_count;
    std::vector<std::thread> loader_threads;
    std::atomic<uint32_t> active_loader_idx;
};

/* Image loading process */
void background_image_loader(ImageBuffer* buf, uint32_t loader_idx);
/* Initialize image buffer */
RETURN_STATUS ImageBuffer_Init(
    ImageBuffer* buffer,
    std::string path,
    uint32_t buffer_size,
    uint32_t n_loaders
);
/* Copy image from at buffer head */
RETURN_STATUS ImageBuffer_NextImage(const ImageBuffer& buf, cv::Mat& img);
/* Make image at buffer head available for new data */
RETURN_STATUS ImageBuffer_ConsumeImage(ImageBuffer& buf);
/* Shutdown loader threads and clean up image buffer resources */
RETURN_STATUS ImageBuffer_Shutdown(ImageBuffer& buf);
/* Print information about the image buffer */
void ImageBuffer_Stats(const ImageBuffer& buf);

#endif /* UTILS_HPP */
