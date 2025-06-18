#include "utils.hpp"

#include <fstream>
#include <sstream>

void help() {
    printf(
        "Usage: measure [OPTIONS]\n"
        "  -h, --help        Show help text\n"
        "  --enable_rerun    {true, false}. Default is true.\n"
        "  --viewer_addr     IP:PORT for rerun viewer. Default is "
        "127.0.0.1:9876.\n"
        "  --threads         Number of image loader threads. Default is 3.\n"
        //"  --path        Path to images directory\n"
    );
}

std::pair<Cli, RETURN_STATUS> parseArgs(int argc, char** argv) {
    std::stringstream log_ss;
    Cli cli = {};
    cli.threads = 3;
    cli.enable_rerun = true;
    cli.viewer_addr = "127.0.0.1:9876";
    cli.path = "";
    for (size_t i = 1; i < (size_t)argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            help();
            return std::pair(cli, EARLY_OUT);
        }
        // if (std::string(argv[i]) == "--path" && i + 1 < (size_t)argc) {
        //     cli.path = argv[i + 1];
        //     printf("CLI OPTION SET: TMD Path = %s", cli.path.c_str());
        //     continue;
        // }
        if (std::string(argv[i]) == "--enable_rerun" && i + 1 < (size_t)argc) {
            std::string rerun_str = argv[i + 1];
            for (auto& c : rerun_str) {
                c = tolower(c);
            }
            if (rerun_str == "false" || rerun_str == "0") {
                cli.enable_rerun = false;
            }
            printf(
                "CLI OPTION SET: Rerun enabled = %s\n",
                cli.enable_rerun ? "true" : "false"
            );
            continue;
        }
        if (std::string(argv[i]) == "--viewer_addr" && i + 1 < (size_t)argc) {
            cli.viewer_addr = argv[i + 1];
            printf(
                "CLI OPTION SET: Rerun viewer IP:PORT = %s\n",
                cli.viewer_addr.c_str()
            );
            continue;
        }
        if (std::string(argv[i]) == "--threads" && i + 1 < (size_t)argc) {
            cli.threads = atoi(argv[i + 1]);
            printf(
                "CLI OPTION SET: Rerun viewer IP:PORT = %s\n", cli.path.c_str()
            );
            continue;
        }
    }
    return std::pair(cli, OK);
}

void background_image_loader(ImageBuffer* buf, uint32_t loader_idx) {
    std::stringstream fname;
    fname << "loader_" << loader_idx << ".csv";
    std::fstream loader_file(
        fname.str(), std::ios::in | std::ios::out | std::ios::trunc
    );
    loader_file << "path_name,path_idx,tail_idx,loaded_count\n";
    while (!buf->shutdown) {
        uint32_t loaded_count = buf->loaded_count.load();
        uint32_t active_loader_idx = buf->active_loader_idx.load();
        if (loaded_count < buf->buffer_size &&
            active_loader_idx == loader_idx) {
            uint32_t tail_idx = buf->tail_idx.load();
            uint32_t path_idx = buf->path_idx.load();
            uint32_t loaded_count = buf->path_idx.load();

            fs::path image_path = buf->image_paths[buf->path_idx];
            buf->images[tail_idx] = cv::imread(image_path);
            buf->path_idx.store((path_idx + 1) % buf->image_paths.size());
            buf->tail_idx.store((tail_idx + 1) % buf->buffer_size);
            buf->loaded_count.fetch_add(1);
            loader_file << image_path << "," << path_idx << "," << tail_idx
                        << "," << loaded_count << "\n";
        }
    }
}

RETURN_STATUS ImageBuffer_Init(
    ImageBuffer* buffer,
    std::string path,
    uint32_t buffer_size,
    uint32_t n_loaders
) {
    fs::path image_dir = path;
    if (!fs::is_directory(image_dir)) {
        fprintf(stderr, "Path is not a directory: %s\n", path.c_str());
        return ERROR;
    }

    buffer->image_paths.clear();
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (fs::is_regular_file(entry) && entry.path().extension() == ".png") {
            buffer->image_paths.push_back(entry.path());
        }
    }

    if (buffer_size > buffer->image_paths.size()) {
        buffer_size = buffer->image_paths.size();
        fprintf(
            stderr,
            "WARN: Specified buffer size larger than number of images.  "
            "Reducing buffer size to %d\n",
            buffer_size
        );
    }
    std::sort(buffer->image_paths.begin(), buffer->image_paths.end());

    buffer->images.resize(buffer_size);
    buffer->buffer_size = buffer_size;
    std::vector<std::chrono::high_resolution_clock::time_point[2]>
        avg_load_times(buffer->buffer_size);
    /* Pre-load buffer with images */
    for (size_t i = 0; i < buffer->buffer_size; ++i) {
        avg_load_times[i][0] = std::chrono::high_resolution_clock::now();
        buffer->images[i] = cv::imread(buffer->image_paths[i]);
        avg_load_times[i][1] = std::chrono::high_resolution_clock::now();
    }
    double sum = 0.0;
    for (size_t i = 0; i < buffer->buffer_size; ++i) {
        auto duration = avg_load_times[i][1] - avg_load_times[i][0];
        sum += std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count();
    }
    printf("Avg load time: %10.4fms\n", sum / buffer->buffer_size);
    buffer->tail_idx.store(0);
    buffer->loaded_count.store(buffer_size);
    buffer->path_idx.store(buffer_size);
    buffer->head_idx.store(0);
    buffer->shutdown.store(false);

    buffer->active_loader_idx.store(0);
    buffer->loader_threads_count.store(n_loaders);
    for (uint32_t i = 0; i < n_loaders; ++i) {
        buffer->loader_threads.emplace_back(
            std::thread(background_image_loader, buffer, i)
        );
    }
    return OK;
}

RETURN_STATUS ImageBuffer_NextImage(const ImageBuffer& buf, cv::Mat& img) {
    uint32_t loaded_count = buf.loaded_count.load();
    if (loaded_count == 0) {
        fprintf(stderr, "Attempt to get image from empty buffer\n");
        return ERROR;
    }

    uint32_t head_idx = buf.head_idx.load();
    cv::Mat src = buf.images[head_idx];
    if (src.size != img.size && src.type() != img.type()) {
        fprintf(stderr, "Image shape mismatch\n");
        return ERROR;
    }
    memcpy(img.data, src.data, src.total() * src.elemSize());
    return OK;
}

RETURN_STATUS ImageBuffer_ConsumeImage(ImageBuffer& buf) {
    uint32_t loaded_count = buf.loaded_count.load();
    uint32_t active_loader_idx = buf.active_loader_idx.load();
    if (loaded_count > 0) {
        uint32_t head_idx = buf.head_idx.load();
        buf.head_idx.store((head_idx + 1) % buf.buffer_size);
        buf.loaded_count.fetch_sub(1);
        buf.active_loader_idx.store(
            (active_loader_idx + 1) % buf.loader_threads_count
        );
    } else {
        fprintf(stderr, "Attempt to get image from empty buffer\n");
        return ERROR;
    }
    return OK;
}

RETURN_STATUS ImageBuffer_Shutdown(ImageBuffer& buf) {
    buf.shutdown.store(true);
    for (uint32_t i = 0; i < buf.loader_threads.size(); ++i) {
        if (buf.loader_threads[i].joinable()) {
            buf.loader_threads[i].join();
        }
    }
    return OK;
}

void ImageBuffer_Stats(const ImageBuffer& buf) {
    printf("Image buffer size   : %d\n", buf.buffer_size);
    printf("Image paths count   : %ld\n", buf.image_paths.size());
    printf("Head index          : %d\n", buf.head_idx.load());
    printf("Tail index          : %d\n", buf.tail_idx.load());
    printf("Current index       : %d\n", buf.path_idx.load());
    printf("Loaded images count : %d\n\n", buf.loaded_count.load());
}

// Quick and dirty sanitizing
std::string build_url(const char* ip_str) {
    char ip_addr[19] = {0};
    strncpy(ip_addr, ip_str, 18);
    char buf[38] = {0};
    snprintf(buf, sizeof(buf), "rerun+http://%s/proxy", ip_addr);
    return std::string(buf);
}

bool is_image_file(const fs::path& img_path) {
    static const std::vector<std::string> extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif"
    };
    std::string ext = img_path.extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

int load_saved_images(
    std::string rr_path,
    const fs::path img_path,
    std::vector<cv::Mat>& images,
    const rerun::RecordingStream& rec
) {
    rr_path += "/load_saved_images";
    std::stringstream log_ss;
    if (fs::is_regular_file(img_path)) {
        if (is_image_file(img_path)) {
            cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            images.push_back(img);
        } else {
            log_ss << "File is not recognized as an image: " << img_path;
            rr_log_stream_and_clear(
                rr_path, log_ss, rec, rerun::TextLogLevel::Error
            );
        }
    } else if (fs::is_directory(img_path)) {
        // Gather all image files in the directory.
        std::vector<fs::directory_entry> paths;
        for (const auto& entry : fs::directory_iterator(img_path)) {
            if (fs::is_regular_file(entry.status()) &&
                is_image_file(entry.path())) {
                paths.push_back(entry);
            }
        }
        // Sort by file name
        std::sort(
            paths.begin(),
            paths.end(),
            [](const fs::directory_entry& a, const fs::directory_entry& b) {
                return a.path().filename() < b.path().filename();
            }
        );
        for (const auto& entry : paths) {
            cv::Mat img = cv::imread(entry.path(), cv::IMREAD_COLOR);
            images.push_back(img);
        }
    } else {
        log_ss << "Path is neither a file nor a directory: " << img_path;
        rr_log_stream_and_clear(
            rr_path, log_ss, rec, rerun::TextLogLevel::Error
        );
    }

    log_ss << "Loaded " << images.size() << " images from " << img_path;
    rr_log_stream_and_clear(rr_path, log_ss, rec);
    return 0;
}
