#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>

#include "data.hpp"
#include "rerun.hpp"
#include "rerun_helpers.hpp"

#define IMAGES_PATH "doom_gif"

namespace fs = std::filesystem;

void help_msg(char** argv) {
    printf("Usage: %s IP:PORT\n", argv[0]);
    return;
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

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("You must provide the viewer IP:PORT\n");
        help_msg(argv);
        return 1;
    }

    if (std::string(argv[1]) == "help") {
        help_msg(argv);
        return 0;
    }

    const auto rec = rerun::RecordingStream("mve");
    auto connect_result = rec.connect_grpc(build_url(argv[1]));
    if (connect_result.is_err()) {
        fprintf(stderr, "Unable to connect to Rerun viewer\n");
        return 1;
    }
    std::string rr_path("remote_logging");
    rr_log_message(rr_path, "Connected...", rec);

    std::vector<cv::Mat> images;
    if (load_saved_images(rr_path, IMAGES_PATH, images, rec) != 0) {
        return 1;
    }

    for (size_t i = 0; i < images.size(); ++i) {
        rec.set_time_sequence("poses", i);
        std::string img_path(rr_path + "/original");
        rr_log_mat_image(
            img_path, images[i], rerun::ColorModel::BGR, rec, img_path
        );
        img_path = rr_path + "/image1";
        rr_log_mat_image(
            img_path, images[i], rerun::ColorModel::BGR, rec, img_path
        );
        img_path = rr_path + "/image2";
        rr_log_mat_image(
            img_path, images[i], rerun::ColorModel::BGR, rec, img_path
        );

        cv::Matx31d rvec = data::POSES[i][0];
        cv::Matx31d tvec = data::POSES[i][1];
        rr_log_pose_estimation(
            rr_path, images[i], rvec, tvec, data::CAMERA_MATRIX, rec
        );
    }

    return 0;
}
