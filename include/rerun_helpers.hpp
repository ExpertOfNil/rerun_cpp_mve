#ifndef RERUN_HELPERS_HPP
#define RERUN_HELPERS_HPP

#include <opencv2/core.hpp>
#include <rerun.hpp>

using TextLogLevel = rerun::components::TextLogLevel;

void rr_log_message(
    std::string path,
    const char* message,
    const rerun::RecordingStream& rec,
    TextLogLevel level = TextLogLevel::Info
);

void rr_log_stream_and_clear(
    std::string path,
    std::stringstream& ss,
    const rerun::RecordingStream& rec,
    TextLogLevel level = TextLogLevel::Info
);

void rr_log_transform3d(
    std::string path,
    cv::Matx44f transform,
    rerun::Vec3D scale,
    const rerun::RecordingStream& rec
);

void rr_log_axis_system(
    std::string path, float scale, const rerun::RecordingStream& rec
);

#ifdef SKIP_IMG_LOG
inline void rr_log_mat_image(
    [[maybe_unused]] std::string path,
    [[maybe_unused]] const cv::Mat img,
    [[maybe_unused]] rerun::ColorModel color_model,
    [[maybe_unused]] const rerun::RecordingStream& rec,
    [[maybe_unused]] std::string tag = ""
) {}
#else
void rr_log_mat_image(
    std::string path,
    cv::Mat img,
    rerun::ColorModel color_model,
    const rerun::RecordingStream& rec,
    std::string tag = ""
);
#endif

void rr_log_keypoints_image(
    std::string path,
    std::vector<cv::KeyPoint> keypoints,
    cv::Mat image,
    const rerun::RecordingStream& rec
);

void rr_log_points3d(
    std::string path,
    const std::vector<cv::Point3f>& points,
    const rerun::RecordingStream& rec
);

void rr_log_pose_estimation(
    std::string path,
    const cv::Mat image,
    const cv::Matx31d& rvec,
    const cv::Matx31d& tvec,
    const cv::Matx33d& camera_matrix,
    const rerun::RecordingStream& rec
);

#endif /* RERUN_HELPERS_HPP */
