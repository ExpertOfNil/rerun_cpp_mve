#include "rerun_helpers.hpp"

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <set>

#include "matrix_helpers.hpp"

/** NOTE: According to Rerun, if RecordingStream is not enabled, all log
 * functions "early out".  See `RecordingStream::is_enabled()`
 */

/** Log text to stdout or stderr and, if enabled, to the Rerun viewer */
void rr_log_message(
    std::string path,
    const char* message,
    const rerun::RecordingStream& rec,
    TextLogLevel level
) {
    if (level.c_str() == TextLogLevel::Error.c_str()) {
        std::cerr << message << std::endl;
    } else {
        std::cout << message << std::endl;
    }
    rec.log(path, rerun::TextLog(message).with_level(level));
    return;
}

/** Logs stringstream content to stdout and, if enabled, Rerun, then clears the
 * stringstream.
 */
void rr_log_stream_and_clear(
    std::string path,
    std::stringstream& ss,
    const rerun::RecordingStream& rec,
    TextLogLevel level
) {
    if (level.c_str() == TextLogLevel::Error.c_str()) {
        std::cerr << ss.str() << std::endl;
    } else {
        std::cout << ss.str() << std::endl;
    }
    rec.log(path, rerun::TextLog(ss.str()).with_level(level));
    ss.str("");
    return;
}

/** Converts `cv::Matx44f` transformation matrix and logs it to Rerun
 *
 * Note that this will transform whatever 3D components are also logged at
 * `path`.  If you want to log and visualize just a transform, you can turn on
 * the axis system visualization in the Rerun viewer, or log a 3D element to the
 * same path.
 */
void rr_log_transform3d(
    std::string path,
    cv::Matx44f transform,
    rerun::Vec3D scale,
    const rerun::RecordingStream& rec
) {
    cv::Matx33f cv_rot = rotation_from_transform(transform);
    cv::Matx31f cv_xlat = translation_from_transform(transform);
    auto xlat = rerun::components::Translation3D(
        cv_xlat(0, 0), cv_xlat(1, 0), cv_xlat(2, 0)
    );
    auto rot = rerun::datatypes::Mat3x3({
        rerun::Vec3D(cv_rot(0, 0), cv_rot(1, 0), cv_rot(2, 0)),
        rerun::Vec3D(cv_rot(0, 1), cv_rot(1, 1), cv_rot(2, 1)),
        rerun::Vec3D(cv_rot(0, 2), cv_rot(1, 2), cv_rot(2, 2)),
    });
    rec.log(
        path,
        rerun::Transform3D::from_translation_mat3x3(xlat, rot).with_scale(scale)
    );
}

/** Log an axis system to rerun */
void rr_log_axis_system(
    std::string path, float scale, const rerun::RecordingStream& rec
) {
    std::vector<rerun::Vec3D> vecs = {
        {scale, 0.0f, 0.0f},
        {0.0f, scale, 0.0f},
        {0.0f, 0.0f, scale},
    };
    std::vector<rerun::Position3D> origins = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    std::vector<rerun::Color> colors = {
        rerun::Color(255, 0, 0, 255),
        rerun::Color(0, 255, 0, 255),
        rerun::Color(0, 0, 255, 255),
    };
    rec.log(
        path,
        rerun::Arrows3D::from_vectors(vecs).with_origins(origins).with_colors(
            colors
        )
    );
}

void rr_log_mat_image(
    std::string path,
    cv::Mat img,
    rerun::ColorModel color_model,
    const rerun::RecordingStream& rec,
    std::string tag = ""
) {
    if (!tag.empty()) {
        cv::putText(
            img, tag, {10, 10}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}
        );
    }
    rec.log(
        path,
        rerun::Image(
            rerun::borrow(img.data, img.total() * img.elemSize()),
            rerun::WidthHeight(
                static_cast<uint32_t>(img.cols), static_cast<uint32_t>(img.rows)
            ),
            color_model
        )
    );
}

void rr_log_keypoints_image(
    std::string path,
    std::vector<cv::KeyPoint> keypoints,
    cv::Mat image,
    const rerun::RecordingStream& rec
) {
    cv::Mat draw;
    cv::cvtColor(image, draw, cv::COLOR_GRAY2BGR);
    cv::drawKeypoints(draw, keypoints, draw);
    rr_log_mat_image(path, draw, rerun::ColorModel::BGR, rec);
}

void rr_log_points3d(
    std::string path,
    const std::vector<cv::Point3f>& points,
    const rerun::RecordingStream& rec
) {
    std::vector<std::array<float, 3>> pts;
    std::vector<rerun::Color> colors;
    std::vector<float> radii;
    for (const cv::Point3f& pt : points) {
        pts.push_back({pt.x, pt.y, pt.z});
        colors.emplace_back(255, 255, 255, 255);
        radii.push_back(0.1f);
    }
    rec.log(path, rerun::Points3D(pts).with_colors(colors).with_radii(radii));
}

/** Logs source image and pose estimation results to Rerun
 *
 * Image will be accessible via 2D, as well as in 3D with camera intrinsics,
 * frustum, and transformation matrix.
 */
void rr_log_pose_estimation(
    std::string path,
    const cv::Mat image,
    const cv::Matx31d& rvec,
    const cv::Matx31d& tvec,
    const cv::Matx33d& camera_matrix,
    const rerun::RecordingStream& rec
) {
    std::stringstream log_ss;
    std::string main_log_path = path + "/pose_estimate";

    log_ss << "curr_rvec = [" << rvec(0, 0) << ", " << rvec(1, 0) << ", "
           << rvec(2, 0) << "]";
    rr_log_stream_and_clear(main_log_path, log_ss, rec);

    // TODO : Find a better way to log Vec3D, preferably something queryable
    rec.log(
        main_log_path + "/pose_vecs/rvec",
        rerun::Tensor().with_data(
            rerun::TensorData(
                {3, 1},
                std::array<float, 3>{
                    (float)rvec(0, 0), (float)rvec(1, 0), (float)rvec(2, 0)
                }
            )
        )
    );
    log_ss << "curr_tvec = [" << tvec(0, 0) << ", " << tvec(1, 0) << ", "
           << tvec(2, 0) << "]";
    rr_log_stream_and_clear(main_log_path, log_ss, rec);
    rec.log(
        main_log_path + "/pose_vecs/tvec",
        rerun::Tensor().with_data(
            rerun::TensorData(
                {3, 1},
                std::array<float, 3>{
                    (float)tvec(0, 0), (float)tvec(1, 0), (float)tvec(2, 0)
                }
            )
        )
    );

    // Log camera intrinsics
    log_ss << "Camera Matrix: " << camera_matrix;
    rr_log_stream_and_clear(main_log_path + "/calibration", log_ss, rec);

    // Transform image and pinhole
    cv::Matx44f transform =
        transform_from_translation_rotation_rodrigues(tvec, rvec);

    /* We currently have a view matrix.  We need a camera world transform, which
     * is the inverse of the view matrix. We aslo need to convert from RUB
     * orientation to RFU.
     */
    cv::Matx44f rfu_to_rub = {
        // clang-format off
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
        // clang-format on
    };
    transform = rfu_to_rub * transform.inv();

    // Log the world axis system (default is RUB in rerun)
    std::string axis_system_path = main_log_path + "/axis";
    rr_log_axis_system(axis_system_path, 1.0, rec);
    rr_log_transform3d(
        axis_system_path, rfu_to_rub, {10.0f, 10.0f, 10.0f}, rec
    );

    // Log source image with pose estimate and camera intrinsics
    std::string image_log_path = main_log_path + "/image";
    if (image.empty()) {
        rr_log_message(
            image_log_path, "Image is EMPTY!", rec, rerun::TextLogLevel::Error
        );
    } else {
        // Log dummy image in place of logging back-projected points to Rerun
        rr_log_mat_image(image_log_path, image, rerun::ColorModel::BGR, rec);
    }

    // Log camera intrinsics as a pinhole camera frustum.  Contains image above.
    auto focal_length = rerun::Vec2D{
        static_cast<float>(camera_matrix(0, 0)),
        static_cast<float>(camera_matrix(1, 1)),
    };
    auto resolution = rerun::Vec2D{
        static_cast<float>(camera_matrix(0, 2)) * 2.0f,
        static_cast<float>(camera_matrix(1, 2)) * 2.0f,
    };

    // Log the camera
    rec.log(
        image_log_path,
        rerun::Pinhole::from_focal_length_and_resolution(
            focal_length, resolution
        )
            .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
            .with_many_image_plane_distance(
                rerun::components::ImagePlaneDistance(1.0f)
            )
    );

    // Transform camera to estimated pose
    rr_log_transform3d(image_log_path, transform, {1.0f, 1.0f, 1.0f}, rec);
}
