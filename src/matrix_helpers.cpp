#include "matrix_helpers.hpp"

// NOTE: OpenCV matrices are stored (and initialized) in row-major order

/* Create rotation matrix from 4x4 transformation matrix */
cv::Matx33f rotation_from_transform(const cv::Matx44f mat) {
    return cv::Matx33f{
        // clang-format off
        mat(0, 0), mat(0, 1), mat(0, 2),
        mat(1, 0), mat(1, 1), mat(1, 2),
        mat(2, 0), mat(2, 1), mat(2, 2),
        // clang-format on
    };
}

/* Create translation vector from 4x4 transformation matrix */
cv::Matx31f translation_from_transform(const cv::Matx44f mat) {
    return cv::Matx31f{
        // clang-format off
        mat(0, 3), mat(1, 3), mat(2, 3),
        // clang-format on
    };
}

/* P_local = M^{-1} * P_world */
cv::Matx31f convert_world_to_local(
    const cv::Matx44f xform, const cv::Matx31f vec
) {
    cv::Matx44f mat = xform.inv();
    float veci = vec(0, 0);
    float vecj = vec(1, 0);
    float veck = vec(2, 0);
    return cv::Matx31f{
        mat(0, 0) * veci + mat(0, 1) * vecj + mat(0, 2) * veck,
        mat(1, 0) * veci + mat(1, 1) * vecj + mat(1, 2) * veck,
        mat(2, 0) * veci + mat(2, 1) * vecj + mat(2, 2) * veck,
    };
}

/* P_world = M * P_local */
cv::Matx31f convert_local_to_world(
    const cv::Matx44f xform, const cv::Matx31f vec
) {
    float veci = vec(0, 0);
    float vecj = vec(1, 0);
    float veck = vec(2, 0);
    return cv::Matx31f{
        xform(0, 0) * veci + xform(0, 1) * vecj + xform(0, 2) * veck,
        xform(1, 0) * veci + xform(1, 1) * vecj + xform(1, 2) * veck,
        xform(2, 0) * veci + xform(2, 1) * vecj + xform(2, 2) * veck,
    };
}

/* Create a transformation matrix from a translation and euler rotation */
cv::Matx44f transform_from_translation_rotation_rodrigues(
    const cv::Matx31f pos, const cv::Matx31f rodrigues
) {
    cv::Matx33f rot_mat;
    // Assume output is normalized
    cv::Rodrigues(rodrigues, rot_mat);
    return cv::Matx44f{
        // clang-format off
        rot_mat(0, 0), rot_mat(0, 1), rot_mat(0, 2), pos(0, 0),
        rot_mat(1, 0), rot_mat(1, 1), rot_mat(1, 2), pos(1, 0),
        rot_mat(2, 0), rot_mat(2, 1), rot_mat(2, 2), pos(2, 0),
        0, 0, 0, 1,
        // clang-format on
    };
}

/* Create a Rodrigues vector from a transformation matrix */
cv::Matx31f rodrigues_from_transform(const cv::Matx44f transform) {
    cv::Matx31f rodrigues;
    // Assume output is normalized
    cv::Rodrigues(rotation_from_transform(transform), rodrigues);
    return rodrigues;
}
