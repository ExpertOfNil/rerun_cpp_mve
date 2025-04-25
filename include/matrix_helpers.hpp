#ifndef MATRIX_HELPERS_HPP
#define MATRIX_HELPERS_HPP

#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>

cv::Matx33f rotation_from_transform(const cv::Matx44f mat);
cv::Matx31f translation_from_transform(const cv::Matx44f mat);
cv::Matx31f convert_world_to_local(
    const cv::Matx44f xform, const cv::Matx31f vec
);
cv::Matx31f convert_local_to_world(
    const cv::Matx44f xform, const cv::Matx31f vec
);
cv::Matx44f transform_from_translation_rotation_rodrigues(
    const cv::Matx31f pos, const cv::Matx31f rodrigues
);
cv::Matx31f rodrigues_from_transform(const cv::Matx44f transform);

#endif /* MATRIX_HELPERS_HPP */
