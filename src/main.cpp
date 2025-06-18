#include <rerun.hpp>

#include "data.hpp"
#include "rerun_helpers.hpp"
#include "utils.hpp"

#define IMAGES_PATH "doom_gif"

int main(int argc, char** argv) {
    auto [cli, err] = parseArgs(argc, argv);
    if (err != OK) {
        if (err == EARLY_OUT) {
            return EXIT_SUCCESS;
        }
        return EXIT_FAILURE;
    }

    /* This must be set prior to creating the `RecordingStream`. However, if set
     * to false and a connection is established, no logs are sent
     */
    if (!cli.enable_rerun) {
        rerun::set_default_enabled(false);
    }

    const auto rec = rerun::RecordingStream("mve");
    if (cli.enable_rerun) {
        if (!rec.connect_tcp(cli.viewer_addr).is_ok()) {
            fprintf(stderr, "Failed to spawn Rerun\n");
            return 1;
        } else {
            printf("Connected to %s\n", cli.viewer_addr.c_str());
        }
    } else {
        printf("Rerun logging disabled.\n");
    }

    ImageBuffer buf = {};
    if (ImageBuffer_Init(&buf, "doom_gif", 20, 1) != OK) {
        return EXIT_FAILURE;
    }
    ImageBuffer_Stats(buf);
    printf("Warming up background loader...\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    printf("Starting image loop...\n");

    if (buf.images.size() <= 0) {
        fprintf(stderr, "Image buffer has no images\n");
        return 1;
    }
    cv::Mat img(buf.images[0].rows, buf.images[0].cols, buf.images[0].type());
    while(true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        ImageBuffer_NextImage(buf, img);
        rr_log_mat_image("images", img, rerun::ColorModel::BGR, rec);
        if (ImageBuffer_ConsumeImage(buf) != OK) {
            return EXIT_FAILURE;
        }
    }

    printf("Shutting down...\n");
    ImageBuffer_Shutdown(buf);
    return EXIT_SUCCESS;
}
