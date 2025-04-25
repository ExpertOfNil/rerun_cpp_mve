cmake_minimum_required(VERSION 3.16)
project(rerun_cpp_mve LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -ggdb")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# OpenCV : Tested with v4.6.0
find_package(OpenCV REQUIRED)

# Rerun
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/download/0.23.0/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/rerun_helpers.cpp
    src/matrix_helpers.cpp
)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC ${OpenCV_LIBS}
    PRIVATE rerun_sdk
)
