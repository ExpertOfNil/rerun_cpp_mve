cmake_minimum_required(VERSION 3.16)
project(rerun_cpp_mve LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -ggdb")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# OpenCV : Tested with v4.6.0
find_package(OpenCV REQUIRED)
find_package(rerun_sdk 0.22.1 REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/rerun_helpers.cpp
    src/matrix_helpers.cpp
    src/utils.cpp
)

# If image logging is disabled, set the SKIP_IMG_LOG flag
if(SKIP_IMG_LOG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SKIP_IMG_LOG)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC ${OpenCV_LIBS}
    PRIVATE rerun_sdk
)

message(STATUS "CMAKE_BUILD_TYPE:  ${CMAKE_BUILD_TYPE}")
message(STATUS "SKIP_IMG_LOG    :  ${SKIP_IMG_LOG}")
