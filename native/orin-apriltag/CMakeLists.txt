cmake_minimum_required(VERSION 3.28)

project(PhotonOrinAprilTag LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architecture")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

enable_testing()

if(NOT DEFINED PV_SOURCE_REVISION)
    set(PV_SOURCE_REVISION "unknown")
endif()
string(TIMESTAMP PV_BUILD_TIMESTAMP UTC)
set(PV_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${PV_GENERATED_INCLUDE_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/BuildInfo.h.in"
    "${PV_GENERATED_INCLUDE_DIR}/BuildInfo.h"
    @ONLY)

set(APRILTAG_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/upstream/third_party/apriltag")
set(APRILTAG_VENDOR_SOURCES
    "${APRILTAG_VENDOR_DIR}/apriltag.c"
    "${APRILTAG_VENDOR_DIR}/apriltag_pose.c"
    "${APRILTAG_VENDOR_DIR}/apriltag_quad_thresh.c"
    "${APRILTAG_VENDOR_DIR}/common/g2d.c"
    "${APRILTAG_VENDOR_DIR}/common/getopt.c"
    "${APRILTAG_VENDOR_DIR}/common/homography.c"
    "${APRILTAG_VENDOR_DIR}/common/image_u8.c"
    "${APRILTAG_VENDOR_DIR}/common/image_u8x3.c"
    "${APRILTAG_VENDOR_DIR}/common/image_u8x4.c"
    "${APRILTAG_VENDOR_DIR}/common/matd.c"
    "${APRILTAG_VENDOR_DIR}/common/pam.c"
    "${APRILTAG_VENDOR_DIR}/common/pjpeg.c"
    "${APRILTAG_VENDOR_DIR}/common/pjpeg-idct.c"
    "${APRILTAG_VENDOR_DIR}/common/pnm.c"
    "${APRILTAG_VENDOR_DIR}/common/string_util.c"
    "${APRILTAG_VENDOR_DIR}/common/svd22.c"
    "${APRILTAG_VENDOR_DIR}/common/time_util.c"
    "${APRILTAG_VENDOR_DIR}/common/unionfind.c"
    "${APRILTAG_VENDOR_DIR}/common/workerpool.c"
    "${APRILTAG_VENDOR_DIR}/common/zarray.c"
    "${APRILTAG_VENDOR_DIR}/common/zhash.c"
    "${APRILTAG_VENDOR_DIR}/common/zmaxheap.c"
    "${APRILTAG_VENDOR_DIR}/tag16h5.c"
    "${APRILTAG_VENDOR_DIR}/tag25h9.c"
    "${APRILTAG_VENDOR_DIR}/tag36h11.c"
    "${APRILTAG_VENDOR_DIR}/tagCircle21h7.c"
    "${APRILTAG_VENDOR_DIR}/tagCircle49h12.c"
    "${APRILTAG_VENDOR_DIR}/tagCustom48h12.c"
    "${APRILTAG_VENDOR_DIR}/tagStandard41h12.c"
    "${APRILTAG_VENDOR_DIR}/tagStandard52h13.c")

add_library(apriltag_vendor STATIC ${APRILTAG_VENDOR_SOURCES})
set_target_properties(apriltag_vendor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(apriltag_vendor PUBLIC "${APRILTAG_VENDOR_DIR}")
target_link_libraries(apriltag_vendor PUBLIC Threads::Threads m)
