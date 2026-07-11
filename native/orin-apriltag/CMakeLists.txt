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

find_path(CUDA_CCCL_INCLUDE_DIR
    cub/device/device_copy.cuh
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES cccl
    REQUIRED)

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

set(ORIN_UPSTREAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/upstream")
set(ORIN_CUDA_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/staged/frc971/orin")
file(REMOVE_RECURSE "${ORIN_CUDA_SOURCE_DIR}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/staged/frc971")
file(COPY "${ORIN_UPSTREAM_ROOT}/frc971/orin"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/staged/frc971")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/overrides/frc971/orin/cuda.h"
    "${ORIN_CUDA_SOURCE_DIR}/cuda.h"
    COPYONLY)

set(ORIN_LINE_FIT_SOURCE "${ORIN_CUDA_SOURCE_DIR}/line_fit_filter.cu")
file(READ "${ORIN_LINE_FIT_SOURCE}" ORIN_LINE_FIT_CONTENT)
set(ORIN_FIT_LINES_ANCHOR "  constexpr size_t kThreads = 128;")
set(ORIN_FIT_QUADS_ANCHOR "  constexpr size_t kThreads = MaxRankedIndex();")
string(FIND "${ORIN_LINE_FIT_CONTENT}" "${ORIN_FIT_LINES_ANCHOR}"
    ORIN_FIT_LINES_OFFSET)
string(FIND "${ORIN_LINE_FIT_CONTENT}" "${ORIN_FIT_QUADS_ANCHOR}"
    ORIN_FIT_QUADS_OFFSET)
if(ORIN_FIT_LINES_OFFSET EQUAL -1 OR ORIN_FIT_QUADS_OFFSET EQUAL -1)
    message(FATAL_ERROR "Pinned line-fit source no longer matches zero-work override")
endif()
string(REPLACE "${ORIN_FIT_LINES_ANCHOR}"
    "  if (points == 0) return;\n${ORIN_FIT_LINES_ANCHOR}"
    ORIN_LINE_FIT_CONTENT "${ORIN_LINE_FIT_CONTENT}")
string(REPLACE "${ORIN_FIT_QUADS_ANCHOR}"
    "  if (num_extents == 0) return;\n${ORIN_FIT_QUADS_ANCHOR}"
    ORIN_LINE_FIT_CONTENT "${ORIN_LINE_FIT_CONTENT}")
file(WRITE "${ORIN_LINE_FIT_SOURCE}" "${ORIN_LINE_FIT_CONTENT}")

set(ORIN_CUDA_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/compat")
file(MAKE_DIRECTORY "${ORIN_CUDA_COMPAT_INCLUDE_DIR}/cub/iterator")
file(WRITE
    "${ORIN_CUDA_COMPAT_INCLUDE_DIR}/cub/iterator/discard_output_iterator.cuh"
    [=[#pragma once
#include <thrust/iterator/discard_iterator.h>
namespace cub {
template <typename OffsetT>
using DiscardOutputIterator = thrust::discard_iterator<>;
}  // namespace cub
]=])
file(WRITE
    "${ORIN_CUDA_COMPAT_INCLUDE_DIR}/cub/iterator/transform_input_iterator.cuh"
    [=[#pragma once
#include <thrust/iterator/transform_iterator.h>
namespace cub {
template <typename Value, typename UnaryOperation, typename InputIterator>
using TransformInputIterator =
    thrust::transform_iterator<UnaryOperation, InputIterator, Value, Value>;
}  // namespace cub
]=])

set(ORIN_CUDA_SOURCES
    ${ORIN_CUDA_SOURCE_DIR}/971apriltag.cu
    ${ORIN_CUDA_SOURCE_DIR}/apriltag_detect.cu
    ${ORIN_CUDA_SOURCE_DIR}/labeling_allegretti_2019_BKE.cu
    ${ORIN_CUDA_SOURCE_DIR}/line_fit_filter.cu
    ${ORIN_CUDA_SOURCE_DIR}/points.cu
    ${ORIN_CUDA_SOURCE_DIR}/threshold.cu
    ${ORIN_CUDA_SOURCE_DIR}/cuda.cc
    jni/CudaAprilTagDetector.cu)

add_library(orin_apriltag_backend STATIC ${ORIN_CUDA_SOURCES})
target_include_directories(orin_apriltag_backend
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/jni"
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/staged"
        "${ORIN_CUDA_COMPAT_INCLUDE_DIR}"
        "${ORIN_UPSTREAM_ROOT}"
        "${CUDA_CCCL_INCLUDE_DIR}")
target_compile_options(orin_apriltag_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(orin_apriltag_backend
    PUBLIC CUDA::cudart apriltag_vendor Threads::Threads)

add_executable(detector_registry_test
    jni/DetectorRegistry.cpp
    tests/DetectorRegistryTest.cpp)
target_include_directories(detector_registry_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/jni")
target_link_libraries(detector_registry_test PRIVATE Threads::Threads)
add_test(NAME detector_registry_test COMMAND detector_registry_test)
set_tests_properties(detector_registry_test PROPERTIES TIMEOUT 10)

add_executable(cuda_detector_smoke_test
    tests/CudaDetectorSmokeTest.cpp)
target_link_libraries(cuda_detector_smoke_test PRIVATE orin_apriltag_backend)
add_test(NAME cuda_detector_smoke_test COMMAND cuda_detector_smoke_test)
set_tests_properties(cuda_detector_smoke_test PROPERTIES TIMEOUT 180)
