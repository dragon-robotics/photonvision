#include "frc971/orin/971apriltag.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

void ExpectNear(double expected, double actual, double tolerance,
                const char* label) {
  if (std::abs(expected - actual) > tolerance) {
    throw std::runtime_error(std::string(label) + " expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

void TestTangentialP2RoundTrip() {
  const frc971::apriltag::CameraMatrix camera{
      .fx = 800.0,
      .cx = 640.0,
      .fy = 810.0,
      .cy = 480.0,
  };
  const frc971::apriltag::DistCoeffs distortion{
      .k1 = 0.0,
      .k2 = 0.0,
      .p1 = 0.0,
      .p2 = 0.15,
      .k3 = 0.0,
  };
  constexpr double kUndistortedX = 0.2;
  constexpr double kUndistortedY = -0.1;
  const double r_sq = kUndistortedX * kUndistortedX +
                      kUndistortedY * kUndistortedY;
  const double distorted_x =
      kUndistortedX + distortion.p2 * (r_sq + 2.0 * kUndistortedX * kUndistortedX);
  const double distorted_y =
      kUndistortedY + 2.0 * distortion.p2 * kUndistortedX * kUndistortedY;
  double u = distorted_x * camera.fx + camera.cx;
  double v = distorted_y * camera.fy + camera.cy;

  if (!frc971::apriltag::GpuDetector::UnDistort(&u, &v, &camera, &distortion)) {
    throw std::runtime_error("Tangential distortion round trip did not converge");
  }

  ExpectNear(kUndistortedX * camera.fx + camera.cx, u, 1e-3,
             "undistorted u");
  ExpectNear(kUndistortedY * camera.fy + camera.cy, v, 1e-3,
             "undistorted v");
}

}  // namespace

int main() {
  try {
    TestTangentialP2RoundTrip();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
