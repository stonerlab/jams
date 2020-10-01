#include <jams/maths/parallelepiped.h>
#include <jams/containers/vec3.h>

double jams::maths::parallelogram_area(const Vec3 &a, const Vec3 &b) {
  return norm(cross(a, b));
}

double jams::maths::parallelogram_height(const Vec3 &a, const Vec3 &b) {
  return parallelogram_area(a, b) / norm(b);
}

double jams::maths::parallelogram_inradius(const Vec3 &a, const Vec3 &b) {
  return 0.5 * std::min(parallelogram_height(a, b), parallelogram_height(b, a));
}

double jams::maths::parallelepiped_volume(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return abs(dot(cross(a, b), c));
}

double jams::maths::parallelepiped_height(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return parallelepiped_volume(a, b, c) / norm(cross(a, b));
}

double jams::maths::parallelepiped_inradius(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return 0.5 * std::min({parallelepiped_height(a, b, c), parallelepiped_height(c, a, b), parallelepiped_height(b, c, a)});
}

double jams::maths::parallelepiped_longest_diagonal(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return std::max({norm(a + b + c), norm(-a + b + c), norm(a - b + c), norm(a + b - c)});
}



