#include <jams/lattice/interaction_neartree.h>
#include <jams/maths/parallelepiped.h>

#include <vector>
#include <array>
#include <cassert>

#include <iostream>

jams::InteractionNearTree::InteractionNearTree(const Vec3&a, const Vec3& b, const Vec3& c, const Vec3b& pbc, const double& r_cutoff, const double& epsilon) :
  r_cutoff_(r_cutoff),
  epsilon_(epsilon),
  a_(a),
  b_(b),
  c_(c),
  neartree_(
      [](const NearTreeDataType& a, const NearTreeDataType& b)->double {
        return norm(a.first-b.first);}){
  
  num_image_cells_[0] = pbc[0] ? int(ceil((r_cutoff) / jams::maths::parallelepiped_height(b, c, a))) : 0;
  num_image_cells_[1] = pbc[1] ? int(ceil((r_cutoff) / jams::maths::parallelepiped_height(c, a, b))) : 0;
  num_image_cells_[2] = pbc[2] ? int(ceil((r_cutoff) / jams::maths::parallelepiped_height(a, b, c))) : 0;
}

std::vector<jams::InteractionNearTree::NearTreeDataType>
jams::InteractionNearTree::neighbours(const Vec3 &r, const double &radius) const {
  assert(radius <= r_cutoff_);
  return neartree_.find_in_annulus(epsilon_, radius, {r, 0}, epsilon_);
}

//void jams::InteractionNearTree::insert_sites(const std::vector<Vec3>& sites) {
//  // There are 6 surface normals for a parallelepiped but the parallel nature
//  // means each pair of opposite surfaces has the same normal with opposite
//  // sign, so we only need calculate 3 normals.
//
//  //        +------------+
//  //       / \         /  \
//  //      /    +-----------+
//  //     /    /c     /    /
//  //    /    /      /    /
//  //   +----/------+    /
//  // b  \  /         \ /
//  //      +-----------+
//  //     o      a
//  //
//  // Order of a,b,c defines the normal as either out of the volume (positive)
//  // or into the volume (negative)
//  //
//  // Planes touching the origin:
//  // ac-plane: front
//  // ba-plane: bottom
//  // cb-plane: left
//  //
//  // opposite planes:
//  // ac-plane: back
//  // ab-plane: top
//  // bc-plane: right
//  //
//  //   //      b +------------+
//  //  //       / \  o       /  \
//  //  //      /    +-----------+ a
//  //  //     /    /      /    /
//  //  //    /    /      /    /
//  //  //   +----/------+    /
//  //  //    \  /c        \ /
//  //  //      +-----------+
//  //  //
//  // Planes touching the origin:
//  // ac-plane: front
//  // ba-plane: bottom
//  // cb-plane: left
//  //
//  // opposite planes:
//  // ac-plane: back
//  // ab-plane: top
//  // bc-plane: right
//
//  enum class Handedness { right, left };
//
//  Handedness handedness = Handedness::right;
//  if (dot(cross(a_, b_), c_) < 0) {
//    handedness = Handedness::left;
//  }
//
//  Vec3 front, bottom, left, back, top, right;
//
//  if (handedness == Handedness::right) {
//    front = normalize(cross(a_, c_));
//    bottom = normalize(cross(b_, a_));
//    left = normalize(cross(c_, b_));
//
//    back = -front;
//    top = -bottom;
//    right = -left;
//  } else {
//    front = normalize(cross(c_, a_));
//    bottom = normalize(cross(b_, a_));
//    left = normalize(cross(b_, c_));
//
//    back = -front;
//    top = -bottom;
//    right = -left;
//  }
//
//  std::vector<NearTreeDataType> haloed_sites;
//
//  for (auto i = 0; i < sites.size(); ++i) {
//    const auto& r = sites[i];
//    for (auto h = -num_image_cells_[0]; h < num_image_cells_[0] + 1; ++h) {
//      for (auto k = -num_image_cells_[1]; k < num_image_cells_[1] + 1; ++k) {
//        for (auto l = -num_image_cells_[2]; l < num_image_cells_[2] + 1; ++l) {
//          auto r_image = r + (h * a_ + k * b_ + l * c_);
//
//
//          // To calculate the distance of a point, r_image, to the plane
//          // we need to know the normal of the plane and a single point within
//          // that plane (tells us where along the normal in 3D space the plane
//          // exists). The dot product then projects the distance and the sign
//          // tells us whether the point, r_image, is in front of the plane
//          // (positive sign is along the positive normal direction) or behind
//          // the plane).
//          const auto distance_to_plane = [&](const Vec3 &normal,
//                                             const Vec3 &point_in_plane)->double {
//              return dot(normal, r_image - point_in_plane);
//          };
//
//          // These are the signed distances to the infinite plane aligned with
//          // each face of the parallelepiped.
//          std::array<double, 6> distances;
//
//          if (handedness == Handedness::right) {
//            distances[0] = distance_to_plane(front, {0, 0, 0});
//            distances[1] = distance_to_plane(bottom, {0, 0, 0});
//            distances[2] = distance_to_plane(left, {0, 0, 0});
//            distances[3] = distance_to_plane(back, b_);
//            distances[4] = distance_to_plane(top, c_);
//            distances[5] = distance_to_plane(right, a_);
//          } else {
//            distances[0] = distance_to_plane(front, {0, 0, 0});
//            distances[1] = distance_to_plane(top, {0, 0, 0});
//            distances[2] = distance_to_plane(left, {0, 0, 0});
//            distances[3] = distance_to_plane(back, b_);
//            distances[4] = distance_to_plane(bottom, c_);
//            distances[5] = distance_to_plane(right, a_);
//          }
//
//          // The surface normals all point outwards, so any point with
//          // all negative or zero distances must be inside the parallelepiped. We want
//          // a halo of site 'r_cutoff_' outside of the parallelepiped to also
//          // be included so we accept any point where all signed distances
//          // are less than this cutoff.
//          //
//          // NOTES:
//          // - This method does not give us the shortest distance to a
//          //   face but is efficient because we never have to work out if the
//          //   point is within the bounds of a face (because the distance calculation
//          //   above is for the infinite plane regardless of the face extent).
//          //
//          // - We use !definately_greater_than rather than definately_less_than
//          //   to make this inclusive of the very edge of the radius.
//
//          bool inside_cutoff = std::none_of(distances.begin(), distances.end(),
//                  [&](const double &x) {
//                      return definately_greater_than(x, r_cutoff_, epsilon_);
//                  });
//
//          if (!inside_cutoff) {
//            continue;
//          }
//
//          haloed_sites.emplace_back(r_image, i);
//        }
//      }
//    }
//  }
//
//  neartree_.insert(haloed_sites);
//}

void jams::InteractionNearTree::insert_sites(const std::vector<Vec3>& sites) {
  // There are 6 surface normals for a parallelepiped but the parallel nature
  // means each pair of opposite surfaces has the same normal with opposite
  // sign, so we only need calculate 3 normals.

  //        +------------+
  //       / \         /  \
  //      /    +-----------+
  //     /    /c     /    /
  //    /    /      /    /
  //   +----/------+    /
  // b  \  /         \ /
  //      +-----------+
  //     o      a
  //
  // Order of a,b,c defines the normal as either out of the volume (positive)
  // or into the volume (negative)
  //
  // Planes touching the origin:
  // ac-plane: front
  // ba-plane: bottom
  // cb-plane: left
  //
  // opposite planes:
  // ac-plane: back
  // ab-plane: top
  // bc-plane: right
  //
  //   //      b +------------+
  //  //       / \  o       /  \
  //  //      /    +-----------+ a
  //  //     /    /      /    /
  //  //    /    /      /    /
  //  //   +----/------+    /
  //  //    \  /c        \ /
  //  //      +-----------+
  //  //
  // Planes touching the origin:
  // ac-plane: front
  // ba-plane: bottom
  // cb-plane: left
  //
  // opposite planes:
  // ac-plane: back
  // ab-plane: top
  // bc-plane: right

  enum class Handedness { right, left };

  Handedness handedness = Handedness::right;
  if (dot(cross(a_, b_), c_) < 0) {
    handedness = Handedness::left;
  }

  Vec3 front, bottom, left, back, top, right;

  if (handedness == Handedness::right) {
    front = normalize(cross(a_, c_));
    bottom = normalize(cross(b_, a_));
    left = normalize(cross(c_, b_));

    back = -front;
    top = -bottom;
    right = -left;
  } else {
    front = normalize(cross(c_, a_));
    bottom = normalize(cross(b_, a_));
    left = normalize(cross(b_, c_));

    back = -front;
    top = -bottom;
    right = -left;
  }

  std::vector<NearTreeDataType> haloed_sites;

  for (auto i = 0; i < sites.size(); ++i) {
    const auto& r = sites[i];
    for (auto h = -num_image_cells_[0]; h < num_image_cells_[0] + 1; ++h) {
      for (auto k = -num_image_cells_[1]; k < num_image_cells_[1] + 1; ++k) {
        for (auto l = -num_image_cells_[2]; l < num_image_cells_[2] + 1; ++l) {
          auto r_image = r + (h * a_ + k * b_ + l * c_);


          // To calculate the distance of a point, r_image, to the plane
          // we need to know the normal of the plane and a single point within
          // that plane (tells us where along the normal in 3D space the plane
          // exists). The dot product then projects the distance and the sign
          // tells us whether the point, r_image, is in front of the plane
          // (positive sign is along the positive normal direction) or behind
          // the plane).
          const auto distance_to_plane = [&](const Vec3 &normal,
                                             const Vec3 &point_in_plane)->double {
              return dot(normal, r_image - point_in_plane);
          };

          // These are the signed distances to the infinite plane aligned with
          // each face of the parallelepiped.
          std::array<double, 6> distances;

          if (handedness == Handedness::right) {
            distances[0] = distance_to_plane(front, {0, 0, 0});
            distances[1] = distance_to_plane(bottom, {0, 0, 0});
            distances[2] = distance_to_plane(left, {0, 0, 0});
            distances[3] = distance_to_plane(back, b_);
            distances[4] = distance_to_plane(top, c_);
            distances[5] = distance_to_plane(right, a_);
          } else {
            distances[0] = distance_to_plane(front, {0, 0, 0});
            distances[1] = distance_to_plane(top, {0, 0, 0});
            distances[2] = distance_to_plane(left, {0, 0, 0});
            distances[3] = distance_to_plane(back, b_);
            distances[4] = distance_to_plane(bottom, c_);
            distances[5] = distance_to_plane(right, a_);
          }

          // The surface normals all point outwards, so any point with
          // all negative or zero distances must be inside the parallelepiped. We want
          // a halo of site 'r_cutoff_' outside of the parallelepiped to also
          // be included so we accept any point where all signed distances
          // are less than this cutoff.
          //
          // NOTES:
          // - This method does not give us the shortest distance to a
          //   face but is efficient because we never have to work out if the
          //   point is within the bounds of a face (because the distance calculation
          //   above is for the infinite plane regardless of the face extent).
          //
          // - We use !definately_greater_than rather than definately_less_than
          //   to make this inclusive of the very edge of the radius.

          bool inside_cutoff = std::none_of(distances.begin(), distances.end(),
                                            [&](const double &x) {
                                                return definately_greater_than(x, r_cutoff_, epsilon_);
                                            });

          if (!inside_cutoff) {
            continue;
          }

          haloed_sites.emplace_back(r_image, i);
        }
      }
    }
  }

  neartree_.insert(haloed_sites);
}


int jams::InteractionNearTree::num_neighbours(const Vec3 &r, const double &radius) const {
  return neartree_.num_neighbours_in_radius(radius, {r, 0}, epsilon_) - 1;
}

