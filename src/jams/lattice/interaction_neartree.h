#ifndef JAMS_LATTICE_INTERACTION_NEARTREE_H
#define JAMS_LATTICE_INTERACTION_NEARTREE_H

#include <cassert>
#include <jams/containers/vec3.h>
#include <jams/containers/neartree.h>
#include <functional>

#include "jams/maths/parallelepiped.h"

namespace jams {
    ///
    /// A wrapper class which holds a neartree object where the neartree
    /// contains points within a supercell a,b,c which has optional periodic
    /// boundaries and a maximum cutoff radius. In periodic directions, all
    /// points within r_cutoff of the supercell surface are duplicated in a halo
    /// around the supercell. This is essentially a bruteforce minimum image
    /// convention. By only including points which are within r_cutoff we make
    /// the neartree much smaller than a true bruteforce minimum image duplication
    /// of the supercell. This makes both the creation time of the neartree
    /// and the size of it much smaller.
    ///
    template<typename CoordType>
    class InteractionNearTree {
    public:
        using PositionType = std::array<CoordType, 3>;
        using NearTreeDataType = std::pair<PositionType, int>;
        using NearTreeFunctorType = std::function<CoordType(
            const NearTreeDataType &a, const NearTreeDataType &b)>;
        using NearTreeType = jams::NearTree<NearTreeDataType, NearTreeFunctorType, CoordType>;

        InteractionNearTree(const PositionType &a, const PositionType &b, const PositionType &c, const Vec3b& pbc,
                            const CoordType& r_cutoff, const CoordType& epsilon);

        // Insert a vector of site positions. The index will be numbered from
        // zero to sites.size() - 1
        void insert_sites(const std::vector<PositionType>& sites);

        // Returns the number of sites inside the neartree.
        inline std::size_t size() const { return neartree_.size(); };

        // Returns the memory in bytes of the neartree. We ignore the small
        // memory footprint of this InteractionNearTree wrapper.
        inline std::size_t memory() const { return neartree_.memory(); };

        // Return a list of neighbours within radius of point r.
        // The radius must be less than the r_cutoff of the InteractionNearTree
        std::vector<NearTreeDataType>
        neighbours(const PositionType &r, const CoordType &radius) const;

        std::vector<NearTreeDataType>
        shell(const PositionType &r, const CoordType &radius, const CoordType& width) const;

        // Return a the number of neighbours within radius of point r.
        // We assume the point r is on a site as so we subtract 1 from the number
        // of points with the sphere.
        // The radius must be less than the r_cutoff of the InteractionNearTree
        int num_neighbours(const PositionType &r, const CoordType &radius) const;

    private:

        CoordType r_cutoff_;
        CoordType epsilon_;

        PositionType a_;
        PositionType b_;
        PositionType c_;

        PositionType normal_ac_;
        PositionType normal_cb_;
        PositionType normal_ba_;

        Vec3i num_image_cells_ = {0, 0, 0};

        NearTreeType neartree_;
    };


template<typename CoordType>
InteractionNearTree<CoordType>::InteractionNearTree(
    const PositionType&a, const PositionType& b, const PositionType& c,
    const Vec3b& pbc, const CoordType& r_cutoff, const CoordType& epsilon) :
    r_cutoff_(r_cutoff),
    epsilon_(epsilon),
    a_(a),
    b_(b),
    c_(c),
    neartree_(
        [](const NearTreeDataType& a, const NearTreeDataType& b)->CoordType {
          return norm(a.first-b.first);}){

    num_image_cells_[0] = pbc[0] ? int(ceil((r_cutoff) / jams::maths::parallelepiped_height(b, c, a))) : 0;
    num_image_cells_[1] = pbc[1] ? int(ceil((r_cutoff) / jams::maths::parallelepiped_height(c, a, b))) : 0;
    num_image_cells_[2] = pbc[2] ? int(ceil((r_cutoff) / jams::maths::parallelepiped_height(a, b, c))) : 0;
}

template<typename CoordType>
auto InteractionNearTree<CoordType>::neighbours(const PositionType& r,
                                                      const CoordType& radius) const -> std::vector<InteractionNearTree<
    CoordType>::NearTreeDataType>
{
  assert(radius <= r_cutoff_);
  return neartree_.find_in_annulus(epsilon_, radius, {r, 0}, epsilon_);
}

template<typename CoordType>
auto InteractionNearTree<CoordType>::shell(const PositionType& r, const CoordType& radius,
                                                 const CoordType& width) const -> std::vector<InteractionNearTree<
    CoordType>::NearTreeDataType>
{
  assert(radius <= r_cutoff_);
  return neartree_.find_in_annulus(radius - 0.5 * width, radius + 0.5 * width, {r, 0}, epsilon_);
}

template<typename CoordType>
void jams::InteractionNearTree<CoordType>::insert_sites(const std::vector<PositionType>& sites) {
// There are 6 surface normals for a parallelepiped but the parallel nature means each pair of opposite
  // surfaces has the same normal with opposite sign, so we only need calculate
  // 3 normals.
  normal_ac_ = normalize(cross(a_, c_));
  normal_cb_ = normalize(cross(c_, b_));
  normal_ba_ = normalize(cross(b_, a_));

  std::vector<NearTreeDataType> haloed_sites;

  for (auto i = 0; i < sites.size(); ++i) {
    const auto& r = sites[i];
    for (auto h = -num_image_cells_[0]; h < num_image_cells_[0] + 1; ++h) {
      for (auto k = -num_image_cells_[1]; k < num_image_cells_[1] + 1; ++k) {
        for (auto l = -num_image_cells_[2]; l < num_image_cells_[2] + 1; ++l) {
          auto r_image = r + (h * a_ + k * b_ + l * c_);

          const auto distance_to_plane = [&](const PositionType &normal,
                                             const PositionType &point_in_plane)->CoordType {
              return dot(normal, r_image - point_in_plane);
          };

          // These are the signed distances to the infinite plane aligned with
          // each face of the parallelepiped.
          std::array<CoordType, 6> distances = {
              distance_to_plane(normal_ac_, a_),
              distance_to_plane(-normal_ac_, b_),
              distance_to_plane(normal_cb_, c_),
              distance_to_plane(-normal_cb_, a_),
              distance_to_plane(normal_ba_, b_),
              distance_to_plane(-normal_ba_, c_)};

          // The surface normals all point outwards, so any point with
          // all negative distances must be inside the parallelepiped. We want
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
                  [&](const CoordType &x) {
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

template<typename CoordType>
int jams::InteractionNearTree<CoordType>::num_neighbours(const PositionType &r, const CoordType &radius) const {
  return neartree_.num_neighbours_in_radius(radius, {r, 0}, epsilon_) - 1;
}
}

#endif //JAMS_LATTICE_INTERACTION_NEAR_TREE_H
