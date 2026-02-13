#ifndef JAMS_LATTICE_INTERACTION_NEARTREE_H
#define JAMS_LATTICE_INTERACTION_NEARTREE_H

#include <jams/containers/vec3.h>
#include <jams/containers/neartree.h>
#include <functional>

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
    class InteractionNearTree {
    public:
        using NearTreeDataType = std::pair<Vec3, int>;
        using NearTreeFunctorType = std::function<double(
            const NearTreeDataType &a, const NearTreeDataType &b)>;
        using NearTreeType = jams::NearTree<NearTreeDataType, NearTreeFunctorType>;

        InteractionNearTree(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3b& pbc,
                            const double& r_cutoff, const double& epsilon);

        // Insert a vector of site positions. The index will be numbered from
        // zero to sites.size() - 1.
        // Calling this method replaces any previously inserted sites.
        void insert_sites(const std::vector<Vec3>& sites);

        // Returns the number of sites inside the neartree.
        inline std::size_t size() const { return neartree_.size(); };

        // Returns the memory in bytes of the neartree. We ignore the small
        // memory footprint of this InteractionNearTree wrapper.
        inline std::size_t memory() const { return neartree_.memory(); };

        // Return a list of neighbours within radius of point r.
        // The radius must be less than the r_cutoff of the InteractionNearTree
        std::vector<NearTreeDataType>
        neighbours(const Vec3 &r, const double &radius) const;

        std::vector<NearTreeDataType>
        shell(const Vec3 &r, const double &radius, const double& width) const;

        // Return a the number of neighbours within radius of point r.
        // We assume the point r is on a site as so we subtract 1 from the number
        // of points with the sphere.
        // The radius must be less than the r_cutoff of the InteractionNearTree
        int num_neighbours(const Vec3 &r, const double &radius) const;

    private:

        double r_cutoff_;
        double epsilon_;

        Vec3 a_;
        Vec3 b_;
        Vec3 c_;

        Vec3 normal_ac_;
        Vec3 normal_cb_;
        Vec3 normal_ba_;

        Vec3i num_image_cells_ = {0, 0, 0};

        NearTreeType neartree_;
    };
}

#endif //JAMS_LATTICE_INTERACTION_NEAR_TREE_H
