#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <map>

#include <libconfig.h++>

#include "geometry.h"
#include "globals.h"
#include "utils.h"


void invert_matrix(float in[3][3], float out[3][3]) {
  float det = in[0][0]*(in[1][1]*in[2][2]-in[1][2]*in[2][1])
             +in[0][1]*(in[1][2]*in[2][0]-in[1][0]*in[2][2])
             +in[0][2]*(in[1][0]*in[2][1]-in[1][1]*in[2][0]);
 
  det = 1.0/det;

  out[0][0] = det*(in[1][1]*in[2][2]-in[1][2]*in[2][1]);
  out[0][1] = det*(in[0][2]*in[2][1]-in[0][1]*in[2][2]);
  out[0][2] = det*(in[0][1]*in[1][2]-in[0][2]*in[1][1]);

  out[1][0] = det*(in[1][2]*in[2][0]-in[1][0]*in[2][2]);
  out[1][1] = det*(in[0][0]*in[1][1]-in[0][2]*in[2][0]);
  out[1][2] = det*(in[0][2]*in[1][0]-in[0][0]*in[1][2]);

  out[2][0] = det*(in[1][0]*in[2][1]-in[1][1]*in[2][0]);
  out[2][1] = det*(in[0][1]*in[2][0]-in[0][0]*in[2][1]);
  out[2][2] = det*(in[0][0]*in[1][1]-in[0][1]*in[1][0]);
}

namespace {
  const float r_eps = 1.0e-5;
  bool fzero(float x, float y) {
    return (std::abs(y-x) < r_eps);    
  }
} // anon namespace 

void Geometry::readFromConfig()
{
  using namespace globals;

  try {
    const libconfig::Setting& basis = config.lookup("lattice.unitcell.basis");

    float lvec[3][3];
    float lvec_inv[3][3];

    for(int i=0; i<3; ++i) {
      lvec[0][i] = basis[0][i];
      lvec[1][i] = basis[1][i];
      lvec[2][i] = basis[2][i];

//      a0[i] = basis[0][i];
//      a1[i] = basis[1][i];
//      a2[i] = basis[2][i];
    }

    invert_matrix(lvec,lvec_inv);

    output.write("\nInverse lattice vectors\n");
    output.write("%f\t%f\t%f\n",lvec[0][0],lvec[0][1],lvec[0][2]);
    output.write("%f\t%f\t%f\n",lvec[1][0],lvec[1][1],lvec[1][2]);
    output.write("%f\t%f\t%f\n",lvec[2][0],lvec[2][1],lvec[2][2]);

    output.write("\nLattice translation vector\n");
    output.write("---------------------------\n");
    output.write("%-3.6f\t%-3.6f\t%-3.6f\n",a0[0],a0[1],a0[2]);
    output.write("%-3.6f\t%-3.6f\t%-3.6f\n",a1[0],a1[1],a1[2]);
    output.write("%-3.6f\t%-3.6f\t%-3.6f\n",a2[0],a2[1],a2[2]);

    const libconfig::Setting& atoms = config.lookup("lattice.unitcell.atoms");
    int natoms = atoms.getLength();              
                
    output.write("\nAtoms in unit cell\n");
    output.write("------------------\n");
    
    std::map<std::string,int>::iterator tit;
    int tcount=0;

    for (int n=0; n<natoms; ++n) {
      std::string t=atoms[n][0];
      float x=atoms[n][1][0], y=atoms[n][1][1], z=atoms[n][1][2];
      output.write("%s\t%-3.6f\t%-3.6f\t%-3.6f\n",t.c_str(),x,y,z);
      tit = atom_type_map.find(t);
      if (tit == atom_type_map.end()) {
        atom_type_map.insert( std::pair<std::string,int>(t,tcount) );
        tcount++;
      }
    }

    output.write("\nUnique types found: %d\n",tcount);

    lattice_size[0] = config.lookup("lattice.size.[0]");
    lattice_size[1] = config.lookup("lattice.size.[1]");
    lattice_size[2] = config.lookup("lattice.size.[2]");

    output.write("\nLattice size:\t%d\t%d\t%d\n",lattice_size[0],lattice_size[1],lattice_size[2]);
            
    for (int n=0; n<natoms; ++n) {
      const vec3<float> u (atoms[n][1][0], atoms[n][1][1], atoms[n][1][2]);
      const double nx = lvec_inv[0][0]*u[0]+lvec_inv[1][0]*u[1]+lvec_inv[2][0]*u[2];
      const double ny = lvec_inv[0][1]*u[0]+lvec_inv[1][1]*u[1]+lvec_inv[2][1]*u[2];
      const double nz = lvec_inv[0][2]*u[0]+lvec_inv[1][2]*u[1]+lvec_inv[2][2]*u[2];
      output.write("Integer coords: %f %f %f\n",nx,ny,nz);
    }

    for (int i=0; i<lattice_size[0]; ++i) {
      for (int j=0; j<lattice_size[1]; ++j) {
        for (int k=0; k<lattice_size[2]; ++k) {
          for (int n=0; n<natoms; ++n) {
            const std::string t = atoms[n][0];
            atom_type.push_back(atom_type_map[t]);

            const vec3<float> u (atoms[n][1][0], atoms[n][1][1], atoms[n][1][2]);

            const vec3<float> b( (i+u[0])*a0[0] + (j+u[0])*a1[0] + (k+u[0])*a2[0],
                               (i+u[1])*a0[1] + (j+u[1])*a1[1] + (k+u[1])*a2[1],
                               (i+u[2])*a0[2] + (j+u[2])*a1[2] + (k+u[2])*a2[2]);

            r.push_back(b);
          }
        }
      }
    }

    nspins = r.size();
    
    assert(nspins > 0);

    output.write("\nAtoms created: %d\n",nspins);

    s.resize(nspins);
    h.resize(nspins);
    w.resize(nspins);

    mus.resize(nspins);
    gyro.resize(nspins);
    alpha.resize(nspins);


    const libconfig::Setting& mat = config.lookup("materials");


    assert(static_cast<unsigned int>(mat.getLength()) >= atom_type_map.size());

    for (int i=0; i<nspins; ++i) {
      int a = atom_type[i];
      mus(i) = mat[a]["moment"];
      alpha(i) = mat[a]["alpha"];
      const double g = mat[a]["gyro"];
      gyro(i) = g/((1.0+alpha(i)*alpha(i))*mus(i));

      // ensure initial spins are normalised
      const vec3<double> spin  (mat[a]["spin"][0],
                             mat[a]["spin"][1],
                             mat[a]["spin"][2]);

      const double norm = 1.0/sqrt(spin[0]*spin[0]+spin[1]*spin[1]+spin[2]*spin[2]);

      s.x(i) = spin[0]*norm;
      s.y(i) = spin[1]*norm;
      s.z(i) = spin[2]*norm;
    }
    
    bool jsym = config.lookup("lattice.jsym");

    if (jsym == true) {
      output.write("\nInteraction point symmetry operations ON\n");
    } else {
      output.write("\nInteraction point symmetry operations OFF\n");
    }

    const libconfig::Setting& inter = config.lookup("exchange");
    const int ninter = inter.getLength();

    std::vector<float>          ir(ninter);
    std::vector< vec3<float> >   ii(ninter);

    std::vector<double> jxx(ninter);
    std::vector<double> jxy(ninter);
    std::vector<double> jxz(ninter);

    std::vector<double> jyx(ninter);
    std::vector<double> jyy(ninter);
    std::vector<double> jyz(ninter);

    std::vector<double> jzx(ninter);
    std::vector<double> jzy(ninter);
    std::vector<double> jzz(ninter);

    std::vector<int> itype1(ninter);
    std::vector<int> itype2(ninter);

    printf("\nInteraction vectors read: %d\n",ninter);

    for (int i=0; i<ninter; ++i) {
      const std::string st1 = inter[i][0];
      const std::string st2 = inter[i][1];

      const int t1 = atom_type_map[st1];
      const int t2 = atom_type_map[st2];
      itype1[i] = (t1);
      itype2[i] = (t2);

      const vec3<float> v(inter[i][2][0], inter[i][2][1], inter[i][2][2]);
      ii[i] = v;
      ir[i] = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

      jxx[i] = inter[i][3][0][0];
      jxy[i] = inter[i][3][0][1];
      jxz[i] = inter[i][3][0][2];

      jyx[i] = inter[i][3][1][0];
      jyy[i] = inter[i][3][1][1];
      jyz[i] = inter[i][3][1][2];

      jzx[i] = inter[i][3][2][0];
      jzy[i] = inter[i][3][2][1];
      jzz[i] = inter[i][3][2][2];
    }

    float rcutsq = *(std::max_element(ir.begin(),ir.end()));
    output.write("\nMaximum interaction range: %8.6f\n",sqrt(rcutsq));

    // exchange is mutually symmetric so loop j > i
    
    // guess the number of nonzeros of sparse matrix as the max possible
    // interactions for a simple cubic lattice all spins. This will
    // hopefully be an overestimate to avoid lots of vector resizing in
    // the sparse matrix routine.
    unsigned int nnz_guess = nspins*ninter*6;
    output.write("Number of nonzero elements guess: %i\n",nnz_guess);
    
    jij = new SparseMatrix(3*nspins,3*nspins,9*nnz_guess);

    unsigned int xoffset = 0;
    unsigned int yoffset = nspins;
    unsigned int zoffset = 2*nspins;

    int count=0;
    for (int i=0; i<nspins; ++i) {
      // TODO: insert anisotropy for i = j
      for (int j=i+1; j<nspins; ++j) {

        const vec3<float> dr((r[j][0]-r[i][0]), (r[j][1]-r[i][1]), (r[j][2]-r[i][2]));
        const float rsq = dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2];

        assert( (rcutsq+r_eps) > 0.0 );
        if ( rsq < (rcutsq+r_eps) ) {
          std::vector<float>::iterator it=ir.begin();
          while(it != ir.end()){  // make sure we get any degenerate distances
            it = std::find_first_of(it,ir.end(),&rsq,(&rsq)+1,fzero);

            if ((it!=ir.end())) {
                int idx = it-ir.begin();
                assert(static_cast<unsigned int>(idx) < itype1.size());
              if ((itype1[idx] == atom_type[i]) && (itype2[idx] == atom_type[j])) {

                if (jsym==true) {
                  float pts[3] = {ii[idx][0],ii[idx][1],ii[idx][2]};
                  std::sort(pts,pts+3);
                  do {
                    if (fzero(dr[0],pts[0]) && fzero(dr[1],pts[1]) && fzero(dr[2],pts[2])) {
                      count++;
                      /////////////////////////////////////////
                      if (std::abs(jxx[idx]) > energy_cutoff) {
                        jij->insert(i+xoffset,j+xoffset,jxx[idx]);
                      }
                      if (std::abs(jxy[idx]) > energy_cutoff) {
                        jij->insert(i+xoffset,j+yoffset,jxy[idx]);
                      }
                      if (std::abs(jxz[idx]) > energy_cutoff) {
                        jij->insert(i+xoffset,j+zoffset,jxz[idx]);
                      }
                      /////////////////////////////////////////
                      if (std::abs(jyx[idx]) > energy_cutoff) {
                        jij->insert(i+yoffset,j+xoffset,jyx[idx]);
                      }
                      if (std::abs(jyy[idx]) > energy_cutoff) {
                        jij->insert(i+yoffset,j+yoffset,jyy[idx]);
                      }
                      if (std::abs(jyz[idx]) > energy_cutoff) {
                        jij->insert(i+yoffset,j+zoffset,jyz[idx]);
                      }
                      /////////////////////////////////////////
                      if (std::abs(jzx[idx]) > energy_cutoff) {
                        jij->insert(i+zoffset,j+xoffset,jzx[idx]);
                      }
                      if (std::abs(jzy[idx]) > energy_cutoff) {
                        jij->insert(i+zoffset,j+yoffset,jzy[idx]);
                      }
                      if (std::abs(jzz[idx]) > energy_cutoff) {
                        jij->insert(i+zoffset,j+zoffset,jzz[idx]);
                      }
                      /////////////////////////////////////////

                      break; // don't need to check more symmetries
                    }
                  } while (next_point_symmetry(pts));
                } else {
                  if (fzero(dr[0],ii[idx][0]) && fzero(dr[1],ii[idx][1]) && fzero(dr[2],ii[idx][2])) {
                      count++;
                    // insert interactions
                  }
                }  // if jsym
              }  // if typecheck

              it++; // increase for next time round
            }  // if range
          }  //while it != ir.end()
        }  // if rsq < rcutsq
      }  // for j
    }  // for i
    output.write("\nInteractions: %d\n",count);

    output.write("Converting Jij matrix to CSR format\n");
    jij->coocsr();
    output.write("SparseMatrix memory footprint ~%5.2f MB\n",jij->memorySize());


  }
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_error("Setting '%s' not found",nfex.getPath());
  }


}                  
                   
