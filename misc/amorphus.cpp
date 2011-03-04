#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include "maths.h"
#include "array3d.h"
#include "rand.h"

int main() {

  Random rand;
  rand.seed(time(NULL));

  // size
  int dim[3] = {25,25,25};
  
  std::vector<std::string> atomTypeNames;
  std::vector<double>       atomComposition;

  atomTypeNames.push_back("Fe");
  atomComposition.push_back(0.5);

  atomTypeNames.push_back("Gd");
  atomComposition.push_back(0.5);

  double Jij[2][2];
  Jij[0][0] = 4.5E-21;
  Jij[1][1] = 1.2E-21;
  Jij[0][1] = -4.5E-21;
  Jij[1][0] = -4.5E-21;



  Array3D<int> atomTypes(2*dim[0],2*dim[1],2*dim[2]);
  Array3D<int> atomNumbers(2*dim[0],2*dim[1],2*dim[2]);

  for(int i=0; i<2*dim[0]; ++i) {
    for(int j=0; j<2*dim[1]; ++j) {
      for(int k=0; k<2*dim[2]; ++k) {
        atomTypes(i,j,k) = -1;
        atomNumbers(i,j,k) = -1;
      }
    }
  }
  
  int atomCount=0;
  std::vector<int> typeCount(atomComposition.size(),0);

  for(int i=0; i<2*dim[0]; i++) {
    for(int j=0; j<2*dim[1]; j++) {
      for(int k=0; k<2*dim[2]; k++) {
        
        if(i%2 == 0){
          if(j%2 == 0){
            if(k%2 == 0){
              double rnum = rand.uniform();
              double compositionTotal = 0.0;
              for(int n=0; n<atomComposition.size(); ++n){
                compositionTotal+=atomComposition[n];
                if(rnum < compositionTotal){
                  atomTypes(i,j,k) = n;
                  atomNumbers(i,j,k) = atomCount;
                  atomCount++;
                  typeCount[n]++;
                  break;
                }
              } 
            }
          }else{
            if(k%2 != 0){
              double rnum = rand.uniform();
              double compositionTotal = 0.0;
              for(int n=0; n<atomComposition.size(); ++n){
                compositionTotal+=atomComposition[n];
                if(rnum < compositionTotal){
                  atomTypes(i,j,k) = n;
                  atomNumbers(i,j,k) = atomCount;
                  atomCount++;
                  typeCount[n]++;
                  break;
                }
              } 
            }
          }
        }else{
          if(j%2 != 0){
            if(k%2 == 0){
              double rnum = rand.uniform();
              double compositionTotal = 0.0;
              for(int n=0; n<atomComposition.size(); ++n){
                compositionTotal+=atomComposition[n];
                if(rnum < compositionTotal){
                  atomTypes(i,j,k) = n;
                  atomNumbers(i,j,k) = atomCount;
                  atomCount++;
                  typeCount[n]++;
                  break;
                }
              } 
            }
          }else{
            if(k%2 != 0){
              double rnum = rand.uniform();
              double compositionTotal = 0.0;
              for(int n=0; n<atomComposition.size(); ++n){
                compositionTotal+=atomComposition[n];
                if(rnum < compositionTotal){
                  atomTypes(i,j,k) = n;
                  atomNumbers(i,j,k) = atomCount;
                  atomCount++;
                  typeCount[n]++;
                  break;
                }
              } 
            }
          }

        }
      }
    }
  }

  std::cout<<atomCount<<"\n"<<std::endl;


  std::cout.precision(5);
  std::cout << std::fixed;

  std::cout<<"atoms = ("<<std::endl;
  for(int i=0; i<2*dim[0]; ++i) {
    for(int j=0; j<2*dim[1]; ++j) {
      for(int k=0; k<2*dim[2]; ++k) {
        if(atomNumbers(i,j,k) != -1) {
          int n = atomTypes(i,j,k);
          std::cout<<"( \""<<atomTypeNames[n]<<"\", ["<<i*0.5<<", "<<j*0.5<<", "<<k*0.5<<"] ),\n";
          //std::cout<<atomTypeNames[n]<<" "<<i*0.5<<" "<<j*0.5<<" "<<k*0.5<<std::endl;
        }
      }
    }
  }
  std::cout<<");"<<std::endl;
  
  std::cout<<"exchange = ("<<std::endl;
  for(int i=0; i<2*dim[0]; ++i) {
    for(int j=0; j<2*dim[1]; ++j) {
      for(int k=0; k<2*dim[2]; ++k) {
        int atom = atomNumbers(i,j,k);
        if(atom != -1) {
          int r[3] = {0, 1, 1};
          std::sort(r,r+3);
          do {
            int v[3] = {i,j,k};
            for(int n=0; n<3; ++n){
              v[n] = v[n] + r[n];
              v[n] = (2*dim[n]+v[n])%(2*dim[n]);
            }
            int nbr = atomNumbers(v[0],v[1],v[2]);
            // check ferrimagnetic coupling
            int atomType = atomTypes(i,j,k);
            int nbrType = atomTypes(v[0],v[1],v[2]);
            if(nbr != -1) {
              std::cout<<"( "<<atom+1<<", "<<nbr+1<<", [ "<<std::fixed<<r[0]*0.5<<", "<<r[1]*0.5<<", "<<r[2]*0.5<<"], ["<< std::scientific<< Jij[atomType][nbrType] <<"]),\n";
            }
          }while(next_point_symmetry(r));
        }
      }
    }
  }
  std::cout<<");"<<std::endl;





  for(int n=0;n<atomTypeNames.size();++n){
    std::cerr<<atomTypeNames[n]<<"("<<atomComposition[n]<<")"<<": "<<double(typeCount[n])/double(atomCount)<<std::endl;
  }

  return EXIT_SUCCESS;
}
