#ifndef __JAMS_CUDA_SPARSE_TYPES_H__
#define __JAMS_CUDA_SPARSE_TYPES_H__
typedef struct devDIA {
  int     *row;
  int     *col;
  float   *val;
  size_t  pitch;
  int     blocks;
} devDIA;

typedef struct devCSR {
  int     *pointers;
  int     *coords;
  float   *val;
  int     blocks;
} devCSR;
#endif
