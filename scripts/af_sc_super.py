#!/usr/bin/python

import sys
import math
import numpy

def cshift(i,dim):
  return (i+dim)%dim

exchange = numpy.zeros((2,2))

exchange[0,0] = -6.26E-21
exchange[0,1] = -6.26E-21
exchange[1,0] = -6.26E-21
exchange[1,1] = -6.26E-21

lx=16
ly=16
lz=16

latticeType = numpy.zeros((lx,ly,lz))
latticeNum = numpy.zeros((lx,ly,lz))

AF2Count = 0
count = 0
for i in range(0,lx):
  for j in range(0,ly):
    for k in range(0,lz):
      latticeType[i,j,k] = 0
      latticeNum[i,j,k] = count
      count = count + 1
      if k%2 == 0:
        if i%2 == 0:
          if j%2 == 0:
            latticeType[i,j,k] = 1
        else:
          if j%2 == 1:
            latticeType[i,j,k] = 1
      else:
        if i%2 == 1:
          if j%2 == 0:
            latticeType[i,j,k] = 1
        else:
          if j%2 == 1:
            latticeType[i,j,k] = 1

posfile = file("af_pos.in","w")

for i in range(0,lx):
  for j in range (0,ly):
    for k in range (0,lz):
      if latticeType[i,j,k] == 0:
        print >> posfile, "AF1\t%2.5f\t%2.5f\t%2.5f" % (i/float(lx),j/float(ly),k/float(lz))
      elif latticeType[i,j,k] == 1:
        print >> posfile, "AF2\t%2.5f\t%2.5f\t%2.5f" % (i/float(lx),j/float(ly),k/float(lz))

posfile.close()

for i in range(0,lx):
  for j in range (0,ly):
    for k in range (0,lz):
      if latticeType[i,j,k] == 0:
        print "AF1\t%2.5f\t%2.5f\t%2.5f" % (i,j,k)

print "\n\n"

for i in range(0,lx):
  for j in range (0,ly):
    for k in range (0,lz):
      if latticeType[i,j,k] == 1:
        print "AF2\t%2.5f\t%2.5f\t%2.5f" % (i,j,k)

print "Conc:\t%1.4f AF1, %1.4f AF2" % (float(lx*ly*lz-AF2Count)/float(lx*ly*lz), float(AF2Count)/float(lx*ly*lz))

excfile = file("af_exc.in","w")
for i in range(0,lx):
  for j in range (0,ly):
    for k in range (0,lz):
# 1 (0,0,1)
      x=i
      y=j
      z=cshift(k+1,lz)
      n1 = latticeNum[i,j,k]
      n2 = latticeNum[x,y,z]
      t1 = latticeType[i,j,k]
      t2 = latticeType[x,y,z]
      print >> excfile, "%d\t%d\t%2.2f\t%2.2f\t%2.2f\t%2.4e" % (n1+1, n2+1, 0.0, 0.0, 1.0, exchange[t1,t2])

# 2 (0,0,-1)
      x=i
      y=j
      z=cshift(k-1,lz)
      n1 = latticeNum[i,j,k]
      n2 = latticeNum[x,y,z]
      t1 = latticeType[i,j,k]
      t2 = latticeType[x,y,z]
      print >> excfile, "%d\t%d\t%2.2f\t%2.2f\t%2.2f\t%2.4e" % (n1+1, n2+1, 0.0, 0.0, -1.0, exchange[t1,t2])

# 3 (0,1,0)
      x=i
      y=cshift(j+1,ly)
      z=k
      n1 = latticeNum[i,j,k]
      n2 = latticeNum[x,y,z]
      t1 = latticeType[i,j,k]
      t2 = latticeType[x,y,z]
      print >> excfile, "%d\t%d\t%2.2f\t%2.2f\t%2.2f\t%2.4e" % (n1+1, n2+1, 0.0, 1.0, 0.0, exchange[t1,t2])

# 4 (0,-1,0)
      x=i
      y=cshift(j-1,ly)
      z=k
      n1 = latticeNum[i,j,k]
      n2 = latticeNum[x,y,z]
      t1 = latticeType[i,j,k]
      t2 = latticeType[x,y,z]
      print >> excfile, "%d\t%d\t%2.2f\t%2.2f\t%2.2f\t%2.4e" % (n1+1, n2+1, 0.0, -1.0, 0.0, exchange[t1,t2])

# 5 (1,0,0)
      x=cshift(i+1,lx)
      y=j
      z=k
      n1 = latticeNum[i,j,k]
      n2 = latticeNum[x,y,z]
      t1 = latticeType[i,j,k]
      t2 = latticeType[x,y,z]
      print >> excfile, "%d\t%d\t%2.2f\t%2.2f\t%2.2f\t%2.4e" % (n1+1, n2+1, 1.0, 0.0, 0.0, exchange[t1,t2])

# 6 (-1,0,0)
      x=cshift(i-1,lx)
      y=j
      z=k
      n1 = latticeNum[i,j,k]
      n2 = latticeNum[x,y,z]
      t1 = latticeType[i,j,k]
      t2 = latticeType[x,y,z]
      print >> excfile, "%d\t%d\t%2.2f\t%2.2f\t%2.2f\t%2.4e" % (n1+1, n2+1, -1.0, 0.0, 0.0, exchange[t1,t2])

excfile.close()
