CXX=llvm-g++
WARN=-Wall -Wextra -Weffc++ -Wold-style-cast -Wswitch-default \
		 -Wswitch-enum -Wfloat-equal -Werror=shadow -Winline \
		 -Wno-long-long -pedantic 
CFLAGS=-O2 $(WARN) -std=c++98 -pipe -j2 -g
#CFLAGS=-O2 $(WARN) -pipe -DNDEBUG -fstrict-aliasing -funroll-all-loops -fprefetch-loop-arrays -std=c++98 -j2

INC=-I./include -isystem /opt/local/include -isystem /opt/local/include/metis
LDFLAGS=-g -L/opt/local/lib
#LDFLAGS=-L/opt/local/lib
LIBS=-lconfig++ -lmetis

OBJS=src/jams++.o \
		 src/output.o \
		 src/rand.o \
		 src/maths.o \
		 src/solver.o \
		 src/lattice.o \
		 src/heunllg.o

jams++: $(OBJS) 
	$(CXX) -o $@ $(CFLAGS) $(LDFLAGS) $(LIBS) $^ 

$(OBJS) : %.o : %.cc include/globals.h
	$(CXX) -c -o $@ $(INC) $(CFLAGS) $<

clean:
	rm -rf src/*.o jams++
