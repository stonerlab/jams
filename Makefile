CXX=g++
WARN=-Wall -Wextra -Weffc++ -Wold-style-cast -Wswitch-default \
		 -Wswitch-enum -Wfloat-equal -Werror=shadow -Winline \
		 -Wno-long-long -pedantic -fmessage-length=72
CFLAGS=-O2 $(WARN) -pipe -g -std=c++98 -j2
#CFLAGS=-O3 -Wall -pipe -j2 -DNDEBUG

INC=-I./include -I/opt/local/include
LDFLAGS=-g -L/opt/local/lib
#LDFLAGS=-L/opt/local/lib
LIBS=-lconfig++

OBJS=src/jams++.o \
		 src/output.o \
		 src/rand.o \
		 src/maths.o \
		 src/solver.o \
		 src/lattice.o \
		 src/sparsematrix.o \
		 src/heunllg.o

jams++: $(OBJS) 
	$(CXX) -o $@ $(CFLAGS) $(LDFLAGS) $(LIBS) $^

$(OBJS) : %.o : %.cc include/globals.h
	$(CXX) -c -o $@ $(INC) $(CFLAGS) $<

clean:
	rm -rf src/*.o jams++
