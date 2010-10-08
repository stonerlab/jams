CXX=g++
CFLAGS=-O2 -Wall -pipe -g -j2
#CFLAGS=-O2 -Wall -pipe -j2 -NDEBUG

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

$(OBJS) : %.o : %.cc
	$(CXX) -c -o $@ $(INC) $(CFLAGS) $<

clean:
	rm -rf src/*.o jams++
