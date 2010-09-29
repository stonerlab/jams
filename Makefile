CXX=g++
CFLAGS=-O2 -Wall -pipe -g -j2

INC=-I./include
LDFLAGS=-g

OBJS=src/jams++.o \
		 src/output.o

jams++: $(OBJS)
	$(CXX) -o $@ $(CFLAGS) $(LDFLAGS) $(LIBS) $^

$(OBJS) : %.o : %.cc
	$(CXX) -c -o $@ $(INC) $(CFLAGS) $<

clean:
	rm -rf src/*.o jams++
