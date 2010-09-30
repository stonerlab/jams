CXX=g++
CFLAGS=-O2 -Wall -pipe -g -j2

INC=-I./include -I/opt/local/include
LDFLAGS=-g -L/opt/local/lib
LIBS=-lconfig++

OBJS=src/jams++.o \
		 src/output.o \
		 src/config.o 

jams++: $(OBJS)
	$(CXX) -o $@ $(CFLAGS) $(LDFLAGS) $(LIBS) $^

$(OBJS) : %.o : %.cc
	$(CXX) -c -o $@ $(INC) $(CFLAGS) $<

clean:
	rm -rf src/*.o jams++
