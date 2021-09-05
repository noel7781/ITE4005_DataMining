CC=g++
CXXFLAGS=-std=c++11
OBJS=assignment2.o
TARGET=assignment2.out

$(TARGET) : $(OBJS)
	$(CC) -o $@ $(OBJS)

assignment2.o: assignment2.cpp

clean:
	rm -f *.o
	rm -f $(TARGET)
