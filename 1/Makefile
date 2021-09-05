CC=g++
CXXFLAGS=-std=c++11
OBJS=assignment1.o
TARGET=assignment1

$(TARGET) : $(OBJS)
	$(CC) -o $@ $(OBJS)

assignment1.o: assignment1.cpp

clean:
	rm -f *.o
	rm -f $(TARGET)
