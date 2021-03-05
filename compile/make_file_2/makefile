#object files
OBJS=aux.o main.o

TARGET = myprog

all: myprog

myprog: $(OBJS)        
	$(CC) -o $(TARGET) $(OBJS)

clean:
	rm -rf *.o $(TARGET)
