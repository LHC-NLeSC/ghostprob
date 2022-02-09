all	:	GhostDetection.o
	$(CC) $(CFLAGS) -o main GhostDetection.o

GhostDetection.o: GhostDetection.cpp
	$(CC) $(CFLAGS) -c GhostDetection.cpp


clean: rm main GhostDetection.o