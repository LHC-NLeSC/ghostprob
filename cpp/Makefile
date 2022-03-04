all	:	GhostDetection.o
	$(CXX) $(CXXFLAGS) -o main GhostDetection.o $(LDFLAGS)

GhostDetection.o: GhostDetection.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c GhostDetection.cpp


clean: 
	rm main GhostDetection.o
