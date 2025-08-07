# compiler
CC = gcc

# source files
SRC_1 = assignment_1.c
SRC_2 = omp_assignment_1.c

# compile-time flags
FLAG_1 = -O3
FLAG_2 = -O3 -fopenmp

# executable file name
MAIN_1 = assignment_1
MAIN_2 = omp_assignment_1

all: $(MAIN_1) $(MAIN_2)

$(MAIN_1):
	$(CC) $(FLAG_1) -o $(MAIN_1) $(SRC_1)

$(MAIN_2):
	$(CC) $(FLAG_2) -o $(MAIN_2) $(SRC_2)

clean:
	$(RM) $(MAIN_1) $(MAIN_2)