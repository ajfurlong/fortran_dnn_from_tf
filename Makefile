# Define the Fortran compiler
FC = gfortran

# Define the directories
# Change the HDF5_DIR to match your install
HDF5_DIR = /opt/homebrew/Cellar/hdf5/1.14.3_1
INCLUDE_DIR = $(HDF5_DIR)/include
LIB_DIR = $(HDF5_DIR)/lib

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Define the flags
FFLAGS = -O2 -fdefault-real-8 -I$(INCLUDE_DIR) -J$(OBJ_DIR)
LDFLAGS = -L$(LIB_DIR) -lhdf5_fortran -lhdf5

# Define the source files
SRC = $(SRC_DIR)/read_hdf5_module.f90 $(SRC_DIR)/dnn_module.f90 \
      $(SRC_DIR)/metrics_module.f90 $(SRC_DIR)/main.f90

# Define the object files
OBJ = $(patsubst $(SRC_DIR)/%.f90, $(OBJ_DIR)/%.o, $(SRC))

# Define the executable
EXEC = $(BIN_DIR)/main

# Default rule
all: $(EXEC)

# Rule to link the executable
$(EXEC): $(OBJ)
	@mkdir -p $(BIN_DIR)
	$(FC) -o $@ $(OBJ) $(LDFLAGS)

# Rule to compile the Fortran source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.f90 | $(OBJ_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

# Create object directory if it doesn't exist
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# Clean rule
clean:
	rm -f $(OBJ) $(EXEC) $(OBJ_DIR)/*.mod

.PHONY: all clean