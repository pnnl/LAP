# ==============================================================================
# Unified Makefile for LAP_PNNL
# ==============================================================================
# Usage:
#   make BACKEND=<backend> [TARGET]
#
# Backends:
#   cuda   - NVIDIA GPU using CUDA/cuBLAS/cuSPARSE
#   hip    - AMD GPU using HIP/rocBLAS/rocSPARSE
#   openmp - CPU with OpenMP parallelization
#   noacc  - CPU only, no accelerators (serial)
#
# Targets:
#   all       - Build all executables (laplacian, cg, spmv, lobpcg)
#   laplacian - Build laplacian driver
#   cg        - Build CG driver
#   spmv      - Build SpMV driver
#   lobpcg    - Build LOBPCG eigenvalue solver
#   clean     - Remove object files and executables
#
# Examples:
#   make BACKEND=cuda           # Build all with CUDA
#   make BACKEND=hip            # Build all with HIP
#   make BACKEND=openmp         # Build all with OpenMP
#   make BACKEND=noacc          # Build all without accelerators
#   make BACKEND=cuda cg        # Build only CG driver with CUDA
#   make BACKEND=cuda lobpcg    # Build only LOBPCG with CUDA
#   make BACKEND=hip clean      # Clean HIP build artifacts
#
# Optional variables:
#   CUDA_ARCH  - CUDA architecture (default: sm_80)
#   ROCM_PATH  - Path to ROCm installation (default: /opt/rocm)
#   DEBUG      - Set to 1 for debug build (default: 0)
# ==============================================================================

# Default backend
BACKEND ?= noacc

# Optional settings
CUDA_ARCH ?= sm_80
ROCM_PATH ?= /opt/rocm
DEBUG ?= 0

# ==============================================================================
# Compiler definitions
# ==============================================================================
CC       := gcc
CXX      := g++
NVCC     := nvcc
HIPCC    := hipcc
NVC      := nvc

# ==============================================================================
# Common settings
# ==============================================================================
LIBS     := -lm

ifeq ($(DEBUG),1)
    OPT_FLAGS := -O0 -g
else
    OPT_FLAGS := -O3 -g
endif

# ==============================================================================
# Source files
# ==============================================================================
# Common source files (C)
SRC_COMMON := simple_blas.c blas.c GS.c it_jacobi.c line_jacobi.c prec.c cg.c io_utils.c

# Driver source files
SRC_DRIVER_LAPLACIAN := cg_driver.c
SRC_DRIVER_CG        := cg_driver2.c
SRC_DRIVER_SPMV      := mm_driver.c
SRC_DRIVER_LOBPCG    := lobpcg_driver.c

# LOBPCG specific source
SRC_LOBPCG           := lobpcg.c

# Backend-specific source files
SRC_CUDA   := cuda_blas.cu devMem.cpp
SRC_HIP    := hip_blas.cpp devMem.cpp
SRC_OPENMP := openmp_blas.c
SRC_NOACC  :=

# ==============================================================================
# Backend configuration
# ==============================================================================

ifeq ($(BACKEND),cuda)
    # --------------------------------------------------------------------------
    # CUDA Backend (NVIDIA GPUs)
    # --------------------------------------------------------------------------
    CONF_FLAGS := -DV100=0 -DNOACC=0 -DCUDA=1 -DOPENMP=0 -DHIP=0 -DUSE_FP64=1
    
    COMPILER     := $(NVCC)
    NVCC_FLAGS   := -arch=$(CUDA_ARCH)
    CPP_FLAGS    := -x cu
    CUDA_LIBS    := -lcusparse -lcublas
    
    BACKEND_OBJS := cuda_blas.o devMem.o
    COMMON_OBJS  := simple_blas.o blas.o GS.o it_jacobi.o line_jacobi.o prec.o cg.o io_utils.o
    
    EXE_PREFIX   := lap_cuda
    
    # Compilation rules for CUDA
    define COMPILE_C
		$(NVCC) $(CONF_FLAGS) $(NVCC_FLAGS) $(OPT_FLAGS) -o $@ -c $<
    endef
    
    define COMPILE_CU
		$(NVCC) $(CONF_FLAGS) $(NVCC_FLAGS) $(OPT_FLAGS) $(CUDA_LIBS) -o $@ -c $<
    endef
    
    define COMPILE_CPP
		$(NVCC) $(CONF_FLAGS) $(NVCC_FLAGS) $(CPP_FLAGS) $(OPT_FLAGS) $(CUDA_LIBS) -o $@ -c $<
    endef
    
    define LINK
		$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS) $(CUDA_LIBS)
    endef

else ifeq ($(BACKEND),hip)
    # --------------------------------------------------------------------------
    # HIP Backend (AMD GPUs)
    # --------------------------------------------------------------------------
    CONF_FLAGS := -DV100=0 -DNOACC=0 -DCUDA=0 -DOPENMP=0 -DHIP=1 -DUSE_FP64=1
    
    COMPILER     := $(HIPCC)
    HIP_FLAGS    := -D__HIP_PLATFORM_HCC__
    HIP_INCLUDES := -I$(ROCM_PATH)/include/rocblas -I$(ROCM_PATH)/include/rocsparse
    HIP_LIBS     := -L$(ROCM_PATH)/lib -lrocsparse -lrocblas
    
    BACKEND_OBJS := hip_blas.o devMem.o
    COMMON_OBJS  := simple_blas.o blas.o GS.o it_jacobi.o line_jacobi.o prec.o cg.o io_utils.o
    
    EXE_PREFIX   := lap_hip
    
    # Compilation rules for HIP
    define COMPILE_C
		$(CXX) $(CONF_FLAGS) $(OPT_FLAGS) -o $@ -c $<
    endef
    
    define COMPILE_CPP
		$(HIPCC) $(CONF_FLAGS) $(HIP_FLAGS) $(HIP_INCLUDES) $(OPT_FLAGS) -o $@ -c $<
    endef
    
    define LINK
		$(HIPCC) $(HIP_FLAGS) -o $@ $^ $(LIBS) $(HIP_LIBS)
    endef

else ifeq ($(BACKEND),openmp)
    # --------------------------------------------------------------------------
    # OpenMP Backend (CPU with OpenMP)
    # --------------------------------------------------------------------------
    CONF_FLAGS := -DV100=0 -DNOACC=0 -DCUDA=0 -DOPENMP=1 -DHIP=0 -DUSE_FP64=1
    
    COMPILER   := $(CC)
    OMP_FLAGS  := -fopenmp -std=c99
    OMP_LIBS   := -lgomp
    
    BACKEND_OBJS := openmp_blas.o
    COMMON_OBJS  := blas.o GS.o it_jacobi.o line_jacobi.o prec.o cg.o io_utils.o
    
    EXE_PREFIX   := lap_openmp
    
    # Compilation rules for OpenMP
    define COMPILE_C
		$(CC) $(CONF_FLAGS) $(OPT_FLAGS) $(OMP_FLAGS) -o $@ -c $<
    endef
    
    define LINK
		$(CC) $(CONF_FLAGS) $(OMP_FLAGS) -o $@ $^ $(LIBS) $(OMP_LIBS)
    endef

else ifeq ($(BACKEND),noacc)
    # --------------------------------------------------------------------------
    # NOACC Backend (CPU only, no accelerators)
    # --------------------------------------------------------------------------
    CONF_FLAGS := -DV100=0 -DNOACC=1 -DCUDA=0 -DOPENMP=0 -DHIP=0 -DUSE_FP64=1
    
    COMPILER   := $(CC)
    CFLAGS     := -std=c99
    
    BACKEND_OBJS :=
    COMMON_OBJS  := simple_blas.o blas.o GS.o it_jacobi.o line_jacobi.o prec.o cg.o io_utils.o
    
    EXE_PREFIX   := lap_cpu
    
    # Compilation rules for NOACC
    define COMPILE_C
		$(CC) $(CONF_FLAGS) $(OPT_FLAGS) $(CFLAGS) -o $@ -c $<
    endef
    
    define LINK
		$(CC) $(CONF_FLAGS) $(CFLAGS) -o $@ $^ $(LIBS)
    endef

else
    $(error Unknown BACKEND: $(BACKEND). Valid options: cuda, hip, openmp, noacc)
endif

# ==============================================================================
# Object files
# ==============================================================================
OBJS_LAPLACIAN := $(BACKEND_OBJS) $(COMMON_OBJS) cg_driver.o
OBJS_CG        := $(BACKEND_OBJS) $(COMMON_OBJS) cg_driver2.o
OBJS_SPMV      := $(BACKEND_OBJS) $(COMMON_OBJS) mm_driver.o
OBJS_LOBPCG    := $(BACKEND_OBJS) $(COMMON_OBJS) lobpcg.o lobpcg_driver.o

# ==============================================================================
# Executable names
# ==============================================================================
EXE_LAPLACIAN := $(EXE_PREFIX)_laplacian
EXE_CG        := $(EXE_PREFIX)_cg
EXE_SPMV      := $(EXE_PREFIX)_spmv
EXE_LOBPCG    := $(EXE_PREFIX)_lobpcg

# ==============================================================================
# Targets
# ==============================================================================
.PHONY: all laplacian cg spmv lobpcg clean help

all: laplacian cg spmv lobpcg

laplacian: $(EXE_LAPLACIAN)

cg: $(EXE_CG)

spmv: $(EXE_SPMV)

lobpcg: $(EXE_LOBPCG)

# ==============================================================================
# Build rules
# ==============================================================================

# Link executables
$(EXE_LAPLACIAN): $(OBJS_LAPLACIAN)
	$(LINK)

$(EXE_CG): $(OBJS_CG)
	$(LINK)

$(EXE_SPMV): $(OBJS_SPMV)
	$(LINK)

$(EXE_LOBPCG): $(OBJS_LOBPCG)
	$(LINK)

# Compile C source files
%.o: %.c
	$(COMPILE_C)

# Compile CUDA source files (only for CUDA backend)
ifeq ($(BACKEND),cuda)
%.o: %.cu
	$(COMPILE_CU)

%.o: %.cpp
	$(COMPILE_CPP)
endif

# Compile C++ source files (for HIP backend)
ifeq ($(BACKEND),hip)
%.o: %.cpp
	$(COMPILE_CPP)
endif

# ==============================================================================
# Clean
# ==============================================================================
clean:
	rm -f *.o
	rm -f lap_cuda_laplacian lap_cuda_cg lap_cuda_spmv lap_cuda_lobpcg
	rm -f lap_hip_laplacian lap_hip_cg lap_hip_spmv lap_hip_lobpcg
	rm -f lap_openmp_laplacian lap_openmp_cg lap_openmp_spmv lap_openmp_lobpcg
	rm -f lap_cpu_laplacian lap_cpu_cg lap_cpu_spmv lap_cpu_lobpcg

# ==============================================================================
# Help
# ==============================================================================
help:
	@echo "=============================================================================="
	@echo "Unified Makefile for LAP_PNNL"
	@echo "=============================================================================="
	@echo ""
	@echo "Usage: make BACKEND=<backend> [TARGET]"
	@echo ""
	@echo "Backends:"
	@echo "  cuda   - NVIDIA GPU using CUDA/cuBLAS/cuSPARSE"
	@echo "  hip    - AMD GPU using HIP/rocBLAS/rocSPARSE"
	@echo "  openmp - CPU with OpenMP parallelization"
	@echo "  noacc  - CPU only, no accelerators (serial)"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build all executables (default)"
	@echo "  laplacian - Build laplacian driver only"
	@echo "  cg        - Build CG driver only"
	@echo "  spmv      - Build SpMV driver only"
	@echo "  lobpcg    - Build LOBPCG eigenvalue solver only"
	@echo "  clean     - Remove all object files and executables"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Optional variables:"
	@echo "  CUDA_ARCH=<arch>  - CUDA architecture (default: sm_80)"
	@echo "  ROCM_PATH=<path>  - Path to ROCm installation (default: /opt/rocm)"
	@echo "  DEBUG=1           - Enable debug build (default: 0)"
	@echo ""
	@echo "Examples:"
	@echo "  make BACKEND=cuda                    # Build all with CUDA"
	@echo "  make BACKEND=hip                     # Build all with HIP"
	@echo "  make BACKEND=openmp                  # Build all with OpenMP"
	@echo "  make BACKEND=noacc                   # Build all without accelerators"
	@echo "  make BACKEND=cuda cg                 # Build only CG driver with CUDA"
	@echo "  make BACKEND=hip lobpcg             # Build only LOBPCG with HIP"
	@echo "  make BACKEND=cuda CUDA_ARCH=sm_70    # Build with different CUDA arch"
	@echo "  make BACKEND=hip ROCM_PATH=/opt/rocm-5.0  # Use specific ROCm path"
	@echo "  make clean                           # Clean all build artifacts"
	@echo ""
