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
#   make BACKEND=hip lobpcg     # Build only LOBPCG with HIP
#   make clean                  # Clean build artifacts
# ==============================================================================

# Default backend
BACKEND ?= noacc

# Optional settings
CUDA_ARCH ?= sm_80
ROCM_PATH ?= /opt/rocm
DEBUG ?= 0

# ==============================================================================
# Directory structure
# ==============================================================================
SRC_DIR := src
INC_DIR := inc
HIP_DIR := hip
CUDA_DIR := cuda
OMP_DIR := omp
CPU_DIR := cpu
BUILD_DIR := build

# ==============================================================================
# Compiler definitions
# ==============================================================================
CC       := gcc
CXX      := g++
NVCC     := nvcc
HIPCC    := hipcc

# ==============================================================================
# Common settings
# ==============================================================================
LIBS     := -lm
INCLUDES := -I$(INC_DIR) -I$(CPU_DIR)

ifeq ($(DEBUG),1)
    OPT_FLAGS := -O0 -g
else
    OPT_FLAGS := -O3 -g
endif

# ==============================================================================
# Backend configuration
# ==============================================================================

ifeq ($(BACKEND),cuda)
    # --------------------------------------------------------------------------
    # CUDA Backend (NVIDIA GPUs)
    # --------------------------------------------------------------------------
    CONF_FLAGS := -DV100=0 -DNOACC=0 -DCUDA=1 -DOPENMP=0 -DHIP=0 -DUSE_FP64=1
    
    INCLUDES += -I$(CUDA_DIR)
    NVCC_FLAGS   := -arch=$(CUDA_ARCH)
    CPP_FLAGS    := -x cu
    CUDA_LIBS    := -lcusparse -lcublas -lcurand
    
    BACKEND_OBJS := $(BUILD_DIR)/cuda_blas.o $(BUILD_DIR)/devMem.o
    EXE_PREFIX   := lap_cuda
    
    define COMPILE_C
		$(NVCC) $(CONF_FLAGS) $(INCLUDES) $(NVCC_FLAGS) $(OPT_FLAGS) -o $@ -c $<
    endef
    
    define COMPILE_CPP
		$(NVCC) $(CONF_FLAGS) $(INCLUDES) $(NVCC_FLAGS) $(CPP_FLAGS) $(OPT_FLAGS) -o $@ -c $<
    endef
    
    define LINK
		$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS) $(CUDA_LIBS)
    endef

else ifeq ($(BACKEND),hip)
    # --------------------------------------------------------------------------
    # HIP Backend (AMD GPUs)
    # --------------------------------------------------------------------------
    CONF_FLAGS := -DV100=0 -DNOACC=0 -DCUDA=0 -DOPENMP=0 -DHIP=1 -DUSE_FP64=1
    
    INCLUDES += -I$(HIP_DIR) -I$(ROCM_PATH)/include/rocblas -I$(ROCM_PATH)/include/rocsparse
    HIP_FLAGS    := -D__HIP_PLATFORM_HCC__
    HIP_LIBS     := -L$(ROCM_PATH)/lib -lrocsparse -lrocblas -lrocsolver -lhiprand
    
    BACKEND_OBJS := $(BUILD_DIR)/hip_blas.o $(BUILD_DIR)/devMem.o
    EXE_PREFIX   := lap_hip
    
    define COMPILE_C
		$(CXX) $(CONF_FLAGS) $(INCLUDES) $(OPT_FLAGS) -o $@ -c $<
    endef
    
    define COMPILE_CPP
		$(HIPCC) $(CONF_FLAGS) $(HIP_FLAGS) $(INCLUDES) $(OPT_FLAGS) -o $@ -c $<
    endef
    
    define LINK
		$(HIPCC) $(HIP_FLAGS) -o $@ $^ $(LIBS) $(HIP_LIBS)
    endef

else ifeq ($(BACKEND),openmp)
    # --------------------------------------------------------------------------
    # OpenMP Backend (CPU with OpenMP - no GPU offloading)
    # --------------------------------------------------------------------------
    CONF_FLAGS := -DV100=0 -DNOACC=0 -DCUDA=0 -DOPENMP=1 -DHIP=0 -DUSE_FP64=1
    
    INCLUDES += -I$(OMP_DIR)
    OMP_FLAGS  := -fopenmp -std=c99
    OMP_LIBS   := -lgomp
    
    BACKEND_OBJS := $(BUILD_DIR)/openmp_blas.o $(BUILD_DIR)/devMem_cpu.o
    EXE_PREFIX   := lap_openmp
    
    define COMPILE_C
		$(CC) $(CONF_FLAGS) $(INCLUDES) $(OPT_FLAGS) $(OMP_FLAGS) -o $@ -c $<
    endef
    
    define LINK
		$(CC) $(CONF_FLAGS) $(OMP_FLAGS) -o $@ $^ $(LIBS) $(OMP_LIBS)
    endef

else ifeq ($(BACKEND),openmp_offload)
    # --------------------------------------------------------------------------
    # OpenMP Offload Backend (AMD GPU via OpenMP target offloading)
    # --------------------------------------------------------------------------
    # Requires: ROCm with amdclang
    # GPU architecture: gfx942 = MI300X, gfx90a = MI250X, gfx908 = MI100
    AMD_GPU_ARCH ?= gfx942
    
    CONF_FLAGS := -DV100=0 -DNOACC=0 -DCUDA=0 -DOPENMP=1 -DHIP=0 -DUSE_FP64=1 -DOMP_OFFLOAD=1
    
    AMDCLANG := $(ROCM_PATH)/bin/amdclang
    INCLUDES += -I$(OMP_DIR)
    OMP_OFFLOAD_FLAGS := -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
                         -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AMD_GPU_ARCH)
    OMP_LIBS := -L$(ROCM_PATH)/lib -lamdhip64
    
    BACKEND_OBJS := $(BUILD_DIR)/openmp_blas.o $(BUILD_DIR)/devMem_cpu.o
    EXE_PREFIX   := lap_omp_offload
    
    define COMPILE_C
		$(AMDCLANG) $(CONF_FLAGS) $(INCLUDES) $(OPT_FLAGS) $(OMP_OFFLOAD_FLAGS) -o $@ -c $<
    endef
    
    define LINK
		$(AMDCLANG) $(OMP_OFFLOAD_FLAGS) -o $@ $^ $(LIBS) $(OMP_LIBS)
    endef

else ifeq ($(BACKEND),noacc)
    # --------------------------------------------------------------------------
    # NOACC Backend (CPU only, no accelerators)
    # --------------------------------------------------------------------------
    CONF_FLAGS := -DV100=0 -DNOACC=1 -DCUDA=0 -DOPENMP=0 -DHIP=0 -DUSE_FP64=1
    
    CFLAGS     := -std=c99
    
    BACKEND_OBJS := $(BUILD_DIR)/devMem_cpu.o
    EXE_PREFIX   := lap_cpu
    
    define COMPILE_C
		$(CC) $(CONF_FLAGS) $(INCLUDES) $(OPT_FLAGS) $(CFLAGS) -o $@ -c $<
    endef
    
    define LINK
		$(CC) $(CONF_FLAGS) $(CFLAGS) -o $@ $^ $(LIBS)
    endef

else
    $(error Unknown BACKEND: $(BACKEND). Valid options: cuda, hip, openmp, openmp_offload, noacc)
endif

# ==============================================================================
# Object files
# ==============================================================================
# Core objects needed by all backends
CORE_OBJS := $(BUILD_DIR)/blas.o $(BUILD_DIR)/GS.o \
             $(BUILD_DIR)/it_jacobi.o $(BUILD_DIR)/line_jacobi.o $(BUILD_DIR)/prec.o \
             $(BUILD_DIR)/cg.o $(BUILD_DIR)/io_utils.o

# simple_blas.o is only needed for backends that don't have their own blas implementation
# OpenMP backends have their own blas in openmp_blas.o
# CUDA and HIP have their own blas implementations too
ifneq (,$(filter $(BACKEND),openmp openmp_offload cuda hip))
    COMMON_OBJS := $(CORE_OBJS)
else
    COMMON_OBJS := $(BUILD_DIR)/simple_blas.o $(CORE_OBJS)
endif

OBJS_LAPLACIAN := $(BACKEND_OBJS) $(COMMON_OBJS) $(BUILD_DIR)/cg_driver.o
OBJS_CG        := $(BACKEND_OBJS) $(COMMON_OBJS) $(BUILD_DIR)/cg_driver2.o
OBJS_SPMV      := $(BACKEND_OBJS) $(COMMON_OBJS) $(BUILD_DIR)/mm_driver.o
OBJS_LOBPCG    := $(BACKEND_OBJS) $(COMMON_OBJS) $(BUILD_DIR)/lobpcg.o $(BUILD_DIR)/lobpcg_driver.o

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

laplacian: $(BUILD_DIR) $(EXE_LAPLACIAN)

cg: $(BUILD_DIR) $(EXE_CG)

spmv: $(BUILD_DIR) $(EXE_SPMV)

lobpcg: $(BUILD_DIR) $(EXE_LOBPCG)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

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

# Compile source files from src/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(COMPILE_C)

# Compile CPU backend files
$(BUILD_DIR)/simple_blas.o: $(CPU_DIR)/simple_blas.c
	$(COMPILE_C)

$(BUILD_DIR)/devMem_cpu.o: $(CPU_DIR)/devMem.c
	$(COMPILE_C)

# Compile OpenMP backend files
$(BUILD_DIR)/openmp_blas.o: $(OMP_DIR)/openmp_blas.c
	$(COMPILE_C)

# Compile HIP backend files
$(BUILD_DIR)/hip_blas.o: $(HIP_DIR)/hip_blas.cpp
	$(COMPILE_CPP)

$(BUILD_DIR)/devMem.o: $(HIP_DIR)/devMem.cpp
	$(COMPILE_CPP)

# Compile CUDA backend files
# Note: devMem.cpp supports both CUDA and HIP via preprocessor
ifeq ($(BACKEND),cuda)
$(BUILD_DIR)/cuda_blas.o: $(CUDA_DIR)/cuda_blas.cu
	$(COMPILE_CPP)

$(BUILD_DIR)/devMem.o: $(HIP_DIR)/devMem.cpp
	$(COMPILE_CPP)
endif

# ==============================================================================
# Clean
# ==============================================================================
clean:
	rm -rf $(BUILD_DIR)
	rm -f lap_cuda_laplacian lap_cuda_cg lap_cuda_spmv lap_cuda_lobpcg
	rm -f lap_hip_laplacian lap_hip_cg lap_hip_spmv lap_hip_lobpcg
	rm -f lap_openmp_laplacian lap_openmp_cg lap_openmp_spmv lap_openmp_lobpcg
	rm -f lap_omp_offload_laplacian lap_omp_offload_cg lap_omp_offload_spmv lap_omp_offload_lobpcg
	rm -f lap_cpu_laplacian lap_cpu_cg lap_cpu_spmv lap_cpu_lobpcg

# ==============================================================================
# Help
# ==============================================================================
help:
	@echo "=============================================================================="
	@echo "Unified Makefile for LAP_PNNL"
	@echo "=============================================================================="
	@echo ""
	@echo "Directory structure:"
	@echo "  src/   - Common source files"
	@echo "  inc/   - Common header files"
	@echo "  hip/   - HIP backend (AMD GPUs)"
	@echo "  cuda/  - CUDA backend (NVIDIA GPUs)"
	@echo "  omp/   - OpenMP backend"
	@echo "  cpu/   - CPU-only backend"
	@echo "  build/ - Object files"
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
	@echo "Examples:"
	@echo "  make BACKEND=hip lobpcg    # Build LOBPCG with HIP"
	@echo "  make BACKEND=cuda          # Build all with CUDA"
	@echo "  make clean                 # Clean build artifacts"
