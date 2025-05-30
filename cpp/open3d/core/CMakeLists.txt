open3d_ispc_add_library(core OBJECT)

target_sources(core PRIVATE
    AdvancedIndexing.cpp
    CUDAUtils.cpp
    Device.cpp
    Dtype.cpp
    Indexer.cpp
    MemoryManager.cpp
    MemoryManagerCached.cpp
    MemoryManagerCPU.cpp
    MemoryManagerStatistic.cpp
    ShapeUtil.cpp
    SizeVector.cpp
    SmallVector.cpp
    Tensor.cpp
    TensorCheck.cpp
    TensorFunction.cpp
    TensorKey.cpp
    TensorList.cpp
)

# Compile regardless BUILD_SYCL_MODULE == ON or OFF.
open3d_sycl_target_sources(core PRIVATE
    EigenConverter.cpp
    SYCLUtils.cpp
)

# Compile only when BUILD_SYCL_MODULE == ON.
if (BUILD_SYCL_MODULE)
    open3d_sycl_target_sources(core PRIVATE
        MemoryManagerSYCL.cpp
        SYCLContext.cpp
    )
endif()

target_sources(core PRIVATE
    hashmap/CPU/CPUHashBackendBuffer.cpp
    hashmap/CPU/CreateCPUHashBackend.cpp
    hashmap/DeviceHashBackend.cpp
    hashmap/HashBackendBuffer.cpp
    hashmap/HashMap.cpp
    hashmap/HashSet.cpp
    kernel/Kernel.cpp
    linalg/AddMM.cpp
    linalg/Det.cpp
    linalg/Inverse.cpp
    linalg/LeastSquares.cpp
    linalg/LU.cpp
    linalg/Matmul.cpp
    linalg/Solve.cpp
    linalg/SVD.cpp
    linalg/Tri.cpp
    nns/FixedRadiusIndex.cpp
    nns/FixedRadiusSearchOps.cpp
    nns/KnnIndex.cpp
    nns/NanoFlannIndex.cpp
    nns/NearestNeighborSearch.cpp
    nns/NNSIndex.cpp
)

# core_impl contains the implementation of core functions that are not exposed in the public API
open3d_ispc_add_library(core_impl OBJECT)
set_target_properties(core_impl PROPERTIES CXX_VISIBILITY_PRESET "hidden")

target_sources(core_impl PRIVATE
    kernel/Arange.cpp
    kernel/BinaryEW.cpp
    kernel/IndexGetSet.cpp
    kernel/IndexReduction.cpp
    kernel/NonZero.cpp
    kernel/Reduction.cpp
    kernel/UnaryEW.cpp
    kernel/ArangeCPU.cpp
    kernel/BinaryEWCPU.cpp
    kernel/IndexGetSetCPU.cpp
    kernel/IndexReductionCPU.cpp
    kernel/NonZeroCPU.cpp
    kernel/ReductionCPU.cpp
    kernel/UnaryEWCPU.cpp
    linalg/AddMMCPU.cpp
    linalg/InverseCPU.cpp
    linalg/LeastSquaresCPU.cpp
    linalg/LUCPU.cpp
    linalg/MatmulCPU.cpp
    linalg/SolveCPU.cpp
    linalg/SVDCPU.cpp
    linalg/TriCPU.cpp
)

if (BUILD_SYCL_MODULE)
open3d_sycl_target_sources(core_impl PRIVATE
    kernel/UnaryEWSYCL.cpp
    kernel/BinaryEWSYCL.cpp
    kernel/ArangeSYCL.cpp
    kernel/IndexGetSetSYCL.cpp
    kernel/NonZeroSYCL.cpp
    kernel/IndexReductionSYCL.cpp
    kernel/ReductionSYCL.cpp
    linalg/AddMMSYCL.cpp
    linalg/InverseSYCL.cpp
    linalg/LeastSquaresSYCL.cpp
    linalg/LUSYCL.cpp
    linalg/MatmulSYCL.cpp
    linalg/SolveSYCL.cpp
    linalg/SVDSYCL.cpp
    linalg/TriSYCL.cpp
)
endif()

if (BUILD_CUDA_MODULE)
    target_sources(core PRIVATE
        MemoryManagerCUDA.cpp
    )
    target_sources(core_impl PRIVATE
        hashmap/CUDA/CreateCUDAHashBackend.cu
        hashmap/CUDA/CUDAHashBackendBuffer.cu
        hashmap/CUDA/SlabNodeManager.cu
        kernel/ArangeCUDA.cu
        kernel/BinaryEWCUDA.cu
        kernel/IndexGetSetCUDA.cu
        kernel/IndexReductionCUDA.cu
        kernel/NonZeroCUDA.cu
        kernel/ReductionCUDA.cu
        kernel/UnaryEWCUDA.cu
        linalg/AddMMCUDA.cpp
        linalg/InverseCUDA.cpp
        linalg/LeastSquaresCUDA.cpp
        linalg/LinalgUtils.cpp
        linalg/LUCUDA.cpp
        linalg/MatmulCUDA.cpp
        linalg/SolveCUDA.cpp
        linalg/SVDCUDA.cpp
        linalg/TriCUDA.cu
        nns/FixedRadiusSearchOps.cu
        nns/kernel/BlockSelectFloat32.cu
        nns/kernel/BlockSelectFloat64.cu
        nns/KnnSearchOps.cu
    )
endif()

if (BUILD_ISPC_MODULE)
    target_sources(core PRIVATE
        Indexer.ispc
    )
    target_sources(core_impl PRIVATE
        kernel/BinaryEWCPU.ispc
        kernel/UnaryEWCPU.ispc
    )
endif()

open3d_show_and_abort_on_warning(core)
open3d_set_global_properties(core)
open3d_set_open3d_lib_properties(core)
open3d_link_3rdparty_libraries(core)

open3d_show_and_abort_on_warning(core_impl)
open3d_set_global_properties(core_impl)
open3d_set_open3d_lib_properties(core_impl)
open3d_link_3rdparty_libraries(core_impl)

if(BUILD_CUDA_MODULE)
    target_include_directories(core SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_include_directories(core_impl SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
