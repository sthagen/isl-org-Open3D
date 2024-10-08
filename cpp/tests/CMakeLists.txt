open3d_ispc_add_executable(tests)

add_subdirectory(camera)
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(ml)
add_subdirectory(pipelines)
add_subdirectory(t/geometry)
add_subdirectory(t/io)
add_subdirectory(t/pipelines)
add_subdirectory(test_utility)
add_subdirectory(utility)
add_subdirectory(visualization)

target_sources(tests PRIVATE
    Main.cpp
    Tests.cpp
)

target_include_directories(tests PRIVATE ".")

# If gpu not available, add "DISABLED_" to the gpu test names
if (BUILD_CUDA_MODULE)
    target_compile_definitions(tests PRIVATE GPU_CONDITIONAL_TEST_STR=) # Empty string
else()
    target_compile_definitions(tests PRIVATE GPU_CONDITIONAL_TEST_STR=DISABLED_)
endif()

if (WITH_IPP)
    target_compile_definitions(tests PRIVATE IPP_CONDITIONAL_TEST_STR=) # Empty string (test not disabled)
else()
    target_compile_definitions(tests PRIVATE IPP_CONDITIONAL_TEST_STR=DISABLED_)
endif()

target_link_libraries(tests PRIVATE
    Open3D::Open3D
    Open3D::3rdparty_jsoncpp
    Open3D::3rdparty_googletest
    Open3D::3rdparty_threads
    Open3D::3rdparty_vtk
)

if (TARGET Open3D::3rdparty_openmp)
    target_link_libraries(tests PRIVATE
        Open3D::3rdparty_openmp
    )
endif()
    

open3d_show_and_abort_on_warning(tests)
open3d_set_global_properties(tests)
# On Windows, running tests from the build folder needs tbb.dll to be in the same folder.
if (WIN32)
  add_custom_command(
      TARGET tests
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:tbb> "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/"
  )
endif()

if (BUILD_AZURE_KINECT)
    # K4A headers are directly used in test. Currently we don't need to link
    # the K4A libraries.
    target_include_directories(tests SYSTEM PRIVATE ${K4A_INCLUDE_DIR})
endif()

if (BUILD_CUDA_MODULE)
    # We still need to explicitly link against CUDA libraries.
    # Consider removing dependencies of CUDA headers in the future.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tests PRIVATE CUDA::cudart)
endif()
