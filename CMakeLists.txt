cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
if(POLICY CMP0175)
  cmake_policy(SET CMP0175 NEW)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0175 NEW)

if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
project(pde_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
  COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short HEAD
  OUTPUT_VARIABLE PDE_GIT_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT PDE_GIT_SHA)
  set(PDE_GIT_SHA "unknown")
endif()
string(TIMESTAMP PDE_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)

option(USE_CUDA "Enable CUDA backend" OFF)
if(USE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit not found. CUDA backend will be disabled.")
  endif()
endif()

if(APPLE)
  option(USE_METAL "Enable Metal backend" ON)
else()
  option(USE_METAL "Enable Metal backend" OFF)
endif()
option(USE_TPU "Enable TPU backend" OFF)
option(USE_OPENMP "Enable OpenMP parallelization" ON)

# Find OpenMP
if(USE_OPENMP)
  # On macOS with Apple Clang, find_package(OpenMP) doesn't work with Homebrew's libomp.
  # We need to manually configure it.
  if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Check for Homebrew libomp
    execute_process(
      COMMAND brew --prefix libomp
      OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE BREW_RESULT
    )
    if(BREW_RESULT EQUAL 0 AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}")
      message(STATUS "Found Homebrew libomp at ${HOMEBREW_LIBOMP_PREFIX}")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
      set(OpenMP_CXX_INCLUDE_DIR "${HOMEBREW_LIBOMP_PREFIX}/include")

      # Create imported target for OpenMP
      if(NOT TARGET OpenMP::OpenMP_CXX)
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
          IMPORTED_LOCATION "${OpenMP_omp_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_CXX_INCLUDE_DIR}"
          INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
        )
      endif()
      set(OpenMP_CXX_FOUND TRUE)
      message(STATUS "OpenMP configured with Homebrew libomp")
    else()
      message(STATUS "Homebrew libomp not found. Install with: brew install libomp")
      set(OpenMP_CXX_FOUND FALSE)
    endif()
  else()
    # Non-Apple or non-Clang: use standard find_package
    find_package(OpenMP)
  endif()

  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled for CPU parallelization")
  else()
    message(STATUS "OpenMP not found. CPU parallelization will be disabled.")
  endif()
endif()

if(USE_CUDA AND CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(USE_METAL)
  enable_language(OBJCXX)
endif()
if(APPLE)
  # Enable Objective-C++ for file dialog on macOS
  if(NOT CMAKE_OBJCXX_COMPILER)
    enable_language(OBJCXX)
  endif()
endif()

# Add nlohmann/json using FetchContent (used by both CLI and GUI)
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

set(PDE_CORE_SOURCES
  src/advection.cpp
  src/advection_tests.cpp
  src/time_integrator.cpp
  src/time_integrator_tests.cpp
  src/pressure_projection.cpp
  src/pressure_projection_tests.cpp
  src/backend_detect.cpp
  src/backend_solve.cpp
  src/backend_capability_matrix.cpp
  src/expression_eval.cpp
  src/coefficient_evaluator.cpp
  src/dataset_tools.cpp
  src/latex_lexer.cpp
  src/latex_ast.cpp
  src/latex_parser.cpp
  src/coupled_solver.cpp
  src/coupled_examples.cpp
  src/input_parse_domain.cpp
  src/input_parse_validate.cpp
  src/input_parse.cpp
  src/string_utils.cpp
  src/shape_utils.cpp
  src/shape_io.cpp
  src/mesh_io.cpp
  src/unstructured_solver.cpp
  src/backends/cpu/solvers/relaxation.cpp
  src/backends/cpu/solvers/multigrid.cpp
  src/backends/cpu/solvers/krylov.cpp
  src/backends/cpu/cpu_utils.cpp
  src/backends/cpu/solver.cpp
  src/residual.cpp
  src/vtk_reader.cpp
  src/vtk_writer.cpp
  src/coordinate_metrics.cpp
  src/conserved_monitor.cpp
  src/embedded_boundary.cpp
  src/finite_differences.cpp
  src/boundary_utils.cpp
  src/log_service.cpp
  src/solve_service.cpp
  src/visualization.cpp
  src/run_config.cpp
  src/run_metadata.cpp
  src/run_summary.cpp
  src/solver_tokens.cpp
  src/self_test.cpp
  src/mms.cpp
)

add_executable(pde_sim
  src/main.cpp
  ${PDE_CORE_SOURCES}
)
target_compile_definitions(pde_sim PRIVATE
  PDE_GIT_SHA="${PDE_GIT_SHA}"
  PDE_BUILD_TIMESTAMP="${PDE_BUILD_TIMESTAMP}"
  PDE_BUILD_TYPE="$<CONFIG>"
)

target_include_directories(pde_sim PRIVATE
  include
  src/backends/cuda
  src/backends/metal
)
if(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(pde_sim PRIVATE nlohmann_json::nlohmann_json)
elseif(TARGET nlohmann_json)
  target_link_libraries(pde_sim PRIVATE nlohmann_json)
endif()

# Link OpenMP to pde_sim
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(pde_sim PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(checkpoint_equivalence
  tests/checkpoint_equivalence.cpp
  ${PDE_CORE_SOURCES}
)
target_include_directories(checkpoint_equivalence PRIVATE
  include
  src/backends/cuda
  src/backends/metal
)
if(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(checkpoint_equivalence PRIVATE nlohmann_json::nlohmann_json)
elseif(TARGET nlohmann_json)
  target_link_libraries(checkpoint_equivalence PRIVATE nlohmann_json)
endif()
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(checkpoint_equivalence PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(regression_suite
  tests/regression_suite.cpp
  ${PDE_CORE_SOURCES}
)
target_include_directories(regression_suite PRIVATE
  include
  src/backends/cuda
  src/backends/metal
)
if(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(regression_suite PRIVATE nlohmann_json::nlohmann_json)
elseif(TARGET nlohmann_json)
  target_link_libraries(regression_suite PRIVATE nlohmann_json)
endif()
if(USE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(regression_suite PRIVATE OpenMP::OpenMP_CXX)
endif()

option(BUILD_GUI "Build OpenGL GUI" ON)
if(BUILD_GUI)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

  add_subdirectory(third_party/glfw)
  find_package(OpenGL REQUIRED)

  add_library(imgui STATIC
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_glfw.cpp
    third_party/imgui/backends/imgui_impl_opengl3.cpp
    third_party/imgui/misc/cpp/imgui_stdlib.cpp
  )
  target_include_directories(imgui PUBLIC
    third_party/imgui
    third_party/imgui/backends
    third_party/imgui/misc/cpp
    third_party/glfw/include
  )
  target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_CUSTOM)
  target_compile_definitions(imgui PRIVATE GL_SILENCE_DEPRECATION)

  add_executable(pde_gui
    gui_gl/main.cpp
    gui_gl/core/window_manager.cpp
    gui_gl/core/event_handler.cpp
    gui_gl/core/native_menu.cpp
    gui_gl/core/application.cpp
    gui_gl/core/application_tabs.cpp
    gui_gl/solver/solver_manager.cpp
    gui_gl/GlViewer.cpp
    gui_gl/stb_image_impl.cpp
    gui_gl/rendering/shader_manager.cpp
    gui_gl/rendering/mesh_builder.cpp
    gui_gl/rendering/grid_builder.cpp
    gui_gl/rendering/projection.cpp
    gui_gl/rendering/renderer.cpp
    gui_gl/rendering/render_types.cpp
    gui_gl/rendering/coordinate_transforms.cpp
    gui_gl/rendering/value_sampling.cpp
    gui_gl/validation.cpp
    gui_gl/ui_helpers.cpp
    gui_gl/templates.cpp
    gui_gl/app_state.cpp
    gui_gl/app_helpers.cpp
    gui_gl/components/ui_component.cpp
    gui_gl/components/log_panel.cpp
    gui_gl/components/inspection_tools.cpp
    gui_gl/components/comparison_tools.cpp
    gui_gl/components/color_picker.cpp
    gui_gl/components/error_dialog.cpp
    gui_gl/components/plot_widget.cpp
    gui_gl/rendering/glyph_renderer.cpp
    gui_gl/tools/comparison_tools.cpp
    gui_gl/tools/statistics_compute.cpp
    gui_gl/tools/selection_tool.cpp
    gui_gl/systems/backend_capabilities.cpp
    gui_gl/systems/backend_providers.cpp
    gui_gl/systems/command_history.cpp
    gui_gl/systems/pde_type_registry.cpp
    gui_gl/systems/coordinate_system_registry.cpp
    gui_gl/systems/solver_method_registry.cpp
    gui_gl/systems/ui_config.cpp
    gui_gl/systems/ui_config_validation.cpp
    gui_gl/models/application_state.cpp
    gui_gl/panels/main/equation_panel.cpp
    gui_gl/panels/main/domain_panel.cpp
    gui_gl/panels/main/grid_panel.cpp
    gui_gl/panels/main/boundary_panel.cpp
  gui_gl/panels/main/compute_panel.cpp
  gui_gl/panels/inspect/field_panel.cpp
  gui_gl/panels/inspect/slice_panel.cpp
  gui_gl/panels/inspect/isosurface_panel.cpp
  gui_gl/panels/inspect/export_panel.cpp
  gui_gl/panels/inspect/advanced_panel.cpp
  gui_gl/panels/inspect/comparison_panel.cpp
  gui_gl/panels/inspect/convergence_panel.cpp
  gui_gl/panels/inspect/statistics_panel.cpp
  gui_gl/panels/inspect/point_probe_panel.cpp
  gui_gl/panels/inspect/animation_export_panel.cpp
  gui_gl/panels/main/initial_conditions_panel.cpp
  gui_gl/panels/main/preset_manager_panel.cpp
  gui_gl/panels/main/source_term_panel.cpp
  gui_gl/panels/main/material_properties_panel.cpp
  gui_gl/panels/main/mesh_preview_panel.cpp
  gui_gl/panels/main/parameter_sweep_panel.cpp
  gui_gl/panels/main/testing_panel.cpp
  gui_gl/testing/test_runner.cpp
  gui_gl/panels/preferences/color_preferences_panel.cpp
    gui_gl/panels/main/time_panel.cpp
    gui_gl/panels/main/run_panel.cpp
    gui_gl/panels/preferences/viewer_panel.cpp
    gui_gl/panels/preferences/io_panel.cpp
    gui_gl/panels/preferences/benchmark_panel.cpp
    gui_gl/panels/preferences/ui_config_panel.cpp
    gui_gl/docking/dock_node.cpp
    gui_gl/docking/docking_context.cpp
    gui_gl/docking/view_registry.cpp
    gui_gl/docking/view_state_registry.cpp
    gui_gl/docking/layout_serializer.cpp
    gui_gl/docking/preset_layouts.cpp
    gui_gl/docking/view_registration.cpp
    gui_gl/styles/ui_style.cpp
    gui_gl/handlers/solve_handler.cpp
    gui_gl/handlers/file_handler.cpp
    gui_gl/utils/math_utils.cpp
    gui_gl/utils/string_utils.cpp
    gui_gl/utils/path_utils.cpp
    gui_gl/utils/coordinate_utils.cpp
    gui_gl/utils/file_dialog.cpp
    gui_gl/io/file_utils.cpp
    gui_gl/io/preferences_io.cpp
    gui_gl/io/image_export.cpp
    ${PDE_CORE_SOURCES}
  )
  target_compile_definitions(pde_gui PRIVATE
    PDE_GIT_SHA="${PDE_GIT_SHA}"
    PDE_BUILD_TIMESTAMP="${PDE_BUILD_TIMESTAMP}"
    PDE_BUILD_TYPE="$<CONFIG>"
  )
  target_include_directories(pde_gui PRIVATE include gui_gl third_party/stb src/backends/cuda src/backends/metal)
  target_link_libraries(pde_gui PRIVATE imgui glfw OpenGL::GL)
  # Help debug runtime crashes (e.g., ImGui stack issues) with usable backtraces.
  target_compile_options(pde_gui PRIVATE -g -fno-omit-frame-pointer)
  # nlohmann/json is header-only, link to it to get include directories
  # The target name is nlohmann_json (from PROJECT_NAME) or nlohmann_json::nlohmann_json (alias)
  if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(pde_gui PRIVATE nlohmann_json::nlohmann_json)
  elseif(TARGET nlohmann_json)
    target_link_libraries(pde_gui PRIVATE nlohmann_json)
  endif()
  target_compile_definitions(pde_gui PRIVATE GL_SILENCE_DEPRECATION)
  if(APPLE)
    target_link_libraries(pde_gui PRIVATE "-framework AppKit" "-framework Cocoa")
    # Compile file_dialog.cpp as Objective-C++ on macOS
    set_source_files_properties(gui_gl/utils/file_dialog.cpp
      gui_gl/core/native_menu.cpp
      PROPERTIES LANGUAGE OBJCXX)
  endif()

  # Link OpenMP to pde_gui
  if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(pde_gui PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(USE_METAL)
    set(METAL_SOURCES
      src/backends/metal/MetalSolve.mm
      src/backends/metal/utils/metal_utils.mm
      src/backends/metal/solvers/relaxation.mm
      src/backends/metal/solvers/krylov.mm
      src/backends/metal/solvers/multigrid.mm
      src/backends/metal/solvers/time_series.mm
    )
    target_compile_definitions(pde_gui PRIVATE USE_METAL)
    target_sources(pde_gui PRIVATE ${METAL_SOURCES})
    target_link_libraries(pde_gui PRIVATE "-framework Metal" "-framework Foundation")
    set(METAL_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/metal/solvers/kernels.metal)
    add_custom_command(TARGET pde_gui POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${METAL_SHADER} $<TARGET_FILE_DIR:pde_gui>/kernels.metal
    )
  endif()
  if(USE_CUDA AND CUDAToolkit_FOUND)
    set(CUDA_SOURCES
      src/backends/cuda/CudaSolve.cu
      src/backends/cuda/utils/cuda_utils.cu
      src/backends/cuda/solvers/relaxation.cu
      src/backends/cuda/solvers/krylov.cu
      src/backends/cuda/solvers/multigrid.cu
    )
    target_sources(pde_gui PRIVATE ${CUDA_SOURCES})
    set_target_properties(pde_gui PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(pde_gui PRIVATE USE_CUDA)
  endif()
endif()

if(USE_CUDA AND CUDAToolkit_FOUND)
  target_sources(pde_sim PRIVATE ${CUDA_SOURCES})
  set_target_properties(pde_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(pde_sim PRIVATE USE_CUDA)
endif()

if(USE_METAL)
  target_sources(pde_sim PRIVATE ${METAL_SOURCES})
  target_compile_definitions(pde_sim PRIVATE USE_METAL)
  target_link_libraries(pde_sim PRIVATE "-framework Metal" "-framework Foundation")

  set(METAL_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/metal/solvers/kernels.metal)
  add_custom_command(TARGET pde_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${METAL_SHADER} $<TARGET_FILE_DIR:pde_sim>/kernels.metal
  )
endif()

if(USE_TPU)
  target_compile_definitions(pde_sim PRIVATE USE_TPU)
endif()
