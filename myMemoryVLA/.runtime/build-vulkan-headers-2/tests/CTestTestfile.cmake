# CMake generated Testfile for 
# Source directory: /workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests
# Build directory: /workspace/multimodal/myMemoryVLA/.runtime/build-vulkan-headers-2/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("integration.add_subdirectory" "/workspace/multimodal/myMemoryVLA/.venv/lib/python3.10/site-packages/cmake/data/bin/ctest" "--build-and-test" "/workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests/integration" "/workspace/multimodal/myMemoryVLA/.runtime/build-vulkan-headers-2/tests/add_subdirectory" "--build-generator" "Ninja" "--build-options" "-DFIND_PACKAGE_TESTING=OFF" "-DVULKAN_HEADERS_ENABLE_MODULE=OFF")
set_tests_properties("integration.add_subdirectory" PROPERTIES  _BACKTRACE_TRIPLES "/workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests/CMakeLists.txt;10;add_test;/workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests/CMakeLists.txt;0;")
add_test("integration.install" "/workspace/multimodal/myMemoryVLA/.venv/lib/python3.10/site-packages/cmake/data/bin/cmake" "--install" "/workspace/multimodal/myMemoryVLA/.runtime/build-vulkan-headers-2" "--prefix" "/workspace/multimodal/myMemoryVLA/.runtime/build-vulkan-headers-2/tests/install" "--config" "")
set_tests_properties("integration.install" PROPERTIES  _BACKTRACE_TRIPLES "/workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests/CMakeLists.txt;19;add_test;/workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests/CMakeLists.txt;0;")
add_test("integration.find_package" "/workspace/multimodal/myMemoryVLA/.venv/lib/python3.10/site-packages/cmake/data/bin/ctest" "--build-and-test" "/workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests/integration" "/workspace/multimodal/myMemoryVLA/.runtime/build-vulkan-headers-2/tests/find_package" "--build-generator" "Ninja" "--build-options" "-DFIND_PACKAGE_TESTING=ON" "-DCMAKE_PREFIX_PATH=/workspace/multimodal/myMemoryVLA/.runtime/build-vulkan-headers-2/tests/install")
set_tests_properties("integration.find_package" PROPERTIES  DEPENDS "integration.install" _BACKTRACE_TRIPLES "/workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests/CMakeLists.txt;24;add_test;/workspace/multimodal/myMemoryVLA/.runtime/Vulkan-Headers/tests/CMakeLists.txt;0;")
