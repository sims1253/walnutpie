if(MSVC)
    set(gtest_force_shared_crt on)
endif()

set(BUILD_GMOCK OFF CACHE BOOL "Disable building Google Mock" FORCE)
FetchContent_Declare(googletest
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

##########################
##   Source Coverage    ##
##########################

function(add_test TEST_NAME)
    add_executable(${TEST_NAME}_test ${TEST_NAME}_test.cpp)
    target_include_directories(${TEST_NAME}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TEST_NAME}_test
    PRIVATE
      gtest_main
      Eigen3::Eigen
      walnutpie
  )
    if(WALNUTPIE_COVERAGE)
        target_compile_options(${TEST_NAME}_test PRIVATE
      -fprofile-instr-generate
      -fcoverage-mapping
    )
        target_link_options(${TEST_NAME}_test PRIVATE
      -fprofile-instr-generate
    )
    endif()
    gtest_discover_tests(${TEST_NAME}_test)
endfunction()

add_test(summary)
add_test(config)
add_test(util)

foreach(name IN ITEMS stan_thread_stub stan_thread_missing)
  add_library(${name} SHARED stan_thread_stub.cpp)
  target_link_libraries(${name} PRIVATE BridgeStan::BridgeStan)
  target_compile_definitions(${name} PRIVATE BRIDGESTAN_EXPORT)
endforeach()
target_compile_definitions(stan_thread_missing PRIVATE OMIT_MODEL_INFO)

add_test(stan_threads)
target_link_libraries(stan_threads_test PRIVATE BridgeStan::BridgeStan ${CMAKE_DL_LIBS})
target_compile_definitions(stan_threads_test PRIVATE
  THREAD_STUB_PATH="$<TARGET_FILE:stan_thread_stub>"
  THREAD_MISSING_PATH="$<TARGET_FILE:stan_thread_missing>")
add_dependencies(stan_threads_test stan_thread_stub stan_thread_missing)

# Exercise the native boundary in ordinary C++ CI, without a Python runtime.
if(TARGET walnutpy)
  set(thread_binding_target walnutpy)
else()
  add_library(walnutpy_thread_test_binding SHARED
    "${PROJECT_SOURCE_DIR}/python/src/walnutpie/walnutpy.cpp")
  target_link_libraries(walnutpy_thread_test_binding PRIVATE
    walnutpie BridgeStan::BridgeStan Eigen3::Eigen)
  include(GenerateExportHeader)
  generate_export_header(walnutpy_thread_test_binding
    BASE_NAME walnutpy
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/walnutpy_export.h")
  target_include_directories(walnutpy_thread_test_binding PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}")
  set(thread_binding_target walnutpy_thread_test_binding)
endif()
add_test(stan_binding_threads)
target_link_libraries(stan_binding_threads_test PRIVATE
  ${thread_binding_target} BridgeStan::BridgeStan ${CMAKE_DL_LIBS})
target_compile_definitions(stan_binding_threads_test PRIVATE
  THREAD_STUB_PATH="$<TARGET_FILE:stan_thread_stub>"
  THREAD_MISSING_PATH="$<TARGET_FILE:stan_thread_missing>")
add_dependencies(stan_binding_threads_test stan_thread_stub stan_thread_missing)
