# gloo/transport/peel/CMakeLists.txt
# Standalone Peel transport: UDP/raw-socket data path plus PEEL routing.

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  message(WARNING "Gloo Peel transport is supported only on Linux")
  return()
endif()

find_package(PkgConfig REQUIRED)
# A GLOBAL imported target survives the peel -> transport -> gloo directory
# nesting and carries non-system include paths and library directories too.
pkg_check_modules(PEEL_HIREDIS REQUIRED IMPORTED_TARGET GLOBAL hiredis)

list(APPEND GLOO_TRANSPORT_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_protocol.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_redis.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_full_mesh.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_transport.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_transport_reno.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_broadcast.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_broadcast_ring.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_broadcast_stop_and_wait.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_allgather.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_allgather_ring.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_context.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_discovery.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_tree.cc"
  )

list(APPEND GLOO_TRANSPORT_HDRS
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_protocol.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_redis.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_full_mesh.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_transport.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_broadcast.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_broadcast_ring.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_broadcast_stop_and_wait.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_allgather.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_allgather_ring.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_allreduce_ring.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_context.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_discovery.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/peel_tree.h"
  )

# Keep Peel-specific standalone test programs close to the transport. They are
# not installed and are only built when BUILD_TEST is enabled.
if(BUILD_TEST)
  add_executable(test_peel_broadcast test_peel_broadcast.cc)
  target_link_libraries(test_peel_broadcast gloo PkgConfig::PEEL_HIREDIS)
  target_include_directories(test_peel_broadcast PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PEEL_HIREDIS_INCLUDE_DIRS})
  target_compile_definitions(test_peel_broadcast PRIVATE GLOO_USE_REDIS=1)

  add_executable(test_peel_allgather test_peel_allgather.cc)
  target_link_libraries(test_peel_allgather gloo PkgConfig::PEEL_HIREDIS)
  target_include_directories(test_peel_allgather PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PEEL_HIREDIS_INCLUDE_DIRS})
  target_compile_definitions(test_peel_allgather PRIVATE GLOO_USE_REDIS=1)

  add_executable(test_peel_broadcast_ring test_peel_broadcast_ring.cc)
  target_link_libraries(test_peel_broadcast_ring gloo PkgConfig::PEEL_HIREDIS)
  target_include_directories(test_peel_broadcast_ring PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PEEL_HIREDIS_INCLUDE_DIRS})
  target_compile_definitions(test_peel_broadcast_ring PRIVATE GLOO_USE_REDIS=1)
endif()

set(GLOO_TRANSPORT_SRCS ${GLOO_TRANSPORT_SRCS} PARENT_SCOPE)
set(GLOO_TRANSPORT_HDRS ${GLOO_TRANSPORT_HDRS} PARENT_SCOPE)
