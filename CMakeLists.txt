#
# CMakeLists.txt  cmake for mssg
# 06-Feb-2018  chuck@ece.cmu.edu
#

#
#  general cmake flags:
#    -DCMAKE_INSTALL_PREFIX=/usr/local     -- the prefix for installing
#    -DCMAKE_BUILD_TYPE=type               -- type can be DEBUG, RELEASE, ...
#    -DCMAKE_PREFIX_PATH=/dir              -- external packages
#
#     note that CMAKE_PREFIX_PATH can be a list of directories:
#      -DCMAKE_PREFIX_PATH='/dir1;/dir2;/dir3'
#

cmake_minimum_required(VERSION 3.1)
project (mssg C CXX)

# link shared lib with full rpath
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# setup cache variables for ccmake
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release
         CACHE STRING "Choose the type of build." FORCE)
    set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                  "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()
set (CMAKE_PREFIX_PATH "" CACHE STRING "External dependencies path")
set (BUILD_SHARED_LIBS "OFF" CACHE BOOL "Build a shared library")

# Find required packages
find_package (mercury CONFIG REQUIRED)
find_package (MPI MODULE REQUIRED)
string (REPLACE " " ";" MPI_CXX_COMPILE_FLAGS_LIST "${MPI_CXX_COMPILE_FLAGS}")

#
# configure/load in standard modules we plan to use and probe the enviroment
#
include (CMakePackageConfigHelpers)

# where to install files for "find_package"
set (mssg-pkg-loc "share/cmake/mssg")

#
# library version set here (e.g. for shared libs).
#
set (MSSG_VERSION_MAJOR 1)
set (MSSG_VERSION_MINOR 0)
set (MSSG_VERSION_PATCH 0)
set (mssg-vers "${MSSG_VERSION_MAJOR}.${MSSG_VERSION_MINOR}")
set (MSSG_VERSION "${mssg-vers}.${MSSG_VERSION_PATCH}")

#
# create library target (user can specify shared vs. static using
# BUILD_SHARED_LIBS).  arrange for users of our lib to get the correct
# -Is for our include directories.
#
add_library (mssg mssg.cc)
target_include_directories (mssg PUBLIC $<INSTALL_INTERFACE:include>)

# make sure our build includes are BEFORE a previously installed version
target_include_directories (mssg BEFORE PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries (mssg mercury)

set_target_properties(mssg
                      PROPERTIES VERSION ${MSSG_VERSION}
                      SOVERSION ${MSSG_VERSION_MAJOR})

# XXX: special handling for MPI, where the config comes in via MPI_<lang>_ vars.
# Note that the mpich on ub14 gives a leading space that we need to trim off.
foreach (lcv ${MPI_CXX_COMPILE_FLAGS_LIST})
    if (NOT ${lcv} STREQUAL "")
        target_compile_options (mssg PUBLIC $<BUILD_INTERFACE:${lcv}>)
    endif ()
endforeach ()

# XXX: have to do this one dir at a time!!
foreach (lcv ${MPI_CXX_INCLUDE_PATH})
    target_include_directories (mssg PUBLIC $<BUILD_INTERFACE:${lcv}>)
endforeach ()
foreach (lcv ${MPI_CXX_LIBRARIES})
    target_link_libraries(mssg $<BUILD_INTERFACE:${lcv}>)
endforeach ()

# XXX: this doesn't propagate to lib users, is that a problem?
set_property (TARGET mssg APPEND PROPERTY LINK_FLAGS
              ${MPI_CXX_LINK_FLAGS})

#
# installation stuff (packaging and install commands)
#
write_basic_package_version_file(
    "mssg-config-version.cmake"
    VERSION ${MSSG_VERSION}
    COMPATIBILITY AnyNewerVersion)

# generate our config file for find_package()
configure_file (mssg-config.cmake.in mssg-config.cmake @ONLY)

#
# mssg-runner
#

add_executable(mssg-runner mssg-runner.cc)
target_link_libraries(mssg-runner mssg)

# XXX: special handling for MPI, where the config comes in via MPI_<lang>_ vars.
# Note that the mpich on ub14 gives a leading space that we need to trim off.
foreach (lcv ${MPI_CXX_COMPILE_FLAGS_LIST})
    if (NOT ${lcv} STREQUAL "")
        target_compile_options (mssg-runner PUBLIC ${lcv})
    endif ()
endforeach ()

# XXX: have to do this one dir at a time!!
foreach (lcv ${MPI_CXX_INCLUDE_PATH})
    target_include_directories (mssg-runner PUBLIC ${lcv})
endforeach ()
foreach (lcv ${MPI_CXX_LIBRARIES})
    target_link_libraries(mssg-runner ${lcv})
endforeach ()

set_property (TARGET mssg-runner APPEND PROPERTY LINK_FLAGS
        ${MPI_CXX_LINK_FLAGS})

#
# "make install" rules
#
install (TARGETS mssg EXPORT mssg-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install (TARGETS mssg-runner RUNTIME DESTINATION bin)
install (EXPORT mssg-targets
        DESTINATION ${mssg-pkg-loc}
        FILE "mssg-targets.cmake")
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/mssg-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mssg-config-version.cmake"
        DESTINATION ${mssg-pkg-loc} )
install (FILES mssg.h
        DESTINATION include)
