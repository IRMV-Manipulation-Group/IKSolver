cmake_minimum_required(VERSION 3.26)

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()
set(CMAKE_C_STANDARD 99)

set(CMAKE_CXX_EXTENSIONS OFF)

set(PROJECT_VERSION 0.1.4)
set(PROJECT_SO_VERSION 0)

# Set the author and email
set(AUTHOR "JPengyu")
set(AUTHOR_EMAIL "jiangpengyu@sjtu.edu.cn")
set(PACKAGE_DESCRIPTION "IRMV IKSolver project")

# Set the project name and language
project(
        IKSolver
        VERSION ${PROJECT_VERSION}
        DESCRIPTION ${PACKAGE_DESCRIPTION}
        LANGUAGES CXX C)

# options area
option(ENABLE_WARNINGS "Enable project warnings" ON)
option(ENABLE_SANITIZER "Enable address and undefined behavior sanitizer" OFF)
option(ENABLE_TEST "Enable test " ON)
option(ENABLE_DOXYGEN "Enable doxygen" OFF)
option(ENABLE_STATIC "Build with static library" ON)
option(COMPILE_${PROJECT_NAME}_PYBINDING "Compile pybind " ON)

# Set up modules path and include necessary scripts from cmake_template
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#required VPN
include(FetchContent)
# Include standard project settings
include(StandardProjectSettings)

if (ENABLE_STATIC)
    SET(COMPILE_TYPE "STATIC")
else ()
    SET(COMPILE_TYPE "SHARED")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC
endif ()

# find package area
include(Utilities)

# find IRMV packages
set(THIS_PACKAGE_IRMV_DEPENDS
        "irmv_core 0.3"
#        "imc 1.0"
)
prepare_irmv_dependencies("${THIS_PACKAGE_IRMV_DEPENDS}" IRMV_PACKAGE_DEPENDS)

# find other packages
find_package(Eigen3 3.1 REQUIRED)
find_package(yaml-cpp 0.6 REQUIRED)
# Define cmake dependencies with versions
set(THIS_PACKAGE_DEPENDS
        "Eigen3 3.1"
        "yaml-cpp 0.6"
)
# Define debian dependencies with versions
set(THIS_PACKAGE_BINARY_DEPENDS
        "libeigen3-dev 3.1"
        "libyaml-cpp-dev 0.6"
)
# Find each package with the specified version
prepare_debian_dependencies("${THIS_PACKAGE_BINARY_DEPENDS}" DEBIAN_PACKAGE_DEPENDS)

# add definitions
# common definitions
add_definitions(-DPLOG_CAPTURE_FILE)
add_definitions(-DFMT_HEADER_ONLY)
add_definitions(-D${PROJECT_NAME}_RESOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources")
add_definitions(-D${PROJECT_NAME}_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config")

# include area (should be installed into the system)
set(THIS_PACKAGE_INCLUDE_DIRS
        include
        interface/include
)

# special handle to pass include directories to parent scope
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
else ()
    set(${PROJECT_NAME}_INCLUDE_DIRS_CHILD)
    foreach (DIR ${THIS_PACKAGE_INCLUDE_DIRS})
        set(${PROJECT_NAME}_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}")
        list(APPEND ${PROJECT_NAME}_INCLUDE_DIRS_CHILD ${${PROJECT_NAME}_DIR})
    endforeach ()
    set(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_INCLUDE_DIRS_CHILD} PARENT_SCOPE)
endif ()

# include area
include_directories(
        SYSTEM

        # package include directories
        ${EIGEN3_INCLUDE_DIRS}
        # include directories that should be installed
        ${THIS_PACKAGE_INCLUDE_DIRS}
        # irmv package include directories
#        ${imc_INCLUDE_DIRS} # currently empty
        ${irmv_core_INCLUDE_DIRS}
        # third party include directories
        third_party
)

# project library area
add_library(${PROJECT_NAME}
        ${COMPILE_TYPE}
        src/kinematics_base.cpp
        src/kinematics_screw.cpp
)
# link irmv package and/or other packages
target_link_libraries(${PROJECT_NAME} PUBLIC ${irmv_core_LIBRARIES} yaml-cpp)

# add test
if (ENABLE_TEST)
    add_subdirectory(test)
endif ()

# add executable area
add_subdirectory(app)

#add interface for outside user and python user
add_subdirectory(interface)

# debug for address and undefined behavior sanitizer
if (ENABLE_SANITIZER AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    include(Sanitizers)
    myproject_enable_sanitizers(${PROJECT_NAME}_interface ON ON ON OFF ON)
endif ()

# package area
include(PackageProject)
myproject_package_project(
        NAME ${PROJECT_NAME}
        VERSION ${PROJECT_VERSION}
        TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_interface
        PUBLIC_INCLUDES ${THIS_PACKAGE_INCLUDE_DIRS}
        PUBLIC_DEPENDENCIES ${THIS_PACKAGE_IRMV_DEPENDS} ${THIS_PACKAGE_DEPENDS}
)

set(CPACK_PACKAGE_FILE_NAME
        "${CMAKE_PROJECT_NAME}-${CMAKE_PROJECT_VERSION}-${GIT_SHORT_SHA}-${CMAKE_SYSTEM_NAME}-${CMAKE_BUILD_TYPE}-${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
)

set(CPACK_GENERATOR "DEB")

set(CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${CMAKE_PROJECT_VERSION}")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${AUTHOR}) #given your own name

# Use a valid Debian architecture name
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
else ()
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
endif ()


# Generate CPACK_DEBIAN_PACKAGE_DEPENDS

# Set the package dependencies
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.28), libstdc++6 (>= 9.4), python3 (>= 3.8), ${DEBIAN_PACKAGE_DEPENDS}, ${IRMV_PACKAGE_DEPENDS}")

include(CPack)

if (ENABLE_DOXYGEN)
    include(Doxygen)
    # Exclude the pybind11 and third_party directories from the documentation
    set(EXCLUDE_PATH_AND_ITEMS
            "${CMAKE_CURRENT_SOURCE_DIR}/interface/pybind11"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-release"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-docker-debug"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-docker-release"
            "${CMAKE_CURRENT_SOURCE_DIR}/example"
            "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
    )

    myproject_enable_doxygen("awesome-sidebar" "${EXCLUDE_PATH_AND_ITEMS}")
endif ()