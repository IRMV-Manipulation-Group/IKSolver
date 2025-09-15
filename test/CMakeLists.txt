find_package(GTest REQUIRED)
include(GoogleTest)

add_gtest_discover_for_file(test_kinematics.cpp LINK_LIBRARIES ${PROJECT_NAME} ${PROJECT_NAME}_interface yaml-cpp)
add_gtest_discover_for_file(test_interface.cpp LINK_LIBRARIES ${PROJECT_NAME} ${PROJECT_NAME}_interface yaml-cpp)
