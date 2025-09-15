add_executable(template_app1 example_app.cpp)
target_link_libraries(template_app1 PRIVATE ${PROJECT_NAME} ${PROJECT_NAME}_interface)