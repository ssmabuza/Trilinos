
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  CrsGraph_test_LL
  NAME_POSTFIX unit
  SOURCES cxx_main.cpp
  ARGS -v
  COMM mpi serial
  FAIL_REGULAR_EXPRESSION "tests FAILED"
  )
