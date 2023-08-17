#include "cuLib/context.cuh"


#include <catch2/catch_test_macros.hpp>

TEST_CASE("Create Context", "[context]")
{  
    Context cudaContext(0);  
}