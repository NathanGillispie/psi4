include(TestingMacros)

file(COPY li3_ref_mos.npy DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(ENABLE_Einsums)
    add_regression_test(cghf_li3 "psi;quicktests;cghf")
endif()
