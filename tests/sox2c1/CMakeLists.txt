include(TestingMacros)

if(ENABLE_Einsums)
    add_regression_test(sox2c1 "psi;quicktests;x2c")
    set_tests_properties(sox2c1 PROPERTIES DISABLED True)
endif()
