set(TB_TEST_DEPENDS
  tb-opt
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-mini-triton
  "Running the mini_triton_nvgpu_v1 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TB_TEST_DEPENDS}
)

set_target_properties(check-mini-triton PROPERTIES FOLDER "Tests")
