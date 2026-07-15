add_cressim_test(
    SUITE forward_pipeline
    NAME forward_pipeline
    SOURCE forward_pipeline.cpp
    INCLUDE_SRC
    LIBS cressim_neo_engine cressim_neo_graphics_test_support
)
