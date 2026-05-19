from backend._alias import alias_module

globals().update(alias_module(__name__, "heatmap").__dict__)
