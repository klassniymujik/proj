from backend._alias import alias_module

globals().update(alias_module(__name__, "render").__dict__)
