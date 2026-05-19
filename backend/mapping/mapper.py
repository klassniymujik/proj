from backend._alias import alias_module

globals().update(alias_module(__name__, "mapper").__dict__)
