from backend._alias import alias_module

globals().update(alias_module(__name__, "routes").__dict__)
