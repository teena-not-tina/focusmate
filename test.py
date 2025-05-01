import config
print("Loaded config.py from:", config.__file__)
print("TEMP_DIR in config:", getattr(config, "TEMP_DIR", "NOT FOUND")) 