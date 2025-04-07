class Configurator(dict):
    """Singleton class that behaves like a dictionary for global configurations."""
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

config = Configurator()