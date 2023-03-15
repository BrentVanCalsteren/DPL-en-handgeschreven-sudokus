try:
    from deepproblog.src.deepproblog.engines.approximate_engine import ApproximateEngine
except ModuleNotFoundError:
    import warnings

    warnings.warn("ApproximateEngine is not available as PySwip could not be found")
    ApproximateEngine = None
from deepproblog.src.deepproblog.engines.engine import Engine
from deepproblog.src.deepproblog.engines.exact_engine import ExactEngine
