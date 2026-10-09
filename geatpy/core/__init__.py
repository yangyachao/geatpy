"""
geatpy.core - Core evolutionary algorithm operators implemented in Rust.
"""
import sys
import types
try:
    import _geatpy_core
except ImportError:
    try:
        from geatpy import _geatpy_core
    except ImportError:
        import geatpy._geatpy_core as _geatpy_core

# Wrapper to normalize keyword arguments (e.g. Parallel -> parallel, FixType -> fix_type, etc.)
def _wrap_callable(func):
    sig = getattr(func, '__text_signature__', None)
    if not sig:
        return func
    try:
        sig_str = sig.strip()
        if sig_str.startswith('(') and sig_str.endswith(')'):
            sig_str = sig_str[1:-1]
        param_parts = [p.split('=')[0].strip() for p in sig_str.split(',') if p.strip()]
        known_params = {}
        for p in param_parts:
            p_clean = p.lstrip('*')
            if p_clean:
                known_params[p_clean.lower().replace('_', '')] = p_clean
        if 'sigma' in known_params.values():
            known_params['sigma3'] = 'sigma'
        if 'dis_i' in known_params.values():
            known_params['disi'] = 'dis_i'
        if 'n' in known_params.values():
            known_params['disi'] = 'n'
    except Exception:
        return func

    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not kwargs:
            return func(*args)
        new_kwargs = {}
        for k, v in kwargs.items():
            norm_k = k.lower().replace('_', '')
            if norm_k in known_params:
                new_kwargs[known_params[norm_k]] = v
            else:
                new_kwargs[k] = v
        return func(*args, **new_kwargs)

    return wrapper

# Indicator module
indicator = getattr(_geatpy_core, 'indicator', _geatpy_core)

# Register each operator as a virtual submodule under geatpy.core
# so `from geatpy.core.func import func` works seamlessly.
_current_module = sys.modules[__name__]

for _attr in dir(_geatpy_core):
    if not _attr.startswith('_'):
        _obj = getattr(_geatpy_core, _attr)
        if callable(_obj) and not isinstance(_obj, types.ModuleType):
            _obj = _wrap_callable(_obj)
        setattr(_current_module, _attr, _obj)
        _sub_mod_name = f"{__name__}.{_attr}"
        if _sub_mod_name not in sys.modules:
            if isinstance(_obj, types.ModuleType):
                sys.modules[_sub_mod_name] = _obj
            else:
                _mod = types.ModuleType(_sub_mod_name)
                setattr(_mod, _attr, _obj)
                sys.modules[_sub_mod_name] = _mod

# Meta path finder to dynamically resolve any geatpy.core.<name> imports
class _CoreSubmoduleFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname.startswith("geatpy.core."):
            name = fullname.split(".")[-1]
            if hasattr(_geatpy_core, name):
                from importlib.machinery import ModuleSpec
                return ModuleSpec(fullname, _CoreSubmoduleLoader(name))
        return None

class _CoreSubmoduleLoader:
    def __init__(self, name):
        self.name = name

    def create_module(self, spec):
        obj = getattr(_current_module, self.name, getattr(_geatpy_core, self.name))
        if isinstance(obj, types.ModuleType):
            return obj
        mod = types.ModuleType(spec.name)
        setattr(mod, self.name, obj)
        return mod

    def exec_module(self, module):
        pass

if not any(isinstance(finder, type) and finder.__name__ == "_CoreSubmoduleFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, _CoreSubmoduleFinder)

