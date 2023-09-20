from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    overload,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from _typeshed import Incomplete
from typing_extensions import Never

import torch
from torch._classes import classes as classes
from torch._jit_internal import _qualified_name as _qualified_name
from torch.jit._builtins import _register_builtin as _register_builtin
from torch.jit._fuser import (
    _graph_for as _graph_for,
    _script_method_graph_for as _script_method_graph_for,
)
from torch.jit._monkeytype_config import (
    JitTypeTraceConfig as JitTypeTraceConfig,
    JitTypeTraceStore as JitTypeTraceStore,
    monkeytype_trace as monkeytype_trace,
)
from torch.jit._recursive import (
    _compile_and_register_class as _compile_and_register_class,
    infer_methods_to_compile as infer_methods_to_compile,
    ScriptMethodStub as ScriptMethodStub,
    wrap_cpp_module as wrap_cpp_module,
)
from torch.jit._state import (
    _enabled as _enabled,
    _set_jit_function_cache as _set_jit_function_cache,
    _set_jit_overload_cache as _set_jit_overload_cache,
    _try_get_jit_cached_function as _try_get_jit_cached_function,
    _try_get_jit_cached_overloads as _try_get_jit_cached_overloads,
)
from torch.jit.frontend import (
    get_default_args as get_default_args,
    get_jit_class_def as get_jit_class_def,
    get_jit_def as get_jit_def,
)
from torch.nn import Module as Module
from torch.overrides import (
    has_torch_function as has_torch_function,
    has_torch_function_unary as has_torch_function_unary,
    has_torch_function_variadic as has_torch_function_variadic,
)
from torch.package import (
    PackageExporter as PackageExporter,
    PackageImporter as PackageImporter,
)
from torch.utils import set_module as set_module

from ._serialization import validate_map_location as validate_map_location

ScriptFunction = torch._C.ScriptFunction

type_trace_db: JitTypeTraceStore

# Defined in torch/csrc/jit/python/script_init.cpp
ResolutionCallback = Callable[[str], Callable[..., Any]]
ClassVar = TypeVar("ClassVar", bound=type)

def _reduce(cls) -> None: ...

class Attribute(NamedTuple):
    value: Incomplete
    type: Incomplete

def _get_type_trace_db(): ...
def _get_function_from_type(cls, name): ...
def _is_new_style_class(cls): ...

class OrderedDictWrapper:
    _c: Incomplete
    def __init__(self, _c) -> None: ...
    def keys(self): ...
    def values(self): ...
    def __len__(self) -> int: ...
    def __delitem__(self, k) -> None: ...
    def items(self): ...
    def __setitem__(self, k, v) -> None: ...
    def __contains__(self, k) -> bool: ...
    def __getitem__(self, k): ...

class OrderedModuleDict(OrderedDictWrapper):
    _python_modules: Incomplete
    def __init__(self, module, python_dict) -> None: ...
    def items(self): ...
    def __contains__(self, k) -> bool: ...
    def __setitem__(self, k, v) -> None: ...
    def __getitem__(self, k): ...

class ScriptMeta(type):
    def __init__(cls, name, bases, attrs) -> None: ...

class _CachedForward:
    def __get__(self, obj, cls): ...

class ScriptWarning(Warning): ...

def script_method(fn): ...

class ConstMap:
    const_mapping: Incomplete
    def __init__(self, const_mapping) -> None: ...
    def __getattr__(self, attr): ...

def unpackage_script_module(
    importer: PackageImporter, script_module_id: str
) -> torch.nn.Module: ...

_magic_methods: Incomplete

class RecursiveScriptClass:
    _c: Incomplete
    _props: Incomplete
    def __init__(self, cpp_class) -> None: ...
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value): ...
    def forward_magic_method(self, method_name, *args, **kwargs): ...
    def __getstate__(self) -> None: ...
    def __iadd__(self, other): ...

def method_template(self, *args, **kwargs): ...

class ScriptModule(Module, metaclass=ScriptMeta):
    __jit_unused_properties__: Incomplete
    def __init__(self) -> None: ...
    forward: Callable[..., Any]
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value): ...
    def define(self, src): ...
    def _replicate_for_data_parallel(self): ...
    def __reduce_package__(self, exporter: PackageExporter): ...
    # add __jit_unused_properties__
    @property
    def code(self) -> str: ...
    @property
    def code_with_constants(self) -> Tuple[str, ConstMap]: ...
    @property
    def graph(self) -> torch.Graph: ...
    @property
    def inlined_graph(self) -> torch.Graph: ...
    @property
    def original_name(self) -> str: ...

class RecursiveScriptModule(ScriptModule):
    _disable_script_meta: bool
    _c: Incomplete
    def __init__(self, cpp_module) -> None: ...
    @staticmethod
    def _construct(cpp_module, init_fn): ...
    @staticmethod
    def _finalize_scriptmodule(script_module) -> None: ...
    _concrete_type: Incomplete
    _modules: Incomplete
    _parameters: Incomplete
    _buffers: Incomplete
    __dict__: Incomplete
    def _reconstruct(self, cpp_module) -> None: ...
    def save(self, f, **kwargs): ...
    def _save_for_lite_interpreter(self, *args, **kwargs): ...
    def _save_to_buffer_for_lite_interpreter(self, *args, **kwargs): ...
    def save_to_buffer(self, *args, **kwargs): ...
    def get_debug_state(self, *args, **kwargs): ...
    def extra_repr(self): ...
    def graph_for(self, *args, **kwargs): ...
    def define(self, src) -> None: ...
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value): ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def forward_magic_method(self, method_name, *args, **kwargs): ...
    def __iter__(self): ...
    def __getitem__(self, idx): ...
    def __len__(self) -> int: ...
    def __contains__(self, key) -> bool: ...
    def __dir__(self): ...
    def __bool__(self) -> bool: ...
    def _replicate_for_data_parallel(self): ...

def _get_methods(cls): ...

_compiled_methods_allowlist: Incomplete

def _make_fail(name): ...
def call_prepare_scriptable_func_impl(obj, memo): ...
def call_prepare_scriptable_func(obj): ...
def create_script_dict(obj): ...
def create_script_list(obj, type_hint: Incomplete | None = ...): ...
@overload
def script(
    obj: Type[Module],
    optimize: Optional[bool] = None,
    _frames_up: int = 0,
    _rcb: Optional[ResolutionCallback] = None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
) -> Never: ...
@overload
def script(
    obj: Dict,
    optimize: Optional[bool] = None,
    _frames_up: int = 0,
    _rcb: Optional[ResolutionCallback] = None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
) -> torch.ScriptDict: ...
@overload
def script(
    obj: List,
    optimize: Optional[bool] = None,
    _frames_up: int = 0,
    _rcb: Optional[ResolutionCallback] = None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
) -> torch.ScriptList: ...
@overload
def script(  # type: ignore[misc]
    obj: Module,
    optimize: Optional[bool] = None,
    _frames_up: int = 0,
    _rcb: Optional[ResolutionCallback] = None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
) -> RecursiveScriptModule: ...
@overload
def script(  # type: ignore[misc]
    obj: ClassVar,
    optimize: Optional[bool] = None,
    _frames_up: int = 0,
    _rcb: Optional[ResolutionCallback] = None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
) -> ClassVar: ...
@overload
def script(  # type: ignore[misc]
    obj: Callable,
    optimize: Optional[bool] = None,
    _frames_up: int = 0,
    _rcb: Optional[ResolutionCallback] = None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
) -> ScriptFunction: ...
@overload
def script(
    obj: Any,
    optimize: Optional[bool] = None,
    _frames_up: int = 0,
    _rcb: Optional[ResolutionCallback] = None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
) -> RecursiveScriptClass: ...
@overload
def script(
    obj,
    optimize: Incomplete | None = ...,
    _frames_up: int = ...,
    _rcb: Incomplete | None = ...,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = ...,
): ...
def _check_overload_defaults(impl_defaults, overload_defaults, loc) -> None: ...
def _compile_function_with_overload(overload_fn, qual_name, impl_fn): ...
def _get_overloads(obj): ...
def _check_directly_compile_overloaded(obj) -> None: ...
def interface(obj): ...
def _recursive_compile_class(obj, loc): ...

CompilationUnit: Incomplete

def pad(s: str, padding: int, offset: int = ..., char: str = ...): ...

class _ScriptProfileColumn:
    header: Incomplete
    alignment: Incomplete
    offset: Incomplete
    rows: Incomplete
    def __init__(
        self, header: str, alignment: int = ..., offset: int = ...
    ) -> None: ...
    def add_row(self, lineno: int, value: Any): ...
    def materialize(self): ...

class _ScriptProfileTable:
    cols: Incomplete
    source_range: Incomplete
    def __init__(
        self, cols: List[_ScriptProfileColumn], source_range: List[int]
    ) -> None: ...
    def dump_string(self): ...

class _ScriptProfile:
    profile: Incomplete
    def __init__(self) -> None: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def dump_string(self) -> str: ...
    def dump(self) -> None: ...

def _unwrap_optional(x): ...
