"""
Simplified VM code which works for some cases.
"""

# Virtual Machine class, which executes python bytecode

import builtins
import dis
import types
import typing as tp
import operator


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.10/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.block_stack: tp.Any = []
        self.bytecode_counter = -1

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        for instruction in dis.get_instructions(self.code):
            if self.bytecode_counter == instruction.offset:
                self.bytecode_counter = -1
            if self.bytecode_counter == -1:
                if instruction.opname.lower().find('jump') == -1 and instruction.opname.lower() != 'for_iter':
                    getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
                if instruction.opname.lower().find('jump') != -1 or instruction.opname.lower() == 'for_iter':
                    getattr(self, instruction.opname.lower() + "_op")(instruction.argval, instruction.offset)
                    if self.bytecode_counter != -1:
                        break
                if self.return_value is not None:
                    return self.return_value  # type: ignore
        if self.bytecode_counter != -1:
            return self.run()
        else:
            return self.return_value

    def for_iter_op(self, delta: int, now: int) -> None:
        iterobj = self.top()
        try:
            v = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.bytecode_counter = delta + now

    def jump_forward_op(self, delta: int, now: int) -> None:
        self.bytecode_counter = now + delta

    def jump_absolute_op(self, target: int, arg: tp.Any) -> None:
        self.bytecode_counter = target

    def jump_if_true_or_pop_op(self, target: int, arg: tp.Any) -> None:
        TOS = self.pop()
        if TOS:
            self.bytecode_counter = target
            self.push(TOS)

    def jump_if_false_or_pop_op(self, target: int, arg: tp.Any) -> None:
        TOS = self.pop()
        if not TOS:
            self.bytecode_counter = target
            self.push(TOS)

    def pop_jump_if_true_op(self, target: int, arg: tp.Any) -> None:
        TOS = self.pop()
        if TOS:
            self.bytecode_counter = target

    def pop_jump_if_false_op(self, target: int, arg: tp.Any) -> None:
        TOS = self.pop()
        if not TOS:
            self.bytecode_counter = target

    def call_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-CALL_FUNCTION

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#4243
        """
        if arg > 0:
            arguments = self.popn(arg)
            f = self.pop()
            self.push(f(*arguments))
        else:
            f = self.pop()
            self.push(f())

    def call_function_kw_op(self, arg: int) -> None:
        keys = self.pop()
        kwargs: dict[str, tp.Any] = dict()
        for i in range(1, len(keys) + 1):
            kwargs[keys[-i]] = self.pop()
        args = self.popn(arg - len(keys))
        f = self.pop()
        self.push(f(*args, **kwargs))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-LOAD_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2829
        """
        # TODO: parse all scopes
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-LOAD_GLOBAL

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2958
        """
        # TODO: parse all scopes
        if arg in self.globals:
            val = self.globals[arg]
        elif arg in self.builtins:
            val = self.builtins[arg]
        else:
            raise NameError
        self.push(val)

    # self.push(self.builtins[arg])

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-LOAD_CONST

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1871
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-RETURN_VALUE

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2436
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-POP_TOP

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1886
        """
        self.pop()

    CO_VARARGS = 4
    CO_VARKEYWORDS = 8

    ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
    ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
    ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
    ERR_MISSING_POS_ARGS = 'Missing positional arguments'
    ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
    ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'

    def make_function_op(self, flag: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-MAKE_FUNCTION

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4290

        Parse stack:
            https://github.com/python/cpython/blob/3.10/Objects/call.c#L612

        Call function in cpython:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L4209
        """
        # TODO: use arg to parse function defaults
        name = self.pop()  # the qualified name of the function (at TOS)  # noqa
        code = self.pop()  # the code associated with the function (at TOS1)
        if flag & 0x01 == 0:
            defaults = []
        if flag & 0x02 == 0:
            kwdefaults: dict[str, tp.Any] = dict()
        if flag & 0x01 != 0 and flag & 0x02 != 0:
            kwdefaults = self.pop()
            defaults = self.pop()
        elif flag & 0x01 == 0 and flag & 0x02 != 0:
            kwdefaults = self.pop()
            defaults = []
        elif flag & 0x01 != 0 and flag & 0x02 == 0:
            kwdefaults = dict()
            defaults = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # TODO: parse input arguments using code attributes such as co_argcount
            nonlocal code
            nonlocal defaults
            nonlocal kwdefaults
            func = code
            for key in kwargs:
                if key in func.co_varnames[:func.co_posonlyargcount] and \
                        (func.co_flags & self.CO_VARKEYWORDS == 0):
                    raise TypeError(self.ERR_POSONLY_PASSED_AS_KW)
            if len(args) > func.co_argcount and (func.co_flags & self.CO_VARARGS == 0):
                raise TypeError(self.ERR_TOO_MANY_POS_ARGS)

            if defaults is not None:
                default_names = func.co_varnames[func.co_argcount - len(defaults):
                                                 func.co_argcount]
                default_values = defaults
                defaults_dict = dict(zip(default_names, default_values))
            else:
                defaults_dict = dict()  # type: ignore
            posonly_names = func.co_varnames[:func.co_posonlyargcount]
            posonly_values = args[:func.co_posonlyargcount]
            answ = dict()
            for i in range(len(posonly_names)):
                if i < len(posonly_values):
                    answ[posonly_names[i]] = posonly_values[i]
                elif posonly_names[i] in defaults_dict:
                    answ[posonly_names[i]] = defaults_dict[posonly_names[i]]
                else:
                    raise TypeError(self.ERR_MISSING_POS_ARGS)
            pos_values = args[func.co_posonlyargcount:]
            if func.co_flags & self.CO_VARARGS != 0:
                answ[func.co_varnames[func.co_argcount + func.co_kwonlyargcount]] = tuple()
            if func.co_argcount >= func.co_posonlyargcount + len(pos_values):
                pos_names = func.co_varnames[func.co_posonlyargcount:
                                             func.co_posonlyargcount + len(pos_values)]
                for i in range(len(pos_names)):
                    answ[pos_names[i]] = pos_values[i]
            elif func.co_flags & self.CO_VARARGS != 0:
                pos_names = func.co_varnames[func.co_posonlyargcount:
                                             func.co_argcount]
                for i in range(len(pos_names)):
                    answ[pos_names[i]] = pos_values[i]
                for i in range(len(pos_names), len(pos_values)):
                    answ[func.co_varnames[func.co_argcount + func.co_kwonlyargcount]] += \
                        (pos_values[i],)
            for key in kwargs:
                if key in answ and (func.co_flags & self.CO_VARKEYWORDS == 0):
                    raise TypeError(self.ERR_MULT_VALUES_FOR_ARG)
            kwonly_names = func.co_varnames[func.co_argcount:
                                            func.co_argcount + func.co_kwonlyargcount]
            for i in range(len(kwonly_names)):
                if kwonly_names[i] in kwargs:
                    answ[kwonly_names[i]] = kwargs[kwonly_names[i]]
                if kwdefaults is not None:
                    if kwonly_names[i] in kwdefaults and kwonly_names[i] not in kwargs:
                        answ[kwonly_names[i]] = kwdefaults[kwonly_names[i]]
                    elif kwonly_names[i] not in kwargs:
                        raise TypeError(self.ERR_MISSING_KWONLY_ARGS)
                if kwonly_names[i] not in kwargs and kwdefaults is None:
                    raise TypeError(self.ERR_MISSING_KWONLY_ARGS)

            kw_or_default_names = func.co_varnames[func.co_posonlyargcount + len(pos_values):
                                                   func.co_argcount]
            for i in range(len(kw_or_default_names)):
                if kw_or_default_names[i] in kwargs:
                    answ[kw_or_default_names[i]] = kwargs[kw_or_default_names[i]]
                elif kw_or_default_names[i] in defaults_dict:
                    answ[kw_or_default_names[i]] = defaults_dict[kw_or_default_names[i]]
                else:
                    raise TypeError(self.ERR_MISSING_POS_ARGS)

            if func.co_flags & self.CO_VARKEYWORDS != 0 and func.co_flags & self.CO_VARARGS != 0:
                kekwargs = func.co_varnames[func.co_argcount + func.co_kwonlyargcount + 1]
                answ[kekwargs] = dict()
            if func.co_flags & self.CO_VARKEYWORDS != 0 and func.co_flags & self.CO_VARARGS == 0:
                kekwargs = func.co_varnames[func.co_argcount + func.co_kwonlyargcount]
                answ[kekwargs] = dict()
            for key in kwargs:
                if key not in answ and (func.co_flags & self.CO_VARKEYWORDS == 0):
                    raise TypeError(self.ERR_TOO_MANY_KW_ARGS)
                elif (key not in answ or key in posonly_names) and \
                        func.co_flags & self.CO_VARKEYWORDS != 0 and \
                        func.co_flags & self.CO_VARARGS != 0:
                    answ[func.co_varnames[func.co_argcount + func.co_kwonlyargcount + 1]][key] = \
                        kwargs[key]
                elif (key not in answ or key in posonly_names) and \
                        func.co_flags & self.CO_VARKEYWORDS != 0 and \
                        func.co_flags & self.CO_VARARGS == 0:
                    answ[func.co_varnames[func.co_argcount + func.co_kwonlyargcount]][key] = \
                        kwargs[key]
                elif key in answ and key in pos_names:
                    raise TypeError(self.ERR_MULT_VALUES_FOR_ARG)
            # parsed_args = self.bind_args(code, defaults, kwdefaults, *args, **kwargs)
            parsed_args = answ
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def nop_op(self, arg: tp.Any) -> None:
        pass

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.10.6/library/dis.html#opcode-STORE_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.10/Python/ceval.c#L2758
        """
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, arg: str) -> None:
        const = self.pop()
        self.globals[arg] = const

    def store_subscr_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        third = self.pop()
        second[first] = third

    def delete_subscr_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        del second[first]

    def binary_add_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second + first)

    def binary_multiply_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second * first)

    def binary_matrix_multiply_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second @ first)

    def binary_floor_divide_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second // first)

    def binary_true_divide_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second / first)

    def binary_modulo_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second % first)

    def binary_xor_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second ^ first)

    def binary_or_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second | first)

    def binary_and_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second & first)

    def binary_power_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second ** first)

    def binary_subtract_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second - first)

    def binary_subscr_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second[first])

    def binary_lshift_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second << first)

    def binary_rshift_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second >> first)

    def inplace_add_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second + first)

    def inplace_multiply_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second * first)

    def inplace_matrix_multiply_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second @ first)

    def inplace_floor_divide_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second // first)

    def inplace_true_divide_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second / first)

    def inplace_modulo_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second % first)

    def inplace_xor_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second ^ first)

    def inplace_or_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second | first)

    def inplace_and_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second & first)

    def inplace_power_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second ** first)

    def inplace_subtract_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second - first)

    def inplace_lshift_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second << first)

    def inplace_rshift_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second >> first)

    def unary_invert_op(self, arg: tp.Any) -> None:
        el = self.pop()
        self.push(~el)

    def unary_positive_op(self, arg: tp.Any) -> None:
        el = self.pop()
        self.push(+el)

    def unary_not_op(self, arg: tp.Any) -> None:
        el = self.pop()
        self.push(not el)

    def unary_negative_op(self, arg: tp.Any) -> None:
        el = self.pop()
        self.push(-el)

    def store_fast_op(self, name: str) -> None:
        self.locals[name] = self.pop()

    def load_fast_op(self, name: str) -> None:
        if name in self.locals:
            val = self.locals[name]
        else:
            raise UnboundLocalError
        self.push(val)

    def load_attr_op(self, attr: str) -> None:
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def store_attr_op(self, name: str) -> None:
        val, obj = self.popn(2)
        setattr(obj, name, val)

    COMPARE_OPERATORS = {
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '>=': operator.ge,
        'in': lambda x, y: x in y,
        'not in': lambda x, y: x not in y,
        'is': lambda x, y: x is y,
        'is not': lambda x, y: x is not y,
        'issubclass': lambda x, y: issubclass(x, Exception) and issubclass(x, y)}

    def compare_op_op(self, arg: tp.Any) -> None:
        x, y = self.popn(2)
        self.push(self.COMPARE_OPERATORS[arg](x, y))

    def build_map_op(self, count: int) -> None:
        lst = list(self.popn(2 * count))
        dct = dict()
        for i in range(count):
            dct[lst[2 * i]] = lst[2 * i + 1]
        self.push(dct)

    def build_const_key_map_op(self, count: int) -> None:
        keys = self.pop()
        val = list(self.popn(count))
        self.push(dict(zip(keys, val)))

    def build_list_op(self, count: int) -> None:
        lst = list(self.popn(count))
        self.push(lst)

    def build_set_op(self, count: int) -> None:
        answ = set(self.popn(count))
        self.push(answ)

    def build_slice_op(self, arg: int) -> None:
        TOS = self.pop()
        TOS1 = self.pop()
        if arg == 2:
            self.push(slice(TOS1, TOS))
        if arg == 3:
            TOS2 = self.pop()
            self.push(slice(TOS2, TOS1, TOS))

    def build_tuple_op(self, count: int) -> None:
        arguments = self.popn(count)
        tup = tuple(arguments)
        self.push(tup)

    def build_string_op(self, count: int) -> None:
        arguments = self.popn(count)
        string = ''
        for el in arguments:
            string += el
        self.push(string)

    def list_append_op(self, count: int) -> None:
        val = self.pop()
        the_list = self.data_stack[-count]
        the_list.append(val)

    def list_extend_op(self, count: int) -> None:
        TOS = self.pop()
        the_list = self.data_stack[-count]
        list.extend(the_list, TOS)

    def dict_update_op(self, count: tp.Any) -> None:
        val = self.pop()
        the_dict = self.data_stack[-count]
        dict.update(the_dict, val)

    def list_to_tuple_op(self, count: tp.Any) -> None:
        lst = self.pop()
        self.push(tuple(lst))

    def map_add_op(self, count: int) -> None:
        TOS = self.pop()
        TOS1 = self.pop()
        dict.__setitem__(self.data_stack[-count], TOS1, TOS)

    def set_add_op(self, count: tp.Any) -> None:
        val = self.pop()
        the_set = self.data_stack[-count]
        set.add(the_set, val)

    def set_update_op(self, count: tp.Any) -> None:
        val = self.pop()
        the_set = self.data_stack[-count]
        set.update(the_set, val)

    def get_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def rot_two_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(first)
        self.push(second)

    def rot_three_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        third = self.pop()
        self.push(first)
        self.push(third)
        self.push(second)

    def rot_four_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        third = self.pop()
        four = self.pop()
        self.push(first)
        self.push(four)
        self.push(third)
        self.push(second)

    def dup_top_op(self, arg: tp.Any) -> None:
        first = self.pop()
        self.push(first)
        self.push(first)

    def dup_top_two_op(self, arg: tp.Any) -> None:
        first = self.pop()
        second = self.pop()
        self.push(second)
        self.push(first)
        self.push(second)
        self.push(first)

    def delete_fast_op(self, arg: tp.Any) -> None:
        del self.locals[arg]

    def delete_global_op(self, arg: tp.Any) -> None:
        del self.globals[arg]

    def delete_name_op(self, arg: tp.Any) -> None:
        if arg in self.locals:
            del self.locals[arg]
        elif arg in self.globals:
            del self.globals[arg]

    def delete_attr_op(self, arg: tp.Any) -> None:
        TOS = self.pop()
        if TOS in self.locals:
            del self.locals[TOS].arg
        elif TOS in self.globals:
            del self.globals[TOS].arg

    def contains_op_op(self, invert: int) -> None:
        TOS = self.pop()
        TOS1 = self.pop()
        if invert != 1:
            self.push(TOS1 in TOS)
        else:
            self.push(TOS1 not in TOS)

    def is_op_op(self, invert: int) -> None:
        TOS = self.pop()
        TOS1 = self.pop()
        if invert != 1:
            self.push(TOS1 is TOS)
        else:
            self.push(TOS1 is not TOS)

    def raise_varargs_op(self, arg: int) -> None:
        if arg == 0:
            raise
        if arg == 1:
            TOS = self.pop()
            raise TOS
        if arg == 2:
            TOS = self.pop()
            TOS1 = self.pop()
            raise TOS1 from TOS

    def load_assertion_error_op(self, arg: int) -> None:
        self.push(AssertionError)

    def load_build_class_op(self, arg: tp.Any) -> None:
        self.push(builtins.__build_class__)

    def unpack_sequence_op(self, count: int) -> None:
        TOS = self.pop()
        for i in range(1, count + 1):
            self.push(TOS[-i])
        if len(TOS) != count:
            self.push(TOS[:-count])

    def load_method_op(self, name: tp.Any) -> None:
        obj = self.pop()
        if hasattr(obj, name):
            self.push(name)
            self.push(obj)
        else:
            self.push(None)
            self.push(obj)

    def call_method_op(self, count: int) -> None:
        if count > 0:
            pos_args = self.popn(count)
        obj = self.pop()
        method = self.pop()
        if count > 0:
            self.push(getattr(obj, method)(*pos_args))
        else:
            self.push(getattr(obj, method)())

    def format_value_op(self, flag: tp.Any) -> None:
        pass


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
