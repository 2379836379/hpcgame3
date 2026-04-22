import ctypes
import hashlib
import math
import os
import subprocess
import tempfile

import numpy

try:
    import autograd  # type: ignore
except Exception:
    autograd = None


class _Node:
    __slots__ = ("op", "a", "b", "value")

    def __init__(self, op, a=None, b=None, value=None):
        self.op = op
        self.a = a
        self.b = b
        self.value = value


_NODE_CACHE = {}


def _make_node(op, a=None, b=None, value=None):
    key = (op, a, b, value)
    node = _NODE_CACHE.get(key)
    if node is None:
        node = _Node(op, a, b, value)
        _NODE_CACHE[key] = node
    return node


def _const(val):
    return _make_node("const", value=float(val))


def _var(name, index=None):
    if index is None:
        return _make_node("var", value=name)
    return _make_node("var", value=(name, int(index)))


def _is_const(node, val=None):
    if node.op != "const":
        return False
    if val is None:
        return True
    return node.value == float(val)


def _neg(a):
    if _is_const(a):
        return _const(-a.value)
    if a.op == "neg":
        return a.a
    return _make_node("neg", a=a)


def _add(a, b):
    if _is_const(a, 0.0):
        return b
    if _is_const(b, 0.0):
        return a
    if _is_const(a) and _is_const(b):
        return _const(a.value + b.value)
    return _make_node("add", a=a, b=b)


def _sub(a, b):
    if _is_const(b, 0.0):
        return a
    if _is_const(a, 0.0):
        return _neg(b)
    if _is_const(a) and _is_const(b):
        return _const(a.value - b.value)
    return _make_node("sub", a=a, b=b)


def _mul(a, b):
    if _is_const(a, 0.0) or _is_const(b, 0.0):
        return _const(0.0)
    if _is_const(a, 1.0):
        return b
    if _is_const(b, 1.0):
        return a
    if _is_const(a, -1.0):
        return _neg(b)
    if _is_const(b, -1.0):
        return _neg(a)
    if _is_const(a) and _is_const(b):
        return _const(a.value * b.value)
    return _make_node("mul", a=a, b=b)


def _sin(a):
    if _is_const(a):
        return _const(math.sin(a.value))
    return _make_node("sin", a=a)


def _cos(a):
    if _is_const(a):
        return _const(math.cos(a.value))
    return _make_node("cos", a=a)


def _to_node(x):
    if isinstance(x, _TraceScalar):
        return x.node
    if isinstance(x, _Node):
        return x
    if isinstance(x, (int, float, numpy.floating, numpy.integer)):
        return _const(float(x))
    raise TypeError("Unsupported type in trace")


class _TraceScalar:
    __slots__ = ("node",)
    __array_priority__ = 1000

    def __init__(self, node):
        self.node = node

    def __add__(self, other):
        return _TraceScalar(_add(self.node, _to_node(other)))

    def __radd__(self, other):
        return _TraceScalar(_add(_to_node(other), self.node))

    def __sub__(self, other):
        return _TraceScalar(_sub(self.node, _to_node(other)))

    def __rsub__(self, other):
        return _TraceScalar(_sub(_to_node(other), self.node))

    def __mul__(self, other):
        return _TraceScalar(_mul(self.node, _to_node(other)))

    def __rmul__(self, other):
        return _TraceScalar(_mul(_to_node(other), self.node))

    def __neg__(self):
        return _TraceScalar(_neg(self.node))

    def __pow__(self, power):
        if isinstance(power, (int, numpy.integer)):
            if power == 0:
                return _TraceScalar(_const(1.0))
            if power == 1:
                return self
            if power == 2:
                return self * self
            if power > 2:
                out = self
                for _ in range(power - 1):
                    out = out * self
                return out
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float, numpy.floating, numpy.integer)):
            return _TraceScalar(_mul(self.node, _const(1.0 / float(other))))
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        if ufunc is numpy.sin:
            return _TraceScalar(_sin(_to_node(inputs[0])))
        if ufunc is numpy.cos:
            return _TraceScalar(_cos(_to_node(inputs[0])))
        if ufunc is numpy.add:
            return _TraceScalar(_add(_to_node(inputs[0]), _to_node(inputs[1])))
        if ufunc is numpy.subtract:
            return _TraceScalar(_sub(_to_node(inputs[0]), _to_node(inputs[1])))
        if ufunc is numpy.multiply:
            return _TraceScalar(_mul(_to_node(inputs[0]), _to_node(inputs[1])))
        if ufunc is numpy.negative:
            return _TraceScalar(_neg(_to_node(inputs[0])))
        if ufunc is numpy.power:
            base = _to_node(inputs[0])
            exp = inputs[1]
            if isinstance(exp, (int, numpy.integer)):
                if exp == 0:
                    return _TraceScalar(_const(1.0))
                if exp == 1:
                    return _TraceScalar(base)
                if exp == 2:
                    return _TraceScalar(_mul(base, base))
                if exp > 2:
                    out = _TraceScalar(base)
                    for _ in range(exp - 1):
                        out = out * _TraceScalar(base)
                    return out
        return NotImplemented


class _TraceMath:
    __slots__ = ()

    def sin(self, x):
        return _TraceScalar(_sin(_to_node(x)))

    def cos(self, x):
        return _TraceScalar(_cos(_to_node(x)))


class _TraceArray:
    __slots__ = ("_var", "_n")

    def __init__(self, var_node, n):
        self._n = n
        if isinstance(var_node, str):
            name = var_node
        else:
            name = var_node
        self._var = [
            _TraceScalar(_var(name, i)) for i in range(n)
        ]

    def __getitem__(self, _idx):
        if isinstance(_idx, slice):
            return self._var[_idx]
        if not isinstance(_idx, int):
            raise TypeError("Index must be int or slice")
        if _idx < 0:
            _idx += self._n
        return self._var[_idx]

    def __len__(self):
        return self._n

    def __iter__(self):
        return iter(self._var)


def _diff(node, var_key, cache):
    key = (node, var_key)
    cached = cache.get(key)
    if cached is not None:
        return cached
    op = node.op
    if op == "const":
        out = _const(0.0)
    elif op == "var":
        out = _const(1.0 if node.value == var_key else 0.0)
    elif op == "add":
        out = _add(_diff(node.a, var_key, cache), _diff(node.b, var_key, cache))
    elif op == "sub":
        out = _sub(_diff(node.a, var_key, cache), _diff(node.b, var_key, cache))
    elif op == "mul":
        out = _add(
            _mul(_diff(node.a, var_key, cache), node.b),
            _mul(node.a, _diff(node.b, var_key, cache)),
        )
    elif op == "neg":
        out = _neg(_diff(node.a, var_key, cache))
    elif op == "sin":
        out = _mul(_cos(node.a), _diff(node.a, var_key, cache))
    elif op == "cos":
        out = _mul(_neg(_sin(node.a)), _diff(node.a, var_key, cache))
    else:
        raise ValueError("Unsupported op in diff")
    cache[key] = out
    return out


def _node_deps(node, cache):
    cached = cache.get(node)
    if cached is not None:
        return cached
    op = node.op
    if op == "const":
        deps = 0
    elif op == "var":
        val = node.value
        if isinstance(val, tuple):
            name = val[0]
        else:
            name = val
        deps = 1 if name == "q" else 2
    else:
        deps = 0
        if node.a is not None:
            deps |= _node_deps(node.a, cache)
        if node.b is not None:
            deps |= _node_deps(node.b, cache)
    cache[node] = deps
    return deps


def _topo_nodes(roots):
    order = []
    seen = set()

    def visit(node):
        if node in seen:
            return
        seen.add(node)
        if node.a is not None:
            visit(node.a)
        if node.b is not None:
            visit(node.b)
        order.append(node)

    for root in roots:
        visit(root)
    return order


def _format_const(val):
    s = repr(float(val))
    if "e" not in s and "." not in s:
        s += ".0"
    return s


def _emit_static_array(lines, name, values, indent="    ", cols=8):
    lines.append(f"{indent}static const double {name}[{len(values)}] = {{")
    for i in range(0, len(values), cols):
        chunk = ", ".join(_format_const(v) for v in values[i:i + cols])
        lines.append(f"{indent}    {chunk},")
    lines.append(f"{indent}}};")


def _node_ref(node, temp_names):
    if node.op == "const":
        return _format_const(node.value)
    if node.op == "var":
        val = node.value
        if isinstance(val, tuple):
            name, idx = val
            if name == "q":
                return f"q[{idx}]"
            return f"qdot[{idx}]"
        if val == "q":
            return "q[0]"
        return "qdot[0]"
    return temp_names[node]


def _emit_expr(node, prefix, lines):
    order = _topo_nodes([node])
    temp_names = {}
    temp_index = 0
    for item in order:
        if item.op in ("const", "var"):
            continue
        tname = f"t{prefix}_{temp_index}"
        temp_index += 1
        temp_names[item] = tname

        if item.op == "add":
            expr = f"{_node_ref(item.a, temp_names)} + {_node_ref(item.b, temp_names)}"
        elif item.op == "sub":
            expr = f"{_node_ref(item.a, temp_names)} - {_node_ref(item.b, temp_names)}"
        elif item.op == "mul":
            expr = f"{_node_ref(item.a, temp_names)} * {_node_ref(item.b, temp_names)}"
        elif item.op == "neg":
            expr = f"-{_node_ref(item.a, temp_names)}"
        elif item.op == "sin":
            expr = f"sin({_node_ref(item.a, temp_names)})"
        elif item.op == "cos":
            expr = f"cos({_node_ref(item.a, temp_names)})"
        else:
            raise ValueError("Unsupported op in codegen")

        lines.append(f"    double {tname} = {expr};")

    return _node_ref(node, temp_names)


def _generate_c_diag(f_q, f_qv, f_vv, n):
    lines = ["#include <math.h>", "void cal(const double* q, const double* qdot, double* qddot) {"]

    for i in range(n):
        roots = [f_q[i], f_qv[i], f_vv[i]]
        order = _topo_nodes(roots)
        temp_names = {}
        temp_index = 0
        for node in order:
            if node.op in ("const", "var"):
                continue
            tname = f"t{i}_{temp_index}"
            temp_index += 1
            temp_names[node] = tname

            if node.op == "add":
                expr = f"{_node_ref(node.a, temp_names)} + {_node_ref(node.b, temp_names)}"
            elif node.op == "sub":
                expr = f"{_node_ref(node.a, temp_names)} - {_node_ref(node.b, temp_names)}"
            elif node.op == "mul":
                expr = f"{_node_ref(node.a, temp_names)} * {_node_ref(node.b, temp_names)}"
            elif node.op == "neg":
                expr = f"-{_node_ref(node.a, temp_names)}"
            elif node.op == "sin":
                expr = f"sin({_node_ref(node.a, temp_names)})"
            elif node.op == "cos":
                expr = f"cos({_node_ref(node.a, temp_names)})"
            else:
                raise ValueError("Unsupported op in codegen")

            lines.append(f"    double {tname} = {expr};")

        f_ref = _node_ref(roots[0], temp_names)
        c_ref = _node_ref(roots[1], temp_names)
        m_ref = _node_ref(roots[2], temp_names)
        lines.append(f"    qddot[{i}] = ({f_ref} - {c_ref} * qdot[{i}]) / {m_ref};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _generate_c_full(f_q, f_qv, f_vv, n):
    lines = [
        "#include <math.h>",
        "void cal(const double* q, const double* qdot, double* qddot) {",
        f"    double M[{n * n}];",
        f"    double rhs[{n}];",
    ]

    roots = []
    roots.extend(f_q)
    for i in range(n):
        for j in range(n):
            roots.append(f_qv[i][j])
            roots.append(f_vv[i][j])

    order = _topo_nodes(roots)
    temp_names = {}
    temp_index = 0
    for node in order:
        if node.op in ("const", "var"):
            continue
        tname = f"t{temp_index}"
        temp_index += 1
        temp_names[node] = tname

        if node.op == "add":
            expr = f"{_node_ref(node.a, temp_names)} + {_node_ref(node.b, temp_names)}"
        elif node.op == "sub":
            expr = f"{_node_ref(node.a, temp_names)} - {_node_ref(node.b, temp_names)}"
        elif node.op == "mul":
            expr = f"{_node_ref(node.a, temp_names)} * {_node_ref(node.b, temp_names)}"
        elif node.op == "neg":
            expr = f"-{_node_ref(node.a, temp_names)}"
        elif node.op == "sin":
            expr = f"sin({_node_ref(node.a, temp_names)})"
        elif node.op == "cos":
            expr = f"cos({_node_ref(node.a, temp_names)})"
        else:
            raise ValueError("Unsupported op in codegen")

        lines.append(f"    double {tname} = {expr};")

    for i in range(n):
        lines.append(f"    rhs[{i}] = {_node_ref(f_q[i], temp_names)};")

    for i in range(n):
        for j in range(n):
            lines.append(
                f"    rhs[{i}] -= {_node_ref(f_qv[i][j], temp_names)} * qdot[{j}];"
            )

    for i in range(n):
        for j in range(i, n):
            lines.append(f"    M[{i * n + j}] = {_node_ref(f_vv[i][j], temp_names)};")
            if i != j:
                lines.append(f"    M[{j * n + i}] = M[{i * n + j}];")

    lines.append(f"    for (int k = 0; k < {n}; ++k) {{")
    lines.append("        int pivot = k;")
    lines.append("        double maxv = fabs(M[k * %d + k]);" % n)
    lines.append(f"        for (int i = k + 1; i < {n}; ++i) {{")
    lines.append("            double val = fabs(M[i * %d + k]);" % n)
    lines.append("            if (val > maxv) { maxv = val; pivot = i; }")
    lines.append("        }")
    lines.append("        if (pivot != k) {")
    lines.append(f"            for (int j = k; j < {n}; ++j) {{")
    lines.append("                double tmp = M[k * %d + j];" % n)
    lines.append("                M[k * %d + j] = M[pivot * %d + j];" % (n, n))
    lines.append("                M[pivot * %d + j] = tmp;" % n)
    lines.append("            }")
    lines.append("            double tmp = rhs[k]; rhs[k] = rhs[pivot]; rhs[pivot] = tmp;")
    lines.append("        }")
    lines.append("        double diag = M[k * %d + k];" % n)
    lines.append(f"        for (int i = k + 1; i < {n}; ++i) {{")
    lines.append("            double factor = M[i * %d + k] / diag;" % n)
    lines.append(f"            for (int j = k + 1; j < {n}; ++j) {{")
    lines.append("                M[i * %d + j] -= factor * M[k * %d + j];" % (n, n))
    lines.append("            }")
    lines.append("            rhs[i] -= factor * rhs[k];")
    lines.append("        }")
    lines.append("    }")

    lines.append(f"    for (int i = {n - 1}; i >= 0; --i) {{")
    lines.append("        double acc = rhs[i];")
    lines.append(f"        for (int j = i + 1; j < {n}; ++j) {{")
    lines.append("            acc -= M[i * %d + j] * qddot[j];" % n)
    lines.append("        }")
    lines.append("        qddot[i] = acc / M[i * %d + i];" % n)
    lines.append("    }")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _generate_c_full_unrolled(f_q, f_qv, f_vv, n):
    lines = [
        "#include <math.h>",
        "void cal(const double* q, const double* qdot, double* qddot) {",
        f"    double M[{n * n}];",
        f"    double rhs[{n}];",
        "    int pivot;",
        "    double maxv;",
        "    double val;",
    ]

    roots = []
    roots.extend(f_q)
    for i in range(n):
        for j in range(n):
            roots.append(f_qv[i][j])
            roots.append(f_vv[i][j])

    order = _topo_nodes(roots)
    temp_names = {}
    temp_index = 0
    for node in order:
        if node.op in ("const", "var"):
            continue
        tname = f"t{temp_index}"
        temp_index += 1
        temp_names[node] = tname

        if node.op == "add":
            expr = f"{_node_ref(node.a, temp_names)} + {_node_ref(node.b, temp_names)}"
        elif node.op == "sub":
            expr = f"{_node_ref(node.a, temp_names)} - {_node_ref(node.b, temp_names)}"
        elif node.op == "mul":
            expr = f"{_node_ref(node.a, temp_names)} * {_node_ref(node.b, temp_names)}"
        elif node.op == "neg":
            expr = f"-{_node_ref(node.a, temp_names)}"
        elif node.op == "sin":
            expr = f"sin({_node_ref(node.a, temp_names)})"
        elif node.op == "cos":
            expr = f"cos({_node_ref(node.a, temp_names)})"
        else:
            raise ValueError("Unsupported op in codegen")

        lines.append(f"    double {tname} = {expr};")

    for i in range(n):
        lines.append(f"    rhs[{i}] = {_node_ref(f_q[i], temp_names)};")

    for i in range(n):
        for j in range(n):
            lines.append(
                f"    rhs[{i}] -= {_node_ref(f_qv[i][j], temp_names)} * qdot[{j}];"
            )

    for i in range(n):
        for j in range(i, n):
            lines.append(f"    M[{i * n + j}] = {_node_ref(f_vv[i][j], temp_names)};")
            if i != j:
                lines.append(f"    M[{j * n + i}] = M[{i * n + j}];")

    for k in range(n):
        lines.append(f"    pivot = {k};")
        lines.append(f"    maxv = fabs(M[{k * n + k}]);")
        for i in range(k + 1, n):
            lines.append(f"    val = fabs(M[{i * n + k}]);")
            lines.append(f"    if (val > maxv) {{ maxv = val; pivot = {i}; }}")
        lines.append(f"    if (pivot != {k}) {{")
        lines.append("        double tmp;")
        for j in range(k, n):
            lines.append(
                f"        tmp = M[{k * n + j}]; M[{k * n + j}] = M[pivot * {n} + {j}]; M[pivot * {n} + {j}] = tmp;"
            )
        lines.append(f"        tmp = rhs[{k}]; rhs[{k}] = rhs[pivot]; rhs[pivot] = tmp;")
        lines.append("    }")
        lines.append(f"    double diag_{k} = M[{k * n + k}];")
        for i in range(k + 1, n):
            factor_name = f"f_{k}_{i}"
            lines.append(f"    double {factor_name} = M[{i * n + k}] / diag_{k};")
            for j in range(k + 1, n):
                lines.append(
                    f"    M[{i * n + j}] -= {factor_name} * M[{k * n + j}];"
                )
            lines.append(f"    rhs[{i}] -= {factor_name} * rhs[{k}];")

    for i in range(n - 1, -1, -1):
        acc_name = f"acc_{i}"
        lines.append(f"    double {acc_name} = rhs[{i}];")
        for j in range(i + 1, n):
            lines.append(f"    {acc_name} -= M[{i * n + j}] * qddot[{j}];")
        lines.append(f"    qddot[{i}] = {acc_name} / M[{i * n + i}];")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _generate_c_full_const_m(f_q, f_qv, n, solver_kind, solver_values):
    lines = [
        "#include <math.h>",
        "void cal(const double* q, const double* qdot, double* qddot) {",
        f"    double rhs[{n}];",
    ]

    if solver_kind == "chol":
        Lflat = [0.0] * (n * n)
        for i in range(n):
            for j in range(i + 1):
                Lflat[i * n + j] = float(solver_values[i, j])
        _emit_static_array(lines, "Lmat", Lflat)
        lines.append(f"    double y[{n}];")
    else:
        Aflat = [float(v) for v in solver_values.reshape(-1)]
        _emit_static_array(lines, "Ainv", Aflat)

    roots = []
    roots.extend(f_q)
    for i in range(n):
        for j in range(n):
            roots.append(f_qv[i][j])

    order = _topo_nodes(roots)
    temp_names = {}
    temp_index = 0
    for node in order:
        if node.op in ("const", "var"):
            continue
        tname = f"t{temp_index}"
        temp_index += 1
        temp_names[node] = tname

        if node.op == "add":
            expr = f"{_node_ref(node.a, temp_names)} + {_node_ref(node.b, temp_names)}"
        elif node.op == "sub":
            expr = f"{_node_ref(node.a, temp_names)} - {_node_ref(node.b, temp_names)}"
        elif node.op == "mul":
            expr = f"{_node_ref(node.a, temp_names)} * {_node_ref(node.b, temp_names)}"
        elif node.op == "neg":
            expr = f"-{_node_ref(node.a, temp_names)}"
        elif node.op == "sin":
            expr = f"sin({_node_ref(node.a, temp_names)})"
        elif node.op == "cos":
            expr = f"cos({_node_ref(node.a, temp_names)})"
        else:
            raise ValueError("Unsupported op in codegen")

        lines.append(f"    double {tname} = {expr};")

    for i in range(n):
        lines.append(f"    rhs[{i}] = {_node_ref(f_q[i], temp_names)};")

    for i in range(n):
        for j in range(n):
            lines.append(
                f"    rhs[{i}] -= {_node_ref(f_qv[i][j], temp_names)} * qdot[{j}];"
            )

    if solver_kind == "chol":
        lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
        lines.append("        double sum = rhs[i];")
        lines.append("        for (int k = 0; k < i; ++k) {")
        lines.append("            sum -= Lmat[i * %d + k] * y[k];" % n)
        lines.append("        }")
        lines.append("        y[i] = sum / Lmat[i * %d + i];" % n)
        lines.append("    }")
        lines.append(f"    for (int i = {n - 1}; i >= 0; --i) {{")
        lines.append("        double sum = y[i];")
        lines.append(f"        for (int k = i + 1; k < {n}; ++k) {{")
        lines.append("            sum -= Lmat[k * %d + i] * qddot[k];" % n)
        lines.append("        }")
        lines.append("        qddot[i] = sum / Lmat[i * %d + i];" % n)
        lines.append("    }")
    else:
        lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
        lines.append("        double acc = 0.0;")
        lines.append(f"        for (int j = 0; j < {n}; ++j) {{")
        lines.append("            acc += Ainv[i * %d + j] * rhs[j];" % n)
        lines.append("        }")
        lines.append("        qddot[i] = acc;")
        lines.append("    }")

    lines.append("}")
    return "\n".join(lines) + "\n"


_LIB_CACHE = {}


def _compile_and_load(code):
    flag_sets = [
        # CPU-specific (usually best ROI; minimal compile overhead)
        [
            "-Ofast",
            "-ffast-math",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-march=native",
            "-pipe",
        ],
        # CPU-specific + LTO (single TU, often little gain; higher compile cost)
        [
            "-Ofast",
            "-ffast-math",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-march=native",
            "-flto",
            "-pipe",
        ],
        # Fallback: still fast, more portable
        [
            "-Ofast",
            "-ffast-math",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-pipe",
        ],
    ]

    last_exc = None
    for flags in flag_sets:
        flags_str = " ".join(flags)
        digest = hashlib.sha1((flags_str + "\n" + code).encode("utf-8")).hexdigest()[:16]
        lib = _LIB_CACHE.get(digest)
        if lib is not None:
            return lib

        tmp_dir = tempfile.gettempdir()
        c_path = os.path.join(tmp_dir, f"gc_{digest}.c")
        so_path = os.path.join(tmp_dir, f"gc_{digest}.so")

        if not os.path.exists(so_path):
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(code)
            cmd = [
                "gcc",
                *flags,
                "-shared",
                "-fPIC",
                c_path,
                "-o",
                so_path,
                "-lm",
            ]
            try:
                subprocess.check_call(cmd)
            except Exception as exc:
                last_exc = exc
                continue

        try:
            lib = ctypes.CDLL(so_path)
        except Exception as exc:
            last_exc = exc
            try:
                os.unlink(so_path)
            except OSError:
                pass
            continue

        lib.cal.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.cal.restype = None
        _LIB_CACHE[digest] = lib
        return lib

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("C compilation failed")


def _eval_node(node, q, v, cache):
    cached = cache.get(node)
    if cached is not None:
        return cached
    op = node.op
    if op == "const":
        out = node.value
    elif op == "var":
        val = node.value
        if isinstance(val, tuple):
            name, idx = val
            out = q[idx] if name == "q" else v[idx]
        else:
            out = q if val == "q" else v
    elif op == "add":
        out = _eval_node(node.a, q, v, cache) + _eval_node(node.b, q, v, cache)
    elif op == "sub":
        out = _eval_node(node.a, q, v, cache) - _eval_node(node.b, q, v, cache)
    elif op == "mul":
        out = _eval_node(node.a, q, v, cache) * _eval_node(node.b, q, v, cache)
    elif op == "neg":
        out = -_eval_node(node.a, q, v, cache)
    elif op == "sin":
        out = numpy.sin(_eval_node(node.a, q, v, cache))
    elif op == "cos":
        out = numpy.cos(_eval_node(node.a, q, v, cache))
    else:
        raise ValueError("Unsupported op in eval")
    cache[node] = out
    return out


def _const_matrix_solver(f_vv, n):
    dep_cache = {}
    for i in range(n):
        for j in range(n):
            if _node_deps(f_vv[i][j], dep_cache) != 0:
                return None

    q0 = numpy.zeros(n, dtype=numpy.float64)
    v0 = numpy.zeros(n, dtype=numpy.float64)
    M = numpy.empty((n, n), dtype=numpy.float64)
    cache = {}
    for i in range(n):
        for j in range(n):
            M[i, j] = _eval_node(f_vv[i][j], q0, v0, cache)

    try:
        L = numpy.linalg.cholesky(M)
        return ("chol", L)
    except Exception:
        pass

    try:
        inv = numpy.linalg.inv(M)
        return ("inv", inv)
    except Exception:
        return None


def _build_fast(L, n):
    if n <= 0:
        return None

    q_proxy = _TraceArray("q", n)
    v_proxy = _TraceArray("v", n)
    try:
        expr = L(q_proxy, v_proxy, _TraceMath())
        if isinstance(expr, _TraceScalar):
            expr_node = expr.node
        else:
            expr_node = _to_node(expr)
    except Exception:
        return None

    diff_cache = {}
    f_q = [_diff(expr_node, ("q", i), diff_cache) for i in range(n)]
    f_v = [_diff(expr_node, ("v", i), diff_cache) for i in range(n)]

    f_qv = [[None] * n for _ in range(n)]
    f_vv = [[None] * n for _ in range(n)]
    diag_only = True
    for i in range(n):
        for j in range(n):
            dq = _diff(f_v[i], ("q", j), diff_cache)
            f_qv[i][j] = dq
            if i != j:
                if not _is_const(dq, 0.0):
                    diag_only = False
            if j >= i:
                dv = _diff(f_v[i], ("v", j), diff_cache)
                f_vv[i][j] = dv
                if i != j:
                    f_vv[j][i] = dv
                    if not _is_const(dv, 0.0):
                        diag_only = False
                else:
                    if not _is_const(dv, 0.0):
                        diag_only = False
            elif f_vv[i][j] is None:
                f_vv[i][j] = f_vv[j][i]

    if diag_only:
        code = _generate_c_diag(
            f_q,
            [f_qv[i][i] for i in range(n)],
            [f_vv[i][i] for i in range(n)],
            n,
        )
    else:
        const_solver = _const_matrix_solver(f_vv, n)
        if const_solver is not None:
            code = _generate_c_full_const_m(f_q, f_qv, n, const_solver[0], const_solver[1])
        elif n == 20:
            code = _generate_c_full_unrolled(f_q, f_qv, f_vv, n)
        else:
            code = _generate_c_full(f_q, f_qv, f_vv, n)
    try:
        lib = _compile_and_load(code)
        c_cal = lib.cal
        double_p = ctypes.POINTER(ctypes.c_double)

        def cal(q, qdot, qddot):
            c_cal(
                q.ctypes.data_as(double_p),
                qdot.ctypes.data_as(double_p),
                qddot.ctypes.data_as(double_p),
            )

        return cal
    except Exception:
        pass

    if diag_only:
        def cal(q, qdot, qddot):
            for i in range(n):
                cache = {}
                F_i = _eval_node(f_q[i], q, qdot, cache)
                C_i = _eval_node(f_qv[i][i], q, qdot, cache)
                M_i = _eval_node(f_vv[i][i], q, qdot, cache)
                qddot[i] = (F_i - C_i * qdot[i]) / M_i
    else:
        def cal(q, qdot, qddot):
            F = numpy.empty(n, dtype=numpy.float64)
            C = numpy.empty((n, n), dtype=numpy.float64)
            M = numpy.empty((n, n), dtype=numpy.float64)
            for i in range(n):
                cache = {}
                F[i] = _eval_node(f_q[i], q, qdot, cache)
            for i in range(n):
                for j in range(n):
                    cache = {}
                    C[i, j] = _eval_node(f_qv[i][j], q, qdot, cache)
                    cache = {}
                    M[i, j] = _eval_node(f_vv[i][j], q, qdot, cache)
            qddot[:] = numpy.linalg.solve(M, F - C.dot(qdot))

    return cal


def gc(L, n):
    try:
        cal = _build_fast(L, n)
        if cal is not None:
            return cal
    except Exception:
        pass

    if autograd is not None:
        ff_ = autograd.grad(L, 0)

        def ff(a, b):
            return ff_(a, b, autograd.numpy)

        mf_ = autograd.hessian(L, 1)

        def mf(a, b):
            return mf_(a, b, autograd.numpy)

        cf_ = autograd.jacobian(autograd.grad(L, 1), 0)

        def cf(a, b):
            return cf_(a, b, autograd.numpy)

        def cal(q, qdot, qddot):
            F = ff(q, qdot)
            M = mf(q, qdot)
            C = cf(q, qdot)
            qddot[:] = numpy.linalg.solve(M, F - C.dot(qdot))

        return cal

    def cal(q, qdot, qddot):
        n_local = q.shape[0]
        eps = 1e-6
        F = numpy.empty(n_local, dtype=numpy.float64)
        C = numpy.empty((n_local, n_local), dtype=numpy.float64)
        M = numpy.empty((n_local, n_local), dtype=numpy.float64)

        for i in range(n_local):
            dq = numpy.zeros(n_local, dtype=numpy.float64)
            dq[i] = eps
            F[i] = (L(q + dq, qdot, numpy) - L(q - dq, qdot, numpy)) / (2.0 * eps)

        for i in range(n_local):
            dq = numpy.zeros(n_local, dtype=numpy.float64)
            dq[i] = eps
            for j in range(n_local):
                dv = numpy.zeros(n_local, dtype=numpy.float64)
                dv[j] = eps
                c_pp = L(q + dq, qdot + dv, numpy)
                c_pm = L(q + dq, qdot - dv, numpy)
                c_mp = L(q - dq, qdot + dv, numpy)
                c_mm = L(q - dq, qdot - dv, numpy)
                C[i, j] = (c_pp - c_pm - c_mp + c_mm) / (4.0 * eps * eps)

        for i in range(n_local):
            dv_i = numpy.zeros(n_local, dtype=numpy.float64)
            dv_i[i] = eps
            for j in range(n_local):
                dv_j = numpy.zeros(n_local, dtype=numpy.float64)
                dv_j[j] = eps
                m_pp = L(q, qdot + dv_i + dv_j, numpy)
                m_pm = L(q, qdot + dv_i - dv_j, numpy)
                m_mp = L(q, qdot - dv_i + dv_j, numpy)
                m_mm = L(q, qdot - dv_i - dv_j, numpy)
                M[i, j] = (m_pp - m_pm - m_mp + m_mm) / (4.0 * eps * eps)

        qddot[:] = numpy.linalg.solve(M, F - C.dot(qdot))

    return cal
