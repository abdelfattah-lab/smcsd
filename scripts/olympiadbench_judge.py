"""OlympiadBench mathematical-equivalence grader.

This is a small, typed adaptation of OpenBMB/OlympiadBench's MIT-licensed
``eval/auto_scoring_judge.py``. Keeping it local makes the benchmark adapter
reproducible while preserving the official evaluation order: exact match,
interval, numerical, symbolic-expression, then equation equivalence.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import sympy as sp
from sympy import Eq, Pow, simplify
from sympy.parsing.latex import parse_latex


class OlympiadBenchJudge:
    """Compare a model answer with an OlympiadBench reference answer."""

    def __init__(self):
        self.special_symbol_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8

    @staticmethod
    def split_by_comma(expression: str) -> list[str]:
        """Split on commas that are not inside round or square brackets."""
        depth = 0
        start = 0
        pieces = []
        for index, character in enumerate(expression):
            if character in "([":
                depth += 1
            elif character in ")]":
                depth -= 1
            elif character == "," and depth == 0:
                pieces.append(expression[start:index].strip())
                start = index + 1
        if start < len(expression):
            pieces.append(expression[start:].strip())
        return pieces

    @staticmethod
    def expand_plus_minus(expressions: Sequence[str]) -> list[str]:
        expanded = []
        for expression in expressions:
            if "\\pm" in expression:
                expanded.extend(
                    [expression.replace("\\pm", "+"), expression.replace("\\pm", "-")]
                )
            else:
                expanded.append(expression)
        return expanded

    @staticmethod
    def _extract_boxed_content(text: str) -> str:
        pieces = []
        for match in re.finditer(r"\\boxed\{", text):
            index = match.end()
            depth = 1
            start = index
            while depth and index < len(text):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                index += 1
            if depth:
                raise ValueError("Mismatched braces in boxed answer")
            pieces.append(text[start : index - 1])
        if pieces:
            return ",".join(pieces)

        last_line = text.strip().split("\n")[-1]
        inline_math = re.findall(r"\$(.*?)\$", last_line)
        return ",".join(inline_math) if inline_math else text

    def _replace_special_symbols(self, expression: str) -> str:
        if "\\in " in expression:
            expression = expression.split("\\in ", 1)[1]
        for source, replacement in self.special_symbol_map.items():
            expression = expression.replace(source, replacement)
        expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")
        return re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", expression)

    def preprocess(self, reference: str, prediction: str) -> tuple[str, str]:
        return (
            self._replace_special_symbols(self._extract_boxed_content(reference)),
            self._replace_special_symbols(self._extract_boxed_content(prediction)),
        )

    def judge(
        self,
        reference: str,
        prediction: str,
        precision: float | Sequence[float] = 1e-8,
    ) -> bool:
        """Return official-style mathematical equivalence."""
        try:
            reference, prediction = self.preprocess(reference, prediction)
        except Exception:
            return False
        if reference == prediction:
            return bool(reference)

        reference = re.sub(r"[\u4e00-\u9fff]+", "", reference)
        prediction = re.sub(r"[\u4e00-\u9fff]+", "", prediction)
        references = self.expand_plus_minus(self.split_by_comma(reference))
        predictions = self.expand_plus_minus(self.split_by_comma(prediction))
        precisions = (
            list(precision)
            if isinstance(precision, Sequence) and not isinstance(precision, str)
            else [float(precision)]
        )
        if len(precisions) <= 1:
            precisions *= len(references)
        if len(references) != len(predictions) or len(precisions) != len(references):
            return False

        while references:
            reference_item = references.pop(0)
            self.precision = precisions.pop(0)
            for index, prediction_item in enumerate(predictions):
                if self.is_equal(reference_item, prediction_item):
                    predictions.pop(index)
                    break
            else:
                return False
        return True

    @staticmethod
    def is_interval(expression: str) -> bool:
        return expression.startswith(("(", "[")) and expression.endswith((")", "]"))

    def is_equal(self, reference: str, prediction: str) -> bool:
        if reference == prediction and reference:
            return True
        if self.is_interval(reference) and self.is_interval(prediction):
            try:
                if self.interval_equal(reference, prediction):
                    return True
            except Exception:
                return False
        try:
            if self.numerical_equal(reference, prediction):
                return True
        except Exception:
            pass
        try:
            if not ("=" in reference and "=" in prediction) and self.expression_equal(
                reference, prediction
            ):
                return True
        except Exception:
            pass
        try:
            return self.equation_equal(reference, prediction)
        except Exception:
            return False

    def numerical_equal(self, reference: str, prediction: str) -> bool:
        reference_value = float(reference)
        prediction_value = float(prediction)
        return any(
            abs(candidate - prediction_value) <= self.precision * 1.01
            for candidate in (reference_value / 100, reference_value, reference_value * 100)
        )

    @staticmethod
    def _can_compute_power(expression) -> bool:
        if not isinstance(expression, Pow):
            return True
        base, exponent = expression.as_base_exp()
        return bool(base.is_number and exponent.is_number and abs(exponent.evalf()) <= 1000)

    def expression_equal(self, reference: str, prediction: str) -> bool:
        def expression_part(value: str) -> str:
            return value.split("=", 1)[1].strip() if "=" in value else value.strip()

        reference_sym = parse_latex(expression_part(reference))
        prediction_sym = parse_latex(expression_part(prediction))
        if reference_sym == prediction_sym:
            return True
        reference_sym = reference_sym.subs(self.pi, math.pi)
        prediction_sym = prediction_sym.subs(self.pi, math.pi)
        reference_has_symbol = bool(reference_sym.has(sp.Symbol))
        prediction_has_symbol = bool(prediction_sym.has(sp.Symbol))
        if reference_has_symbol != prediction_has_symbol:
            return False
        if not reference_has_symbol:
            if not (
                self._can_compute_power(reference_sym)
                and self._can_compute_power(prediction_sym)
            ):
                return False
            return bool(
                abs(reference_sym.evalf() - prediction_sym.evalf())
                <= self.precision * 1.01
            )
        try:
            return bool(abs(simplify(reference_sym - prediction_sym).evalf()) < 1e-3)
        except TypeError:
            return False

    @staticmethod
    def equation_equal(reference: str, prediction: str) -> bool:
        def simplify_equation(value: str):
            lhs, rhs = value.split("=")
            equation = Eq(parse_latex(lhs), parse_latex(rhs))
            return simplify(equation.lhs - equation.rhs)

        reference_sym = simplify_equation(reference)
        prediction_sym = simplify_equation(prediction)
        forward = simplify(reference_sym / prediction_sym)
        reverse = simplify(prediction_sym / reference_sym)
        return bool(
            (forward.is_Integer and forward != 0)
            or (reverse.is_Integer and reverse != 0)
        )

    def interval_equal(self, reference: str, prediction: str) -> bool:
        reference_intervals = reference.split("\\cup")
        prediction_intervals = prediction.split("\\cup")
        if len(reference_intervals) != len(prediction_intervals):
            return False
        for reference_interval, prediction_interval in zip(
            reference_intervals, prediction_intervals
        ):
            if (
                reference_interval[0] != prediction_interval[0]
                or reference_interval[-1] != prediction_interval[-1]
            ):
                return False
            reference_items = reference_interval.strip("[]()").split(",")
            prediction_items = prediction_interval.strip("[]()").split(",")
            if len(reference_items) != len(prediction_items):
                return False
            if not all(
                self.expression_equal(left, right)
                for left, right in zip(reference_items, prediction_items)
            ):
                return False
        return True
