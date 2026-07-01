"""
guardrails/
Contract Intelligence — input and output safety guardrails.

Public surface:
    check_input(query, client)                    -> GuardResult
    check_output(answer, context_chunks, client)  -> GuardResult
    GuardResult                                   dataclass (.allowed / .reason / .detail)
"""

from guardrails.input_guard import GuardResult, check_input
from guardrails.output_guard import check_output

__all__ = ["GuardResult", "check_input", "check_output"]
