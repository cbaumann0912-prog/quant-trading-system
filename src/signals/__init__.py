"""
Strategy signal construction.

Every module here returns a signal aligned so that it may only use
information available strictly before the return it is paired with; the
lag contract is enforced centrally in `signal_builder`.
"""
