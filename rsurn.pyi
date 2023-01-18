from typing import List, Tuple

class Gene:
    def __new__(cls: type["Gene"], rho: int, nu: int) -> "Gene": ...

class Environment:
    history: List[Tuple[int, int]]

    def __new__(cls: type["Environment"], gene: Gene) -> "Environment": ...

    def get_caller(self) -> int: ...

    def get_callee(self, caller: int) -> int: ...

    def interact(self, caller: int, callee: int) -> None: ...



