from typing import Any
class Counter:
    def __init__(self,initial_value=0) -> None:
        super().__init__()
        self.value = initial_value

    def __str__(self) -> str:
        return str(self.value)
    
    def __int__(self):
        return self.value
    
    def __iadd__(self, other):
        self.value+=other
        return self
    
    def __isub__(self, other):
        self.value-=other
        return self
    
    def __eq__(self, __value: Any) -> bool:
        try:
            return self.value == int(__value)
        except (TypeError, ValueError):
            return NotImplemented