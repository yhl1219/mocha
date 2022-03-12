class MochaException(Exception):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return self.__str__()


class MochaSyntaxException(MochaException):
    def __init__(self, line: int, context: str, filename: str, message: str) -> None:
        super().__init__()
        self.line = line
        self.context = context
        self.message = message
        self.filename = filename

    def __str__(self) -> str:
        return f'Syntax Exception in {self.filename}, ln {self.line}: {self.message}'


class MochaTypeException(MochaException):
    def __init__(self, expected: type, got: type) -> None:
        super().__init__()
        self.expected, self.got = expected, got

    def __str__(self) -> str:
        return f'Type Exception: expecting {self.expected}, but got {self.got}'
