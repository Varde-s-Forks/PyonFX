from __future__ import annotations

__all__: List[str] = []

import sys
from abc import ABC, ABCMeta
from enum import IntEnum
from threading import Lock
from typing import Any, Callable, Dict, List, NoReturn, TypeVar, overload

import loguru

loguru.logger.remove(0)


F = TypeVar('F', bound=Callable[..., Any])


class LogLevel(IntEnum):
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    USER_WARNING = 60
    USER_INFO = 70


def _loguru_format(record: loguru.Record) -> str:
    if record['extra']['user'] and record['level'].no >= 60 and record['extra']['level'] >= 40:
        return '<level>{message}</level>\n'

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level.name: <12}</level> | "
        "<cyan>{name}</cyan>:<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n{exception}"
    )


class SingletonMeta(ABCMeta):
    _instances: Dict[object, Any] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(ABC, metaclass=SingletonMeta):
    ...


class Logger(Singleton):
    __slots__ = ('__id', '__level')

    def __init__(self) -> None:
        self.__level = 40
        ids_ = loguru.logger.configure(
            handlers=[
                dict(sink=sys.stderr, level=self.__level, format=_loguru_format, backtrace=True, diagnose=True)
            ],
            levels=[  # type: ignore
                dict(name='USER WARNING', no=LogLevel.USER_WARNING, color='<yellow><bold>'),
                dict(name='USER INFO', no=LogLevel.USER_INFO, color='<white><bold>')
            ],
            extra=dict(user=False, level=self.__level)
        )
        self.__id = ids_.pop(0)

    def set_level(self, level: int) -> None:
        loguru.logger.remove(self.__id)
        self.__level = level
        loguru.logger.add(sys.stderr, level=level, format=_loguru_format, backtrace=True, diagnose=True)

    def trace(self, message: Any, /, depth: int = 1) -> None:
        loguru.logger.opt(depth=depth).bind(user=False, level=self.__level).trace(str(message))

    def debug(self, message: Any, /, depth: int = 1) -> None:
        loguru.logger.opt(depth=depth).bind(user=False, level=self.__level).debug(str(message))

    def info(self, message: Any, /, depth: int = 1) -> None:
        loguru.logger.opt(depth=depth).bind(user=False, level=self.__level).info(str(message))

    def success(self, message: Any, /, depth: int = 1) -> None:
        loguru.logger.opt(depth=depth).bind(user=False, level=self.__level).success(str(message))

    def warning(self, message: Any, /, depth: int = 1) -> None:
        loguru.logger.opt(depth=depth).bind(user=False, level=self.__level).warning(str(message))

    def error(self, message: Any, /) -> NoReturn:
        loguru.logger.exception(str(message))
        sys.exit(1)

    def user_warning(self, message: Any, /, depth: int = 1) -> None:
        loguru.logger.opt(depth=depth).bind(user=True, level=self.__level).log('USER WARNING', str(message))

    def user_info(self, message: Any, /, depth: int = 1) -> None:
        loguru.logger.opt(depth=depth).bind(user=True, level=self.__level).log('USER INFO', str(message))

    @overload
    def catch(self, func: F, /) -> F:
        ...

    @overload
    def catch(self, /, *, force_exit: bool = ...) -> Callable[[F], F]:
        ...

    def catch(self, func: F | None = None, /, *, force_exit: bool = False) -> F | Callable[[F], F]:
        if func is None:
            return loguru.logger.catch(onerror=(lambda _: sys.exit(1)) if force_exit else None)
        return loguru.logger.catch(func)


logger: Logger = Logger()
