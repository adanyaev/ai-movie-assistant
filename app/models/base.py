from typing import Annotated
import datetime

from sqlalchemy.types import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import DeclarativeBase

created_at = Annotated[
    datetime.datetime,
    mapped_column(
        default=datetime.datetime.now(datetime.UTC),
        type_=TIMESTAMP(timezone=True)
    ),
]


class Base(DeclarativeBase):

    repr_cols_num = 3
    repr_cols = tuple()

    def __repr__(self):

        cols = []
        for idx, col in enumerate(self.__table__.columns.keys()):
            if col in self.repr_cols or idx < self.repr_cols_num:
                cols.append(f"{col}={getattr(self, col)}")

        return f"<{self.__class__.__name__} {', '.join(cols)}>"
