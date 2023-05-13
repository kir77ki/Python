# Implementation of some mappers and reducers for tables

from abc import abstractmethod, ABC
import typing as tp
import string
import itertools
from collections import defaultdict
import heapq
import re
from collections.abc import Generator

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper(row)


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        yield from self.reducer(tuple(self.keys), rows)
        # else:
        #     flag = True
        #     current_group: tp.List[tp.Any] = []
        #     rows_lst: tp.Any = []
        #     for row in rows:
        #         group = [row[key] for key in self.keys]
        #         if flag:
        #             current_group = group
        #             flag = False
        #
        #         if current_group != group:
        #             yield from self.reducer(tuple(self.keys), rows_lst)
        #             rows_lst = []
        #             current_group = group
        #
        #         rows_lst.append(row)
        #     yield from self.reducer(tuple(self.keys), rows_lst)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        def func(x: dict[tp.Any, tp.Any]) -> list[tp.Any]:
            return [x[key] for key in self.keys]

        function = func

        first_iter = itertools.groupby(rows, function)
        second_iter = itertools.groupby(args[0], function)

        first_key = next(first_iter, None)
        second_key = next(second_iter, None)

        while first_key is not None and second_key is not None:
            first_value, first_object = first_key
            second_value, second_object = second_key
            if first_value < second_value:
                yield from self.joiner(self.keys, first_object, iter([]))
                first_key = next(first_iter, None)
            elif second_value < first_value:
                yield from self.joiner(self.keys, iter([]), second_object)
                second_key = next(second_iter, None)
            else:
                yield from self.joiner(self.keys, first_object, second_object)
                first_key = next(first_iter, None)
                second_key = next(second_iter, None)

        while first_key is not None:
            first_value, first_object = first_key
            yield from self.joiner(self.keys, first_object, iter([]))
            first_key = next(first_iter, None)

        while second_key is not None:
            second_value, second_object = second_key
            yield from self.joiner(self.keys, iter([]), second_object)
            second_key = next(second_iter, None)


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        flag = True
        current_group: tp.List[tp.Any] = []
        rows_lst: tp.Any = []
        for row in rows:
            group = [row[key] for key in group_key]
            if flag:
                current_group = group
                flag = False

            if current_group != group:
                for row1 in rows_lst:
                    yield row1
                    break
                rows_lst = []
                current_group = group

            if len(rows_lst) == 0:
                rows_lst.append(row)
        for row2 in rows_lst:
            yield row2
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = row[self.column].translate(str.maketrans('', '', string.punctuation))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str) -> None:
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: tp.Optional[str] = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def sep_func(self, string: str, sep: tp.Optional[str] = None) -> Generator[str, None, None]:
        if sep is not None:
            if sep in string:
                index = 0
                while True:
                    fnd = string.find(sep, index)
                    if fnd == -1:
                        yield string[index:]
                        break
                    yield string[index:fnd]
                    index = fnd + 1
            else:
                yield string
        else:
            starting = 0
            for match in re.finditer(r'[ \n\t\u00A0]', string):
                ending = match.start()
                yield string[starting:ending]
                starting = match.end()
            yield string[starting:]

    def __call__(self, row: TRow) -> TRowsGenerator:
        for word in self.sep_func(row[self.column], self.separator):
            new_row = row.copy()
            new_row[self.column] = word
            yield new_row


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        res = 1
        for col in self.columns:
            res *= row[col]
        row[self.result_column] = res
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        new_row = {key: value for (key, value) in row.items() if key in self.columns}
        yield new_row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        flag = True
        current_group: tp.List[tp.Any] = []
        rows_lst: tp.Any = []
        c = 0
        for row in rows:
            c += 1
            group = [row[key] for key in group_key]
            if flag:
                current_group = group
                flag = False

            if current_group != group:
                sorted_rows = sorted(list(rows_lst), reverse=True)
                for row1 in sorted_rows[:self.n]:
                    yield row1[2]
                rows_lst = []
                current_group = group

            heapq.heappush(rows_lst, (row[self.column_max], c, row))
            if len(rows_lst) == self.n + 1:
                heapq.heappop(rows_lst)
        sorted_rows = sorted(list(rows_lst), reverse=True)
        for row2 in sorted_rows[:self.n]:
            yield row2[2]


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        groups: dict[tp.Any, tp.Any] = defaultdict(int)
        groups_keys = defaultdict(list)
        for row in rows:
            group = ''
            for key in group_key:
                group += str(row[key]) + ' '
            groups[group] += 1
            if group not in groups_keys:
                for key in group_key:
                    groups_keys[group].append(row[key])

        for group in groups:
            new_row = {self.column: groups[group]}
            for i in range(len(group_key)):
                new_row[group_key[i]] = groups_keys[group][i]
            yield new_row


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        fst = True
        cur_group_key: tp.List[tp.Any] = []
        dct: tp.Dict[str, int] = dict()
        total = 0
        for row in rows:
            group_key1 = [row[key] for key in group_key]

            if fst:
                cur_group_key = group_key1
                fst = False

            if cur_group_key == group_key1:
                total += 1
                if row[self.words_column] not in dct:
                    dct[row[self.words_column]] = 1
                else:
                    dct[row[self.words_column]] += 1

            if cur_group_key != group_key1:
                for el in dct:
                    new_row = dict()
                    for i, key in enumerate(group_key):
                        new_row[key] = cur_group_key[i]
                    new_row[self.result_column] = dct[el] / total
                    new_row[self.words_column] = el
                    yield new_row
                cur_group_key = group_key1
                total = 1
                dct = dict()
                dct[row[self.words_column]] = 1
        for el in dct:
            new_row = dict()
            for i, key in enumerate(group_key):
                new_row[key] = cur_group_key[i]
            new_row[self.result_column] = dct[el] / total
            new_row[self.words_column] = el
            yield new_row


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        groups: dict[tp.Any, tp.Any] = defaultdict(int)
        groups_keys = defaultdict(list)
        for row in rows:
            group = ''
            for key in group_key:
                group += str(row[key]) + ' '
            groups[group] += row[self.column]
            if group not in groups_keys:
                for key in group_key:
                    groups_keys[group].append(row[key])

        for group in groups:
            new_row = {self.column: groups[group]}
            for i in range(len(group_key)):
                new_row[group_key[i]] = groups_keys[group][i]
            yield new_row


# Joiners


def merge_rows(keys: tp.Sequence[str], row1: TRow, row2: TRow, suf1: str, suf2: str) -> TRow:
    answ = row1.copy()
    for key2, val2 in row2.items():
        if key2 not in answ:
            answ[key2] = val2
        elif key2 not in keys:
            val1 = answ[key2]
            del answ[key2]
            answ[key2 + suf1] = val1
            answ[key2 + suf2] = val2
    return answ


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows_b_ = list(rows_b)
        for row_a in rows_a:
            for row_b in rows_b_:
                flag = True
                for key in keys:
                    if row_a[key] != row_b[key]:
                        flag = False
                        break
                if flag:
                    yield merge_rows(keys, row_a, row_b, self._a_suffix, self._b_suffix)


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows_b_ = list(rows_b)
        rows_a, rows_a1 = itertools.tee(rows_a, 2)
        if next(rows_a1, None) is None:
            for row_b in rows_b_:
                yield row_b.copy()
        elif rows_b_ == []:
            for row_a in rows_a:
                yield row_a.copy()
        else:
            for row_a in rows_a:
                for row_b in rows_b_:
                    yield merge_rows(keys, row_a, row_b, self._a_suffix, self._b_suffix)


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows_b_ = list(rows_b)

        if rows_b_ == []:
            for row_a in rows_a:
                yield row_a.copy()
        else:
            for row_a in rows_a:
                flag1 = False
                for row_b in rows_b_:
                    flag = True
                    for key in keys:
                        if row_a[key] != row_b[key]:
                            flag = False
                            break
                    if flag:
                        yield merge_rows(keys, row_a, row_b, self._a_suffix, self._b_suffix)
                        flag1 = True
                if not flag1:
                    yield row_a


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows_a_ = list(rows_a)

        if rows_a_ == []:
            for row_b in rows_b:
                yield row_b.copy()
        else:
            for row_b in rows_b:
                flag1 = False
                for row_a in rows_a_:
                    flag = True
                    for key in keys:
                        if row_a[key] != row_b[key]:
                            flag = False
                            break
                    if flag:
                        yield merge_rows(keys, row_a, row_b, self._a_suffix, self._b_suffix)
                        flag1 = True
                if not flag1:
                    yield row_b
