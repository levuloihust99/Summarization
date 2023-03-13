import os
import re

from .exceptions import UnsupportedModeError, NotEnoughMaxLinesPerFile, BufferExceedError


class PandasParser:
    def __init__(self, working_file, mode='binary', sep=',', header=True):
        self.mode = mode
        if mode not in ['binary', 'text']:
            raise UnsupportedModeError("Unsupported mode: `{}`.".format(mode))

        if hasattr(working_file, "read"):
            self._reader = working_file
        elif self.mode == "text":
            self._reader = open(working_file, "r")
        elif self.mode == "binary":
            self._reader = open(working_file, "rb")

        if self.mode == 'binary':
            self._newline = b'\n'
            self._sep = sep.encode()
            self._double_quote = b'"'
            self._blank = b''
            self._carriage_return = b'\r'
        else:
            self._newline = '\n'
            self._sep = sep
            self._double_quote = '"'
            self._blank = ''
            self._carriage_return = '\r'

        if header:
            headers = self._next_record()
            self.headers = [h.decode() for h in headers] if self.mode == 'binary' else headers
            self.next_record = self.named_column_decorator(self._next_record)
        else:
            self.next_record = self._next_record

    def __iter__(self):
        return self

    def __next__(self):
        record = self.next_record()
        if self._reader.closed:
            raise StopIteration()
        return record

    def named_column_decorator(self, func):
        def wrapper(*args, **kwargs):
            record = func(*args, **kwargs)
            return dict(zip(self.headers, record))
        return wrapper

    def _next_record(self):
        res = [] # return value
        is_within_double_quotes = False # check if current position is in double quotes
        col = self._blank # capture one column

        while True:
            ch = self._reader.read(1)
            if not ch: # end of file
                self._reader.close()
                if not is_within_double_quotes:
                    res.append(col)
                break

            if ch == self._double_quote: # encounter double quote
                if not is_within_double_quotes: # if not within double quotes, a double quote means begin of a column.
                    is_within_double_quotes = True
                    continue
                next_ch = self._reader.read(1) # check the next character
                if not next_ch: # end of file
                    self._reader.close()
                    if not is_within_double_quotes:
                        res.append(col)
                    break
                if next_ch != self._double_quote: # this is not string quote
                    is_within_double_quotes = not is_within_double_quotes
                if is_within_double_quotes: # if inside double quotes, append this character to the captured column
                    col += next_ch
                elif next_ch == self._newline: # outside double quotes, '\n' means end of record
                    res.append(col)
                    col = self._blank
                    break
                elif next_ch == self._sep:
                    res.append(col)
                    col = self._blank
                        
            elif ch == self._newline: # encounter newline
                if not is_within_double_quotes: # outside double quotes, '\n' means end of record 
                    res.append(col)
                    col = self._blank
                    break
                else: # inside double quotes, '\n' is nothing different from other characters
                    col += ch
            
            elif ch == self._sep:
                if not is_within_double_quotes: # outside double quotes, ',' means the column seperator
                    res.append(col)
                    col = self._blank
                else: # inside double quotes, ',' is nothing different from other characters
                    col += ch

            elif ch == self._carriage_return:
                if is_within_double_quotes: # inside double quotes, '\r' is nothing different from other characters
                    col += ch

            else: # non-special characters
                col += ch

        return res

