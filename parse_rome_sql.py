import re
from typing import Dict, Iterator, List, Tuple, Any


CREATE_TABLE_RE = re.compile(r"^CREATE TABLE `(?P<name>[^`]+)`")
INSERT_INTO_RE = re.compile(r"^INSERT INTO `(?P<name>[^`]+)`", re.IGNORECASE)


def _parse_columns_from_create(lines: Iterator[str]) -> List[str]:
    cols: List[str] = []
    for line in lines:
        line = line.strip()
        if line.startswith(")"):
            break
        if line.startswith("`"):
            # Column line: `name` type ... ,
            end = line.find("`", 1)
            if end > 1:
                cols.append(line[1:end])
    return cols


def _collect_insert_statement(first_line: str, lines: Iterator[str]) -> str:
    # Gather lines until we hit a semicolon ending the INSERT
    buf = [first_line.rstrip("\n")]
    paren = first_line.count("(") - first_line.count(")")
    if ";" in first_line and paren <= 0:
        return first_line
    for line in lines:
        buf.append(line.rstrip("\n"))
        paren += line.count("(") - line.count(")")
        if ";" in line and paren <= 0:
            break
    return "\n".join(buf)


def _split_values_groups(stmt: str) -> List[str]:
    # Extract the section after VALUES
    p = stmt.upper().find(" VALUES ")
    if p == -1:
        return []
    rest = stmt[p + len(" VALUES ") :]

    # Split at top-level comma between groups '(...),(...)'
    groups: List[str] = []
    depth = 0
    in_quote = False
    esc = False
    start = None
    for i, ch in enumerate(rest):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == "'":
            in_quote = not in_quote
        elif not in_quote:
            if ch == "(":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and start is not None:
                    groups.append(rest[start : i + 1])
                    start = None
    return groups


def _split_fields(group: str) -> List[str]:
    assert group.startswith("(") and group.endswith(")"), "Invalid group"
    inner = group[1:-1]
    fields: List[str] = []
    buf: List[str] = []
    in_quote = False
    esc = False
    for ch in inner:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if ch == "'":
            buf.append(ch)
            in_quote = not in_quote
            continue
        if ch == "," and not in_quote:
            fields.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        fields.append("".join(buf).strip())
    return fields


def _unquote_sql_string(val: str) -> str:
    val = val.strip()
    if val.upper() == "NULL":
        return None  # type: ignore
    if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
        s = val[1:-1]
        # Unescape SQL backslash sequences for readability
        s = s.replace("\\'", "'")
        return s
    return val


def parse_table_from_sql(
    sql_path: str, table_name: str
) -> Tuple[List[str], Iterator[List[str]]]:
    """Parse CREATE TABLE columns and yield INSERT values lists for a table."""
    def gen_rows() -> Iterator[List[str]]:
        with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
            lines = iter(f)
            columns: List[str] = []
            for line in lines:
                m_ct = CREATE_TABLE_RE.match(line.strip())
                if m_ct and m_ct.group("name") == table_name:
                    columns = _parse_columns_from_create(lines)
                    break
            # Now iterate for INSERTS
            for line in lines:
                m_ins = INSERT_INTO_RE.match(line.strip())
                if not (m_ins and m_ins.group("name") == table_name):
                    continue
                stmt = _collect_insert_statement(line, lines)
                for g in _split_values_groups(stmt):
                    fields = _split_fields(g)
                    yield fields

    # First pass just to get columns
    columns: List[str] = []
    with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_ct = CREATE_TABLE_RE.match(line.strip())
            if m_ct and m_ct.group("name") == table_name:
                columns = _parse_columns_from_create(f)
                break
    return columns, gen_rows()


def iter_metier_front(sql_path: str) -> Iterator[Dict[str, Any]]:
    columns, rows_iter = parse_table_from_sql(sql_path, "metier_front")
    for row in rows_iter:
        rec: Dict[str, Any] = {}
        for col, raw in zip(columns, row):
            rec[col] = _unquote_sql_string(raw)
        yield rec

