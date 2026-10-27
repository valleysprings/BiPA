import re

SKILL_FRAME = re.compile(
    r'(?P<open><(?P<namespace>[A-Za-z0-9_.-]+)-skill\b'
    r'(?=[^>]*\spath=(?P<quote>\\*")(?P<path>.+?)(?P=quote))?[^>]*>)'
    r'(?P<body>.*?)(?P<close></(?P=namespace)-skill>)', re.S)
