import sys

for line in sys.stdin:
    if line.strip() == "":
        continue
    else:
        line = line.replace(u'\ufeff', '')
        line = line.replace('_', '')
        line = " ".join(line.split())
        sys.stdout.write(line + "\n")
