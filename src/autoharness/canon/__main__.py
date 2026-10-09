"""python -m autoharness.canon run | recover | rollback <name> | status"""
import json
import sys

from autoharness.canon import gate, release


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "status"
    if cmd == "run":
        result = gate.run_once()
    elif cmd == "recover":
        result = release.recover()
    elif cmd == "rollback" and len(argv) == 2:
        result = release.rollback(argv[1])
    elif cmd == "status":
        result = release.journal_entries()[-10:]
    else:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
