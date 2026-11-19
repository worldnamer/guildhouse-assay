"""assay: shrink a job posting to what matters, then ask Jev whether to apply.

    assay init            create the database (once, before the first run)
    assay resume [file]   save your résumé from a file, or show the saved one
    assay list [ID]       show every saved posting, or one posting's minified text
    assay update ID apply|pass   change the decision on a saved posting
    assay remove ID       delete a saved posting
    assay rebuild         label saved postings and retrain the extractive minifier's classifier
    assay purge           delete assay's data (database and model), after asking; run before uninstalling
    assay export [file]   write every saved posting as JSON lines (to the file, or to stdout)
    assay apply [file...] record postings as ones you applied to (minified with Claude)
    assay pass [file...]  record postings as ones you passed on (minified with Claude)
    assay                 read the clipboard, minify with the extractive minifier
    assay path/to/jd.txt  read a file instead
    assay --claude ...    minify with Claude instead

A plain run saves nothing. Jev's examples are the applied and passed postings in assay.db (next to this
file), which only apply, pass, update and remove change. Everything stops if the database does not exist
or cannot be used. Plain runs need TYPESAFE_API_KEY in the environment.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pyperclip

from assay import jev, paths, store
from assay.classifier import rebuild
from assay.minify import MinifyError, claude, extractive

# Windows terminals default to a legacy encoding, so we select utf-8 for output manually
sys.stdout.reconfigure(encoding="utf-8")

APPLY_AT = 0.75  # at or above: apply
PASS_AT = 0.40   # at or below: pass; in between, read the posting closely
MINIFIERS = {"extractive": extractive, "claude": claude}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="assay", description=__doc__.split("\n")[0],
        epilog="commands:\n  init          create the database; run this once before the first assay\n"
               "  resume [file] save your résumé from a file, or show the saved one\n"
               "  list [ID]     show every saved posting, or one posting's minified text\n"
               "  update ID apply|pass  change the decision on a saved posting\n"
               "  remove ID     delete a saved posting\n"
               "  rebuild       label saved postings with Claude and retrain the classifier\n"
               "  purge         delete assay's data (database and model), after asking\n"
               "  export [file] write every saved posting as JSON lines (to the file, or to stdout)\n"
               "  apply [file...]  record postings you applied to\n  pass [file...]   record postings you passed on",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="job posting file; the clipboard is read if omitted")
    parser.add_argument("--claude", action="store_true", help="minify with Claude instead of the extractive minifier")
    # A bare command word is a command; "./init" or any other path is a file.
    if sys.argv[1:2] == ["init"]:
        if len(sys.argv) > 2:
            parser.error("init takes no arguments")
        return argparse.Namespace(command="init")
    if sys.argv[1:2] == ["resume"]:
        sub = argparse.ArgumentParser(prog="assay resume")
        sub.add_argument("path", nargs="?", help="text file to save as your résumé; shows the saved one if omitted")
        return argparse.Namespace(command="resume", path=sub.parse_args(sys.argv[2:]).path)
    if sys.argv[1:2] == ["purge"]:
        if len(sys.argv) > 2:
            parser.error("purge takes no arguments")
        return argparse.Namespace(command="purge")
    if sys.argv[1:2] == ["rebuild"]:
        if len(sys.argv) > 2:
            parser.error("rebuild takes no arguments")
        return argparse.Namespace(command="rebuild")
    if sys.argv[1:2] == ["list"]:
        sub = argparse.ArgumentParser(prog="assay list")
        sub.add_argument("id", nargs="?", type=int, help="show this posting's minified text instead of the list")
        return argparse.Namespace(command="list", id=sub.parse_args(sys.argv[2:]).id)
    if sys.argv[1:2] == ["export"]:
        sub = argparse.ArgumentParser(prog="assay export")
        sub.add_argument("path", nargs="?", help="file to write; it must not exist. Writes to stdout if omitted")
        return argparse.Namespace(command="export", path=sub.parse_args(sys.argv[2:]).path)
    if sys.argv[1:2] == ["remove"]:
        sub = argparse.ArgumentParser(prog="assay remove")
        sub.add_argument("id", type=int, help="the posting's id, as shown by `assay list`")
        return argparse.Namespace(command="remove", id=sub.parse_args(sys.argv[2:]).id)
    if sys.argv[1:2] == ["update"]:
        sub = argparse.ArgumentParser(prog="assay update")
        sub.add_argument("id", type=int, help="the posting's id, as shown by `assay list`")
        sub.add_argument("decision", choices=list(DECISIONS), help="the new decision")
        parsed = sub.parse_args(sys.argv[2:])
        return argparse.Namespace(command="update", id=parsed.id, decision=parsed.decision)
    if sys.argv[1:2] in (["apply"], ["pass"]):
        sub = argparse.ArgumentParser(prog=f"assay {sys.argv[1]}")
        sub.add_argument("paths", nargs="*", help="job posting files; the clipboard is read if none are given")
        return argparse.Namespace(command=sys.argv[1], paths=sub.parse_args(sys.argv[2:]).paths)
    args = parser.parse_args()
    args.command = None
    return args


class PostingError(Exception):
    """A posting could not be read, or was empty."""


def read_posting(path):
    """Return (text, source): the file at path, or the clipboard if no path was given.

    Raises PostingError if the file cannot be read or there is nothing in it."""
    if path:
        try:
            raw, source = Path(path).read_text(encoding="utf-8"), path
        except (OSError, UnicodeDecodeError) as e:
            raise PostingError(f"Could not read {path}: {e}")
    else:
        raw, source = pyperclip.paste(), "clipboard"
    if not raw.strip():
        raise PostingError("Nothing to read. Copy a job description first, or pass a file.")
    return raw, source


def choose_minifier(args):
    """Return the name of the minifier to use; it is also what gets saved with the run."""
    return "claude" if args.claude else "extractive"


def minify_posting(name, raw, show=True):
    """Minify the posting with the named minifier, showing the result unless show is False. Returns the text.

    Raises MinifyError if the minifier fails or produces nothing."""
    if show:
        print(f"Minifying ({name})...")
    minified = MINIFIERS[name].minify(raw)
    if not minified.strip():
        raise MinifyError("Minification produced nothing.")
    if show:
        print(f"Minified ({len(minified.split())} words, of {len(raw.split())}):\n")
        print(minified)
        print()
    return minified


def recommendation(apply_prob):
    if apply_prob >= APPLY_AT:
        return "APPLY"
    if apply_prob <= PASS_AT:
        return "PASS"
    return "DIG IN (read the posting closely)"


def print_result(result):
    """Show Jev's recommendation for the posting."""
    decision = result["answers"]["decision"]
    apply_prob = decision["probabilities"]["apply"]
    print("=" * 40)
    print(f"Recommendation: {recommendation(apply_prob)}")
    print(f"Apply probability: {apply_prob:.0%}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print(f"Tokens used: {result['usage']['input_tokens']}")


def init_database():
    existed, created = store.init()
    if not existed:
        print(f"Created {paths.db_path()}.")
    elif created:
        print(f"Updated {paths.db_path()}: added {', '.join(created)}. Your data is untouched.")
    else:
        print(f"{paths.db_path()} is up to date; nothing changed.")
    if store.get_resume(store.connect()) is None:
        print("No résumé saved yet. Load one with `assay resume FILE`.")


def resume_command(path):
    """assay resume FILE saves the résumé; assay resume alone shows it."""
    db = store.connect()
    if path is None:
        saved = store.get_resume(db)
        if saved is None:
            raise SystemExit("No résumé saved. Use `assay resume FILE`.")
        print(f"Résumé: {len(saved['text'])} characters, saved {saved['updated_at'][:10]} from {saved['source']}\n")
        print(saved["text"])
        return
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"Could not read {path}: {e}")
    if not text.strip():
        raise SystemExit(f"{path} is empty.")
    previous = store.set_resume(db, text, path)
    if previous is None:
        print(f"Résumé saved ({len(text)} characters).")
    else:
        print(f"Résumé replaced ({len(previous)} characters before, {len(text)} now). The old one is not kept.")


DECISIONS = {"apply": "applied", "pass": "passed"}


def title_of(posting, width=60):
    """The first line of the minified text, which is usually the job title and company."""
    lines = [line.strip() for line in posting["minified_text"].splitlines() if line.strip()]
    title = lines[0] if lines else ""
    return title if len(title) <= width else title[:width - 1] + "…"


def show_posting(posting_id):
    """assay list ID: the posting's details and its minified text."""
    posting = find_or_exit(store.connect(), posting_id)
    print(f"Posting {posting['id']} · {posting['decision']} · saved {posting['created_at'][:10]} · from {posting['source']}\n")
    print(posting["minified_text"])


def list_postings():
    """assay list: one line per saved posting."""
    postings = store.list_postings(store.connect())
    if not postings:
        print("No postings saved yet. Use `assay apply` or `assay pass`.")
        return
    print(f"{'id':>4}  {'decision':<8}  {'saved':<10}  {'words':>5}  posting")
    for p in postings:
        title = title_of(p)
        print(f"{p['id']:>4}  {p['decision']:<8}  {p['created_at'][:10]}  {len(p['minified_text'].split()):>5}  {title}")
    applied = sum(p["decision"] == "applied" for p in postings)
    print(f"\n{len(postings)} postings: {applied} applied, {len(postings) - applied} passed")


def export_postings(path):
    """assay export: the résumé, then every saved posting, one JSON object per line with every column.
    Each line says which table it came from."""
    db = store.connect()
    records = [{"table": "resume", **resume} for resume in [store.get_resume(db)] if resume]
    records += [{"table": "postings", **posting} for posting in store.all_postings(db)]
    if path is None:
        out, close = sys.stdout, False
    else:
        try:
            # "x" refuses to overwrite; mode 600 because this is the raw postings and your decisions.
            out = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8")
        except FileExistsError:
            raise SystemExit(f"{path} already exists; not overwriting it.")
        except OSError as e:
            raise SystemExit(f"Could not write {path}: {e}")
        close = True
    try:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if close:
            out.close()
    kinds = [record["table"] for record in records]
    print(f"Exported {kinds.count('postings')} postings and {kinds.count('resume')} résumé"
          + (f" to {path}." if path else "."), file=sys.stderr)


def rebuild_model():
    summary = rebuild.rebuild(store.connect())
    print(f"\nRebuilt {paths.model_path()}")
    print(f"  {summary['newly_labeled']} postings newly labeled; trained on {summary['postings']} postings, "
          f"{summary['sentences']} sentences ({summary['keep']} keep, {summary['cut']} cut), {summary['clues']} word clues.")


def purge_data():
    # Delete only what assay itself creates, so a mistaken ASSAY_HOME cannot take other files with it.
    folder = paths.data_dir()
    targets = sorted(folder.glob("assay.db*")) + [folder / "model"]
    targets = [t for t in targets if t.exists()]
    if not targets:
        print(f"Nothing to delete: {folder} holds no assay data.")
        return
    print(f"This permanently deletes assay's database (postings, résumé, labels) and trained model in {folder}.")
    print("Run `assay export FILE` first if you want to keep your postings.")
    try:
        answer = input("Type 'yes' to delete: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() != "yes":
        raise SystemExit("Not deleted.")
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    try:
        folder.rmdir()  # only succeeds if empty
    except OSError:
        pass
    print(f"Deleted assay's data in {folder}. You can now run `pipx uninstall assay`.")


def find_or_exit(db, posting_id):
    posting = store.get_posting(db, posting_id)
    if posting is None:
        raise SystemExit(f"No posting with id {posting_id}. See `assay list`.")
    return posting


def remove_posting(posting_id):
    db = store.connect()
    posting = find_or_exit(db, posting_id)
    store.delete_posting(db, posting_id)
    print(f"Removed posting {posting_id} ({posting['decision']}): {title_of(posting)}")


def update_posting(posting_id, decision):
    db = store.connect()
    posting = find_or_exit(db, posting_id)
    if posting["decision"] == decision:
        print(f"Posting {posting_id} is already {decision}.")
        return
    store.set_decision(db, posting_id, decision)
    print(f"Posting {posting_id} changed from {posting['decision']} to {decision}.")


def record_posting(db, decision, path, show):
    """Save one posting with the decision, minified by Claude. Returns the message to show."""
    raw, source = read_posting(path)

    existing = store.find_posting(db, raw)
    if existing and existing["minifier"] == "claude":
        # Already minified by Claude on an earlier run: keep that text and skip the call.
        store.set_decision(db, existing["id"], decision)
    else:
        minified = minify_posting("claude", raw, show)
        if existing:
            store.set_decision(db, existing["id"], decision, "claude", minified)
        else:
            store.save_posting(db, source, raw, "claude", minified, decision)

    if existing and existing["decision"] and existing["decision"] != decision:
        return f"Changed from {existing['decision']} to {decision}."
    if existing and existing["decision"] == decision:
        return f"Already recorded as {decision}."
    return f"Recorded as {decision}."


def record_decision(args):
    """assay apply / assay pass: save each posting with your decision. With several files, a failure is
    reported and the rest still run."""
    decision = DECISIONS[args.command]
    db = store.connect()

    paths = args.paths or [None]  # no file means the clipboard
    many = len(paths) > 1
    failed = 0
    for path in paths:
        label = f"{path}: " if many else ""
        try:
            print(label + record_posting(db, decision, path, show=not many))
        except (MinifyError, PostingError) as e:
            print(f"{label}FAILED: {e}", file=sys.stderr)
            failed += 1
    if many:
        print(f"\n{len(paths) - failed} of {len(paths)} recorded as {decision}.")
    if failed:
        raise SystemExit(1)


def run(args):
    if args.command == "init":
        init_database()
        return
    if args.command == "resume":
        resume_command(args.path)
        return
    if args.command == "list":
        if args.id is None:
            list_postings()
        else:
            show_posting(args.id)
        return
    if args.command == "remove":
        remove_posting(args.id)
        return
    if args.command == "export":
        export_postings(args.path)
        return
    if args.command == "purge":
        purge_data()
        return
    if args.command == "rebuild":
        rebuild_model()
        return
    if args.command == "update":
        update_posting(args.id, DECISIONS[args.decision])
        return
    if args.command in DECISIONS:
        record_decision(args)
        return

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("Set TYPESAFE_API_KEY in your environment.")

    db = store.connect()  # Jev's résumé and examples come from the database
    saved_resume = store.get_resume(db)
    if saved_resume is None:
        raise SystemExit("No résumé saved. Load one with `assay resume FILE`.")
    resume = saved_resume["text"]
    applied, passed = store.examples(db, "applied"), store.examples(db, "passed")
    if not applied or not passed:
        raise SystemExit("Jev needs at least one applied and one passed posting. Add some with `assay apply` and `assay pass`.")

    raw, _source = read_posting(args.path)

    minified = minify_posting(choose_minifier(args), raw)
    sample_applied, sample_passed = jev.select_examples(minified, resume, applied, passed)
    used, total = len(sample_applied) + len(sample_passed), len(applied) + len(passed)
    if used < total:
        print(f"Too many examples for Jev to read at once: using a random sample of {used} of {total} "
              f"({len(sample_applied)} applied, {len(sample_passed)} passed), the same sample every time for this posting.\n")
    result = jev.evaluate(minified, key, resume, sample_applied, sample_passed)
    print_result(result)
    if result["usage"]["input_tokens"] > 0.9 * jev.CONTEXT_LIMIT:
        print(f"\nWarning: Jev read {result['usage']['input_tokens']} tokens, close to its {jev.CONTEXT_LIMIT} limit. "
              "The size estimate used for sampling may be too low.", file=sys.stderr)


def main():
    try:
        run(parse_args())
    except (MinifyError, jev.JevError, store.StoreError, rebuild.RebuildError, PostingError) as e:
        raise SystemExit(str(e))
