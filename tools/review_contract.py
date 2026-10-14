# Purpose: keep review slices small and make Python documentation mechanically checkable.
# Public API: documentation_errors(source, path), review_errors(pr, files, api), main().
# Consumers: the trusted GitHub review-contract workflow and local migration audits.
# Format: modules use Purpose/Public API/Consumers; classes use Purpose/Consumers;
# functions use Purpose/Callers immediately above decorators or the definition.
# Types: annotate all function parameters/results and module/class data bindings;
# local expressions retain Python's inferred types. Any is for dynamic boundary data.
# Scope: every changed Python file is checked; --audit inventories the full codebase.

from __future__ import annotations

import argparse
import ast
import base64
import copy
import json
import os
import re
import subprocess
import unittest
from pathlib import Path
from typing import Any, Callable


# Purpose: isolate the nearest comment block instead of accepting unrelated file comments.
# Callers: documentation_errors for module, class and function contracts.
def comment_block(lines: list[str], before: int) -> str:
    comments: list[str] = []
    position = before - 1
    while position >= 0 and lines[position].strip().startswith("#"):
        comments.append(lines[position].strip()[1:].strip())
        position -= 1
    return "\n".join(reversed(comments))


# Purpose: reject empty labels so a marker alone cannot satisfy documentation policy.
# Callers: documentation_errors; prose quality still requires human review.
def missing_labels(block: str, required: tuple[str, ...]) -> list[str]:
    return [
        label
        for label in required
        if not re.search(rf"^{re.escape(label)}:[ \t]*\S.+$", block, re.MULTILINE)
    ]


# Purpose: expose local documentation and annotation gaps without executing Python code.
# Callers: the workflow on fetched source text and --audit on local tracked Python files.
def documentation_errors(source: str, path: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        return [f"{path}:{error.lineno}: invalid Python syntax"]

    lines = source.splitlines()
    header_end = 0
    while header_end < len(lines) and lines[header_end].strip().startswith("#"):
        header_end += 1
    missing = missing_labels(
        comment_block(lines, header_end), ("Purpose", "Public API", "Consumers")
    )
    if missing:
        errors.append(f"{path}:1: module comments missing {', '.join(missing)}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                errors.append(f"{path}:{getattr(node, 'lineno', 1)}: use comments, not docstrings")
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first_line = min([node.lineno] + [item.lineno for item in node.decorator_list])
            labels = (
                ("Purpose", "Consumers")
                if isinstance(node, ast.ClassDef)
                else ("Purpose", "Callers")
            )
            missing = missing_labels(comment_block(lines, first_line - 1), labels)
            if missing:
                errors.append(
                    f"{path}:{node.lineno}: {node.name} comments missing {', '.join(missing)}"
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
            for argument in arguments:
                if argument.arg not in {"self", "cls"} and argument.annotation is None:
                    errors.append(f"{path}:{node.lineno}: {node.name}.{argument.arg} needs a type")
            if node.returns is None:
                errors.append(f"{path}:{node.lineno}: {node.name} needs a return type")
        if isinstance(node, (ast.Module, ast.ClassDef)):
            annotated = {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                for target in statement.targets:
                    for binding in ast.walk(target):
                        if isinstance(binding, ast.Name) and binding.id not in annotated:
                            errors.append(
                                f"{path}:{statement.lineno}: {binding.id} needs a data type"
                            )
    return errors


# Purpose: read GitHub metadata with existing authentication and redact transport failures.
# Callers: review_errors and main; no shell evaluation or source execution is performed.
def github_api(endpoint: str) -> Any:
    result = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("GitHub metadata read failed; check workflow read permissions")
    return json.loads(result.stdout)


# Purpose: include all changed paths and fail closed if GitHub truncates its file listing.
# Callers: main and parent verification inside review_errors.
def changed_files(
    repository: str, number: int, expected: int, api: Callable[[str], Any]
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = api(f"repos/{repository}/pulls/{number}/files?per_page=100&page={page}")
        files.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    if len(files) != expected:
        raise ValueError("Incomplete changed-file evidence; refusing a partial review")
    return files


# Purpose: allow a large integration PR only when its exact tree was assembled from merged slices.
# Callers: review_errors; parent metadata cannot waive unreviewed or extra changes.
def parent_errors(
    pr: dict[str, Any], files: list[dict[str, Any]], api: Callable[[str], Any]
) -> list[str]:
    body = pr.get("body") or ""
    branch_match = re.search(r"^Review-Parent:\s*(review/[A-Za-z0-9_/-]+)\s*$", body, re.MULTILINE)
    children_match = re.search(r"^Review-Children:\s*((?:#\d+\s*)+)\s*$", body, re.MULTILINE)
    if not branch_match or not children_match:
        return ["Parent requires Review-Parent: review/... and Review-Children: #N #N metadata"]

    repository = pr["base"]["repo"]["full_name"]
    parent_branch = branch_match.group(1)
    child_numbers = [int(value) for value in re.findall(r"#(\d+)", children_match.group(1))]
    if pr["number"] in child_numbers or len(set(child_numbers)) != len(child_numbers):
        return ["Parent child list contains duplicates or itself"]

    reviewed_paths: set[str] = set()
    reviewed_commits: set[str] = set()
    errors: list[str] = []
    for number in child_numbers:
        child = api(f"repos/{repository}/pulls/{number}")
        if not child.get("merged") or child["base"]["ref"] != parent_branch:
            errors.append(f"Child #{number} must be merged into {parent_branch}")
            continue
        if (child.get("merged_by") or {}).get("login") != "rykunk21":
            errors.append(f"Child #{number} must be accepted into the parent by Ryan")
        reviewed_commits.add(child["merge_commit_sha"])
        child_commits = api(f"repos/{repository}/pulls/{number}/commits?per_page=100")
        if len(child_commits) != child["commits"]:
            errors.append(f"Child #{number} commit evidence is incomplete")
        reviewed_commits.update(commit["sha"] for commit in child_commits)
        if child["changed_files"] > 2:
            errors.append(f"Child #{number} exceeds the two-file limit")
        for item in changed_files(repository, number, child["changed_files"], api):
            reviewed_paths.add(item["filename"])
            if item.get("previous_filename"):
                reviewed_paths.add(item["previous_filename"])

    if any(item["filename"] not in reviewed_paths for item in files):
        errors.append("Parent includes paths not covered by its merged review slices")
    parent_commit = api(f"repos/{repository}/commits/{parent_branch}")
    comparison = api(
        f"repos/{repository}/compare/{pr['base']['sha']}...{parent_commit['sha']}?per_page=100"
    )
    if len(comparison["commits"]) != comparison["total_commits"]:
        errors.append("Integration commit evidence is incomplete")
    if any(commit["sha"] not in reviewed_commits for commit in comparison["commits"]):
        errors.append("Integration branch includes commits outside its reviewed child PRs")
    head_repository = pr["head"]["repo"]["full_name"]
    head_commit = api(f"repos/{head_repository}/commits/{pr['head']['sha']}")
    if parent_commit["commit"]["tree"]["sha"] != head_commit["commit"]["tree"]["sha"]:
        errors.append("Parent PR tree differs from the reviewed integration branch")
    return errors


# Purpose: combine review-size and Python contract checks on every changed file.
# Callers: main; parent exceptions still undergo documentation checks.
def review_errors(
    pr: dict[str, Any], files: list[dict[str, Any]], api: Callable[[str], Any]
) -> list[str]:
    errors: list[str] = []
    if len(files) > 2:
        errors.extend(parent_errors(pr, files, api))
    for item in files:
        path = item["filename"]
        if item["status"] == "removed" or not path.endswith(".py"):
            continue
        # The immutable blob SHA binds the check to this exact PR revision.
        repository = pr["head"]["repo"]["full_name"]
        blob = api(f"repos/{repository}/git/blobs/{item['sha']}")
        if blob.get("encoding") != "base64":
            raise ValueError("Unsupported GitHub blob encoding")
        source = base64.b64decode(blob["content"], validate=False).decode("utf-8")
        errors.extend(documentation_errors(source, path))
    return errors


# Purpose: support an offline full inventory and the read-only GitHub workflow entry point.
# Callers: python tools/review_contract.py --audit ROOT or --pr NUMBER --repository OWNER/REPO.
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check review size, inline contracts and annotations"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    args = parser.parse_args()
    errors: list[str] = []
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PolicyChecks)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.audit:
        paths = (
            subprocess.run(
                ["git", "ls-files", "-z", "*.py"], cwd=args.audit, capture_output=True, check=True
            )
            .stdout.decode()
            .split("\0")
        )
        for path in filter(None, paths):
            errors.extend(documentation_errors((args.audit / path).read_text(), path))
    elif args.pr and args.repository and re.fullmatch(r"[\w.-]+/[\w.-]+", args.repository):
        pr = github_api(f"repos/{args.repository}/pulls/{args.pr}")
        files = changed_files(args.repository, args.pr, pr["changed_files"], github_api)
        errors = review_errors(pr, files, github_api)
    else:
        parser.error("Select --audit ROOT or --pr NUMBER --repository OWNER/REPO")
    print(json.dumps({"status": "failed" if errors else "passed", "errors": errors}, indent=2))
    return 1 if errors else 0


# Purpose: exercise the policy's trust boundaries without credentials or network calls.
# Consumers: --self-test and the GitHub workflow; fixtures never execute proposed source.
class PolicyChecks(unittest.TestCase):
    # Purpose: provide a minimal contract that should survive the syntax-only validator.
    # Callers: the documentation acceptance and rejection cases below.
    def valid_source(self) -> str:
        return (
            "# Purpose: compute fixture values.\n# Public API: example(value).\n"
            "# Consumers: offline tests.\n\n"
            "# Purpose: preserve a fixture value.\n# Callers: offline tests.\n"
            "def example(value: int) -> int:\n    return value\n"
        )

    # Purpose: ensure comment-only documentation and annotations are accepted together.
    # Callers: unittest via --self-test.
    def test_documented_function(self) -> None:
        self.assertEqual(documentation_errors(self.valid_source(), "fixture.py"), [])

    # Purpose: prove missing contracts, docstrings and partial annotations fail separately.
    # Callers: unittest via --self-test; mutations represent common policy omissions.
    def test_documentation_rejections(self) -> None:
        source = self.valid_source()
        cases = [
            source.replace("# Public API: example(value).", "# Public API:"),
            source.replace("# Callers: offline tests.", "# unrelated comment"),
            source.replace("value: int", "value"),
            source.replace(" -> int", ""),
            source.replace("    return value", '    """docstring"""\n    return value'),
            source + "\nCONSTANT = 1\n",
            source + "\ndef broken(:\n",
        ]
        for case in cases:
            with self.subTest(source=case):
                self.assertTrue(documentation_errors(case, "fixture.py"))

    # Purpose: check decorated async methods, keyword-only parameters and typed constants.
    # Callers: unittest via --self-test.
    def test_decorators_and_async(self) -> None:
        source = self.valid_source() + (
            "\nCONSTANT: int = 1\n"
            "# Purpose: expose fixture behavior.\n# Consumers: offline tests.\n"
            "class Example:\n"
            "    # Purpose: process a fixture.\n    # Callers: tests.\n"
            "    @staticmethod\n"
            "    async def run(*values: int, flag: bool, **options: str) -> None:\n"
            "        pass\n"
        )
        self.assertEqual(documentation_errors(source, "fixture.py"), [])
        self.assertTrue(documentation_errors(source.replace("flag: bool", "flag"), "fixture.py"))

    # Purpose: verify an ordinary two-file PR is checked without executing its source.
    # Callers: unittest via --self-test; the API stub supplies one immutable Python blob.
    def test_two_files_and_blob(self) -> None:
        pr = {"head": {"repo": {"full_name": "owner/repo"}}}
        files = [
            {"filename": "example.py", "status": "added", "sha": "blob"},
            {"filename": "guide.txt", "status": "added", "sha": "text"},
        ]
        source = self.valid_source() + "\nraise RuntimeError('must never execute')\n"
        encoded = base64.b64encode(source.encode()).decode()
        self.assertEqual(
            review_errors(pr, files, lambda _: {"encoding": "base64", "content": encoded}), []
        )

    # Purpose: prevent a body marker from bypassing the file limit before child review.
    # Callers: unittest via --self-test.
    def test_large_pr_requires_parent_evidence(self) -> None:
        files = [{"filename": f"{index}.txt", "status": "added"} for index in range(3)]
        self.assertTrue(review_errors({"body": "Looks reviewed"}, files, lambda _: None))
        pr = {
            "number": 16,
            "body": "Review-Parent: review/issue-14\nReview-Children: #16",
            "base": {"repo": {"full_name": "owner/repo"}},
        }
        self.assertTrue(parent_errors(pr, files, lambda _: None))

    # Purpose: prevent partial GitHub file lists from silently omitting changed code.
    # Callers: unittest via --self-test.
    def test_file_list_truncation(self) -> None:
        with self.assertRaises(ValueError):
            changed_files("owner/repo", 1, 2, lambda _: [])

    # Purpose: require complete reviewed ancestry and an identical integration tree.
    # Callers: unittest via --self-test; each mutation removes one handoff prerequisite.
    def test_parent_provenance(self) -> None:
        pr = {
            "number": 16,
            "body": "Review-Parent: review/issue-14\nReview-Children: #21 #22",
            "base": {"sha": "base", "repo": {"full_name": "owner/repo"}},
            "head": {"sha": "head", "repo": {"full_name": "fork/repo"}},
        }
        files = [{"filename": name} for name in ("one.txt", "two.txt", "three.txt")]
        evidence: dict[str, Any] = {}
        for number, child_files in ((21, files[:2]), (22, files[2:])):
            endpoint = f"repos/owner/repo/pulls/{number}"
            evidence[endpoint] = {
                "merged": True,
                "merged_by": {"login": "rykunk21"},
                "base": {"ref": "review/issue-14"},
                "commits": 1,
                "merge_commit_sha": f"merge{number}",
                "changed_files": len(child_files),
            }
            evidence[endpoint + "/commits?per_page=100"] = [{"sha": f"commit{number}"}]
            evidence[endpoint + "/files?per_page=100&page=1"] = child_files
        evidence["repos/owner/repo/commits/review/issue-14"] = {
            "sha": "tip",
            "commit": {"tree": {"sha": "tree"}},
        }
        evidence["repos/fork/repo/commits/head"] = {"commit": {"tree": {"sha": "tree"}}}
        comparison_endpoint = "repos/owner/repo/compare/base...tip?per_page=100"
        evidence[comparison_endpoint] = {
            "total_commits": 4,
            "commits": [{"sha": value} for value in ("commit21", "merge21", "commit22", "merge22")],
        }
        self.assertEqual(parent_errors(pr, files, evidence.__getitem__), [])
        for mutation in (
            "not_merged",
            "wrong_reviewer",
            "unreviewed_commit",
            "wrong_tree",
            "truncated",
        ):
            changed = copy.deepcopy(evidence)
            if mutation == "not_merged":
                changed["repos/owner/repo/pulls/21"]["merged"] = False
            elif mutation == "wrong_reviewer":
                changed["repos/owner/repo/pulls/21"]["merged_by"]["login"] = "someone_else"
            elif mutation == "unreviewed_commit":
                changed[comparison_endpoint]["commits"][0]["sha"] = "extra"
            elif mutation == "wrong_tree":
                changed["repos/fork/repo/commits/head"]["commit"]["tree"]["sha"] = "different"
            else:
                changed[comparison_endpoint]["total_commits"] = 5
            with self.subTest(mutation=mutation):
                self.assertTrue(parent_errors(pr, files, changed.__getitem__))

    # Purpose: keep this policy compliant with its own public documentation requirements.
    # Callers: unittest via --self-test.
    def test_policy_documents_itself(self) -> None:
        self.assertEqual(documentation_errors(Path(__file__).read_text(), __file__), [])


if __name__ == "__main__":
    raise SystemExit(main())
