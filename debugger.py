"""Approval-first bug investigation CLI using bounded project context."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".next",
}
SOURCE_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".php", ".rb", ".swift", ".kt", ".html", ".css", ".scss",
    ".json", ".toml", ".yaml", ".yml", ".md", ".sql", ".sh",
}
MAX_TREE_ENTRIES = 400
MAX_CONTEXT_CHARS = 10_000


class DebugState(TypedDict, total=False):
    root: str
    bug_report: str
    tree: str
    attempts: int
    search_path: str
    search_terms: list[str]
    search_result: str
    inspected_files: dict[str, str]
    inspected_excerpts: dict[str, str]
    inspected_hashes: dict[str, str]
    decision: dict[str, Any]
    status: str
    test_suggestion: str


def _inside_root(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("The requested path is outside the project root.") from exc
    return candidate


def _project_tree(root: Path) -> str:
    entries: list[str] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirs[:] = sorted(
            name for name in dirs
            if name not in IGNORED_DIRS and not (current_path / name).is_symlink()
        )
        for name in sorted(dirs + files):
            path = current_path / name
            if path.is_symlink():
                continue
            if path.is_file() and path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            entries.append(path.relative_to(root).as_posix() + ("/" if path.is_dir() else ""))
            if len(entries) >= MAX_TREE_ENTRIES:
                entries.append("... (tree truncated)")
                return "\n".join(entries)
    return "\n".join(entries) if entries else "(no source files found)"


def _parse_json(content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        content = str(content)
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model did not return a JSON object.")
    value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from the model.")
    return value


def _snippets(
    root: Path, requested_path: str, terms: list[str]
) -> tuple[str, dict[str, str], dict[str, str], dict[str, str]]:
    target = _inside_root(root, requested_path or ".")
    if not target.exists():
        return f"Path not found: {requested_path}", {}, {}, {}

    candidates = [target] if target.is_file() else [
        path for path in target.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and not any(part in IGNORED_DIRS for part in path.relative_to(root).parts)
    ]
    candidates = sorted(
        path for path in candidates
        if path.suffix.lower() in SOURCE_SUFFIXES and not path.is_symlink()
    )[:150]
    needles = [term.strip().casefold() for term in terms[:8] if term.strip()]
    if not needles:
        return "No search terms were provided.", {}, {}, {}

    excerpts: list[str] = []
    inspected: dict[str, str] = {}
    inspected_excerpts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    used_chars = 0
    for path in candidates:
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(root).as_posix()
        lines = text.splitlines()
        hit_lines = [
            index for index, line in enumerate(lines)
            if any(term in line.casefold() for term in needles)
        ]
        if not hit_lines:
            continue
        windows: list[tuple[int, int]] = []
        for index in hit_lines:
            start, end = max(0, index - 3), min(len(lines), index + 4)
            if windows and start <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(windows[-1][1], end))
            else:
                windows.append((start, end))
        file_excerpts: list[str] = []
        for start, end in windows[:3]:
            source_excerpt = "\n".join(lines[start:end])
            excerpt = f"--- {relative}:{start + 1} ---\n" + "\n".join(
                f"{number + 1}: {lines[number]}" for number in range(start, end)
            )
            if used_chars + len(excerpt) > MAX_CONTEXT_CHARS:
                break
            excerpts.append(excerpt)
            file_excerpts.append(source_excerpt)
            used_chars += len(excerpt)
        if file_excerpts:
            inspected[relative] = text
            inspected_excerpts[relative] = "\n\n".join(file_excerpts)
            hashes[relative] = hashlib.sha256(raw).hexdigest()
        if used_chars >= MAX_CONTEXT_CHARS:
            break

    if not excerpts:
        return f"No matching code found under {requested_path} for terms: {', '.join(needles)}", {}, {}, {}
    return "\n\n".join(excerpts), inspected, inspected_excerpts, hashes


def _validate_edits(
    edits: Any, inspected: dict[str, str], excerpts: dict[str, str]
) -> tuple[list[dict[str, str]], str | None]:
    if not isinstance(edits, list) or not edits:
        return [], "The proposed fix contains no exact edits."
    valid: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edit in edits:
        if not isinstance(edit, dict):
            return [], "An edit was not a JSON object."
        path, old, new = edit.get("path"), edit.get("old"), edit.get("new")
        if not all(isinstance(value, str) and value for value in (path, old)) or not isinstance(new, str):
            return [], "Each edit needs a path, a non-empty old string, and a new string."
        if path not in inspected:
            return [], f"The model proposed editing a file that was not inspected: {path}"
        source = inspected[path].replace("\r\n", "\n").replace("\r", "\n")
        normalized_old = old.replace("\r\n", "\n").replace("\r", "\n")
        if source.count(normalized_old) != 1:
            return [], f"The old text for {path} must occur exactly once in the inspected file."
        excerpt = excerpts.get(path, "").replace("\r\n", "\n").replace("\r", "\n")
        if normalized_old not in excerpt:
            return [], f"The old text for {path} was not included in the model's search excerpts."
        key = (path, old)
        if key in seen:
            return [], f"Duplicate edit for {path}."
        seen.add(key)
        valid.append({"path": path, "old": old, "new": new})
    return valid, None


def _apply_edits(source: str, edits: list[dict[str, str]]) -> str:
    updated = source
    newline = "\r\n" if "\r\n" in source else "\r" if "\r" in source else "\n"
    for edit in edits:
        old = edit["old"].replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
        new = edit["new"].replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
        updated = updated.replace(old, new, 1)
    return updated


def build_graph(model: ChatGroq):
    def inspect(state: DebugState) -> DebugState:
        return {"tree": _project_tree(Path(state["root"]))}

    def reason(state: DebugState) -> DebugState:
        attempts = state.get("attempts", 0)
        decision_prompt = f"""You are investigating a reported software bug. Use only the supplied project tree and matching code excerpts.
Return one JSON object with these fields:
For a search: {{"action":"search","path":"relative folder or file from tree","terms":["distinct code terms"]}}
For a fix: {{"action":"propose","cause":"...","fix":"...","edits":[{{"path":"relative file path","old":"exact existing text","new":"replacement text"}}],"test":"focused test or check to run"}}
Search at most three times total. Choose the narrowest useful path. Do not invent code: each old string must appear verbatim in the excerpts and each edit path must be among excerpt paths. Keep edits minimal; never return a whole-file rewrite. If evidence is insufficient, search again rather than guessing. Use action search when no safe, evidence-backed fix is possible.

Bug report:
{state['bug_report']}

Project tree:
{state['tree']}

Search attempt: {attempts} of 3
Previous search result:
{state.get('search_result', '(none yet)')}
"""
        response = model.invoke(decision_prompt)
        decision = _parse_json(response.content)
        action = decision.get("action")
        if action == "search":
            terms = decision.get("terms", [])
            if not isinstance(terms, list):
                terms = []
            return {
                "decision": decision,
                "search_path": str(decision.get("path", ".")),
                "search_terms": [str(term)[:80] for term in terms[:8]],
            }
        if action == "propose":
            edits, error = _validate_edits(
                decision.get("edits"),
                state.get("inspected_files", {}),
                state.get("inspected_excerpts", {}),
            )
            if error:
                return {
                    "decision": {"action": "search"},
                    "search_result": f"Previous proposed fix rejected: {error}",
                }
            decision["edits"] = edits
            return {"decision": decision}
        return {"decision": {"action": "stop", "feedback": "The model returned an unsupported action."}}

    def search(state: DebugState) -> DebugState:
        root = Path(state["root"])
        try:
            result, inspected, excerpts, hashes = _snippets(
                root, state.get("search_path", "."), state.get("search_terms", [])
            )
        except (OSError, ValueError) as exc:
            result, inspected, excerpts, hashes = f"Search failed safely: {exc}", {}, {}, {}
        return {
            "attempts": state.get("attempts", 0) + 1,
            "search_result": result,
            "inspected_files": inspected,
            "inspected_excerpts": excerpts,
            "inspected_hashes": hashes,
        }

    def approve(state: DebugState) -> DebugState:
        proposal = state["decision"]
        print("\nSuspected cause:\n" + str(proposal.get("cause", "Not provided")))
        print("\nProposed fix:\n" + str(proposal.get("fix", "Not provided")))
        for edit in proposal["edits"]:
            print(f"\n--- {edit['path']} ---\n- {edit['old']}\n+ {edit['new']}")
        answer = input("\nApply these exact changes? [y/N] ").strip().casefold()
        return {"status": "approved" if answer == "y" else "rejected"}

    def apply(state: DebugState) -> DebugState:
        root = Path(state["root"])
        originals: dict[str, str] = {}
        edits_by_path: dict[str, list[dict[str, str]]] = {}
        for edit in state["decision"]["edits"]:
            edits_by_path.setdefault(edit["path"], []).append(edit)
        for relative, edits in edits_by_path.items():
            path = _inside_root(root, relative)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != state["inspected_hashes"].get(relative):
                return {"status": f"aborted: {relative} changed after inspection; no changes were applied"}
            try:
                original = raw.decode("utf-8")
            except UnicodeDecodeError:
                return {"status": f"aborted: {relative} is no longer valid UTF-8; no changes were applied"}
            for edit in edits:
                normalized_original = original.replace("\r\n", "\n").replace("\r", "\n")
                normalized_old = edit["old"].replace("\r\n", "\n").replace("\r", "\n")
                if normalized_original.count(normalized_old) != 1:
                    return {"status": f"aborted: exact text check failed for {relative}; no changes were applied"}
            originals[relative] = original

        replacements = {
            relative: _apply_edits(original, edits_by_path[relative])
            for relative, original in originals.items()
        }
        for relative, updated in replacements.items():
            _inside_root(root, relative).write_text(updated, encoding="utf-8", newline="")
        return {
            "status": "applied",
            "test_suggestion": str(state["decision"].get("test", "No focused test suggested.")),
        }

    def route(state: DebugState) -> str:
        action = state.get("decision", {}).get("action")
        if action == "propose":
            return "approve"
        if action == "search" and state.get("attempts", 0) < 3:
            return "search"
        return "finish"

    graph = StateGraph(DebugState)
    graph.add_node("inspect", inspect)
    graph.add_node("reason", reason)
    graph.add_node("search", search)
    graph.add_node("approve", approve)
    graph.add_node("apply", apply)
    graph.add_edge(START, "inspect")
    graph.add_edge("inspect", "reason")
    graph.add_conditional_edges("reason", route, {"search": "search", "approve": "approve", "finish": END})
    graph.add_edge("search", "reason")
    graph.add_conditional_edges(
        "approve",
        lambda state: "apply" if state.get("status") == "approved" else "finish",
        {"apply": "apply", "finish": END},
    )
    graph.add_edge("apply", END)
    return graph.compile()


def main() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in the environment or a local .env file.")
    root = Path.cwd().resolve()
    bug_report = input("Describe the bug: ").strip()
    if not bug_report:
        raise SystemExit("A bug report is required.")

    model = ChatGroq(
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
        temperature=0,
    )
    result = build_graph(model).invoke(
        {"root": str(root), "bug_report": bug_report, "attempts": 0, "status": "investigating"}
    )
    status = result.get("status", "no evidence-backed fix found")
    print(f"\nResult: {status}")
    if status == "applied":
        print("Tests were not run. Suggested check: " + result.get("test_suggestion", ""))
    elif result.get("attempts", 0) >= 3:
        print("No safe fix was identified within three searches; no files were changed.")


if __name__ == "__main__":
    main()