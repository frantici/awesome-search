"""
Code symbol index engine: ownership/group-aware search + optional folder watch.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import faiss
import numpy as np

from awesome_search.code_classify import (
    VALID_GROUPS,
    classify_group,
    detect_language,
    is_indexable_source,
)
from awesome_search.code_rules import CodeRuleStore
from awesome_search.code_symbols import CodeSymbol, extract_symbols, refresh_line_numbers


SKIP_DIR_NAMES = {
    ".git",
    ".awesome_index",
    ".svn",
    ".hg",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
}


class CodeSearchEngine:
    PRIMARY_ENGINE = "ivf_emeans"
    FALLBACK_ENGINE = "flat_l2"
    MIN_VECTORS_FOR_IVF = 39
    EMEANS_NEPOCHS = 5

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.index_folder = self.project_root / ".awesome_index"
        self.code_folder = self.index_folder / "code"
        self.symbols_path = self.code_folder / "symbols.pkl"
        self.faiss_path = self.code_folder / "symbols.faiss"
        self.hash_path = self.code_folder / "file_hashes.json"
        self.watch_status_path = self.code_folder / "watch.status.json"
        self.watch_cmd_path = self.code_folder / "watch.cmd.json"
        self.embedding_dim = 384
        self.default_nprobe = 10
        self.rules = CodeRuleStore(self.project_root, self.code_folder)
        self._model = None

    # --- embeddings -----------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)

    # --- filesystem helpers ---------------------------------------------

    def _iter_source_files(self, root: Optional[Path] = None) -> Iterable[Path]:
        base = Path(root) if root else self.project_root
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel_parts = path.relative_to(self.project_root).parts
            except ValueError:
                continue
            if any(part in SKIP_DIR_NAMES for part in rel_parts):
                continue
            if not is_indexable_source(path):
                continue
            yield path

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def _load_hashes(self) -> Dict[str, str]:
        if self.hash_path.exists():
            with open(self.hash_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_hashes(self, hashes: Dict[str, str]) -> None:
        self.code_folder.mkdir(parents=True, exist_ok=True)
        with open(self.hash_path, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)

    def _load_symbols(self) -> List[CodeSymbol]:
        if not self.symbols_path.exists():
            return []
        with open(self.symbols_path, "rb") as f:
            data = pickle.load(f)
        return [CodeSymbol(**item) if isinstance(item, dict) else item for item in data["symbols"]]

    def _save_symbols(self, symbols: List[CodeSymbol], engine_meta: Optional[Dict] = None) -> None:
        self.code_folder.mkdir(parents=True, exist_ok=True)
        with open(self.symbols_path, "wb") as f:
            pickle.dump(
                {
                    "symbols": [s.to_dict() for s in symbols],
                    "engine": engine_meta,
                    "embedding_dim": self.embedding_dim,
                },
                f,
            )

    # --- indexing -------------------------------------------------------

    def index_file(self, path: Path) -> List[CodeSymbol]:
        rel = str(path.relative_to(self.project_root)).replace("\\", "/")
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        language = detect_language(path, content)
        resolved = self.rules.resolve(rel)
        ownership = resolved["ownership"] or "unknown"
        group = classify_group(
            rel,
            ownership,
            self.rules,
            group_override=resolved.get("group_override"),
        )
        return extract_symbols(
            relative_path=rel,
            content=content,
            language=language,
            ownership=ownership,
            group=group,
        )

    def _build_faiss(self, embeddings: np.ndarray):
        vectors = np.ascontiguousarray(embeddings.astype("float32"))
        n, dim = vectors.shape
        if n < self.MIN_VECTORS_FOR_IVF:
            index = faiss.IndexFlatL2(dim)
            index.add(vectors)
            return index, {
                "engine": self.FALLBACK_ENGINE,
                "trainer": None,
                "nlist": 0,
                "nprobe": 0,
                "num_vectors": n,
            }

        nlist = max(1, min(int(4 * np.sqrt(n)), n // 39, n))
        from faiss.contrib.e_means import NumpyEmeans

        em = NumpyEmeans(dim, nlist, nepochs=self.EMEANS_NEPOCHS)
        centroids, _ = em.train(vectors)
        quantizer = faiss.IndexFlatL2(dim)
        quantizer.add(np.ascontiguousarray(centroids.astype("float32")))
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        if not index.is_trained:
            index.is_trained = True
        index.add(vectors)
        nprobe = min(self.default_nprobe, nlist)
        index.nprobe = nprobe
        return index, {
            "engine": self.PRIMARY_ENGINE,
            "trainer": "e_means",
            "nlist": nlist,
            "nprobe": nprobe,
            "nepochs": self.EMEANS_NEPOCHS,
            "num_vectors": n,
        }

    def index_codebase(
        self,
        root: Optional[Path] = None,
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict:
        files = list(self._iter_source_files(root))
        old_hashes = {} if force else self._load_hashes()
        new_hashes: Dict[str, str] = {}
        symbols: List[CodeSymbol] = []

        unchanged_cache: Dict[str, List[CodeSymbol]] = {}
        if not force and self.symbols_path.exists():
            for sym in self._load_symbols():
                unchanged_cache.setdefault(sym.relative_path, []).append(sym)

        changed = 0
        reused = 0
        for path in files:
            rel = str(path.relative_to(self.project_root)).replace("\\", "/")
            digest = self._file_hash(path)
            new_hashes[rel] = digest
            if not force and old_hashes.get(rel) == digest and rel in unchanged_cache:
                # Re-resolve ownership/group in case rules changed
                resolved = self.rules.resolve(rel)
                ownership = resolved["ownership"] or "unknown"
                group = classify_group(
                    rel, ownership, self.rules, resolved.get("group_override")
                )
                for sym in unchanged_cache[rel]:
                    sym.ownership = ownership
                    sym.group = group
                    symbols.append(sym)
                reused += 1
                continue

            file_syms = self.index_file(path)
            symbols.extend(file_syms)
            changed += 1

        if not symbols:
            self._save_symbols([])
            self._save_hashes(new_hashes)
            if self.faiss_path.exists():
                self.faiss_path.unlink()
            return {
                "files": len(files),
                "changed": changed,
                "reused": reused,
                "symbols": 0,
            }

        if show_progress:
            print(f"Embedding {len(symbols)} code symbols...")
        blobs = [s.search_blob() for s in symbols]
        embeddings = self._embed(blobs, show_progress=show_progress)
        index, engine_meta = self._build_faiss(embeddings)
        self.code_folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.faiss_path))
        self._save_symbols(symbols, engine_meta)
        self._save_hashes(new_hashes)
        return {
            "files": len(files),
            "changed": changed,
            "reused": reused,
            "symbols": len(symbols),
            "engine": engine_meta,
        }

    def refresh_path(self, path: Path) -> int:
        """Quick update for one written file: refresh symbols + rebuild index."""
        path = Path(path).resolve()
        if not path.exists() or not is_indexable_source(path):
            return 0
        try:
            path.relative_to(self.project_root)
        except ValueError:
            return 0

        rel = str(path.relative_to(self.project_root)).replace("\\", "/")
        content = path.read_text(encoding="utf-8", errors="ignore")
        language = detect_language(path, content)
        previous = self._load_symbols()
        existing = [s for s in previous if s.relative_path != rel]
        old_for_file = [s for s in previous if s.relative_path == rel]
        if old_for_file:
            new_syms = refresh_line_numbers(content, language, old_for_file)
            if not new_syms:
                new_syms = self.index_file(path)
        else:
            new_syms = self.index_file(path)

        all_syms = existing + new_syms
        hashes = self._load_hashes()
        hashes[rel] = self._file_hash(path)
        if not all_syms:
            self._save_symbols([])
            self._save_hashes(hashes)
            return 0

        embeddings = self._embed([s.search_blob() for s in all_syms])
        index, engine_meta = self._build_faiss(embeddings)
        faiss.write_index(index, str(self.faiss_path))
        self._save_symbols(all_syms, engine_meta)
        self._save_hashes(hashes)
        return len(new_syms)

    # --- search ---------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        ownership: Optional[str] = None,
        group: Optional[str] = None,
        exclude_groups: Optional[List[str]] = None,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        mine_only: bool = False,
    ) -> List[Dict]:
        if not self.faiss_path.exists() or not self.symbols_path.exists():
            raise FileNotFoundError(
                "Code index not found. Run: awesomeSearch code index"
            )

        symbols = self._load_symbols()
        index = faiss.read_index(str(self.faiss_path))
        if hasattr(index, "nprobe") and getattr(index, "nlist", 0):
            index.nprobe = min(self.default_nprobe, index.nlist)

        with open(self.symbols_path, "rb") as f:
            meta = pickle.load(f).get("engine") or {}

        if meta.get("nprobe") and hasattr(index, "nprobe"):
            index.nprobe = min(meta["nprobe"], getattr(index, "nlist", meta["nprobe"]))

        q = self._embed([query]).astype("float32")
        search_k = min(max(k * 20, k), max(1, index.ntotal))
        distances, indices = index.search(q, search_k)

        exclude_groups = [g.lower() for g in (exclude_groups or [])]
        if ownership:
            ownership = ownership.lower()
        if group:
            group = group.lower()
        if language:
            language = language.lower()
        if path_prefix:
            path_prefix = path_prefix.replace("\\", "/").lstrip("./")

        # Default agent-friendly behavior: if asking about app UI and no flags,
        # still require explicit mine_only/group from CLI. We don't auto-exclude
        # libraries unless asked — but expose --mine and --exclude-group.

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(symbols):
                continue
            sym = symbols[idx]

            if mine_only and sym.ownership != "mine":
                continue
            if ownership and sym.ownership != ownership:
                continue
            if group and sym.group != group:
                continue
            if exclude_groups and sym.group in exclude_groups:
                continue
            if language and sym.language != language:
                continue
            if path_prefix and not sym.relative_path.startswith(path_prefix):
                continue

            results.append(
                {
                    "score": float(dist),
                    "name": sym.name,
                    "kind": sym.kind,
                    "language": sym.language,
                    "ownership": sym.ownership,
                    "group": sym.group,
                    "path": sym.relative_path,
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "signature": sym.signature,
                    "context": sym.context,
                }
            )
            if len(results) >= k:
                break
        return results

    # --- purge ----------------------------------------------------------

    def purge(self, keep_rules: bool = True) -> Dict[str, int]:
        """
        Delete code index artifacts (FAISS, symbols, hashes, watch state).

        Keeps path rules by default so ownership config survives a rebuild.
        """
        removed = 0
        self.code_folder.mkdir(parents=True, exist_ok=True)
        keep_names = {"code_rules.json"} if keep_rules else set()
        for path in list(self.code_folder.iterdir()):
            if path.name in keep_names:
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed += 1
            except OSError:
                continue
        return {"removed": removed, "kept_rules": int(keep_rules and (self.code_folder / "code_rules.json").exists())}

    # --- watch control --------------------------------------------------

    def _write_watch_status(self, **fields) -> None:
        self.code_folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "updated_at": time.time(),
            **fields,
        }
        with open(self.watch_status_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _clear_watch_status(self) -> None:
        if self.watch_status_path.exists():
            try:
                self.watch_status_path.unlink()
            except OSError:
                pass

    def _read_watch_status(self) -> Optional[Dict]:
        if not self.watch_status_path.exists():
            return None
        try:
            with open(self.watch_status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _consume_watch_cmd(self) -> Optional[Dict]:
        if not self.watch_cmd_path.exists():
            return None
        try:
            with open(self.watch_cmd_path, "r", encoding="utf-8") as f:
                cmd = json.load(f)
        except (OSError, json.JSONDecodeError):
            cmd = None
        try:
            self.watch_cmd_path.unlink()
        except OSError:
            pass
        return cmd if isinstance(cmd, dict) else None

    def _send_watch_cmd(self, action: str, **extra) -> Dict:
        self.code_folder.mkdir(parents=True, exist_ok=True)
        payload = {"action": action, "sent_at": time.time(), **extra}
        with open(self.watch_cmd_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    def is_watch_running(self) -> bool:
        status = self._read_watch_status()
        if not status:
            return False
        pid = status.get("pid")
        if not isinstance(pid, int):
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def stop_watch(self, minutes: Optional[float] = None) -> Dict:
        """
        Signal a running watcher.

        - minutes is None: stop permanently
        - minutes > 0: pause for that many minutes, then resume automatically
        """
        if not self.is_watch_running():
            return {"ok": False, "reason": "no_running_watch"}
        if minutes is None:
            cmd = self._send_watch_cmd("stop")
            return {"ok": True, "action": "stop", "cmd": cmd}
        if minutes <= 0:
            raise ValueError("minutes must be > 0 for a timed pause")
        cmd = self._send_watch_cmd("pause", minutes=float(minutes))
        return {"ok": True, "action": "pause", "minutes": float(minutes), "cmd": cmd}

    def resume_watch(self) -> Dict:
        if not self.is_watch_running():
            return {"ok": False, "reason": "no_running_watch"}
        cmd = self._send_watch_cmd("resume")
        return {"ok": True, "action": "resume", "cmd": cmd}

    def watch(
        self,
        poll_seconds: float = 1.0,
        stop_after: Optional[float] = None,
        stop_after_minutes: Optional[float] = None,
    ) -> None:
        """
        Poll for file writes and refresh symbol line maps.

        Control from another process:
          awesomeSearch code watch stop
          awesomeSearch code watch stop --minutes 10   # pause during npm update
          awesomeSearch code watch resume
        """
        if stop_after_minutes is not None:
            stop_after = float(stop_after_minutes) * 60.0

        print(f"Watching {self.project_root} for code changes (poll={poll_seconds}s)...")
        print("Stop: awesomeSearch code watch stop")
        print("Pause N min: awesomeSearch code watch stop --minutes N")
        print("Or Ctrl+C.")
        known = self._load_hashes()
        started = time.time()
        pause_until: Optional[float] = None
        self._write_watch_status(state="running", pause_until=None)
        try:
            while True:
                now = time.time()
                if stop_after is not None and (now - started) >= stop_after:
                    print("Reached --for / stop-after limit; exiting watch.")
                    break

                cmd = self._consume_watch_cmd()
                if cmd:
                    action = (cmd.get("action") or "").lower()
                    if action == "stop":
                        print("Received stop command; exiting watch.")
                        break
                    if action == "pause":
                        minutes = float(cmd.get("minutes") or 0)
                        pause_until = time.time() + max(0.0, minutes) * 60.0
                        print(f"Paused for {minutes:g} minute(s); will resume automatically.")
                        self._write_watch_status(state="paused", pause_until=pause_until)
                    elif action == "resume":
                        pause_until = None
                        print("Resumed watching.")
                        self._write_watch_status(state="running", pause_until=None)

                if pause_until is not None:
                    if time.time() < pause_until:
                        remaining = pause_until - time.time()
                        self._write_watch_status(state="paused", pause_until=pause_until)
                        time.sleep(min(poll_seconds, max(0.2, remaining)))
                        continue
                    print("Pause ended; resuming watch.")
                    pause_until = None
                    # Re-baseline hashes so a burst of npm/vendor writes during
                    # the pause does not flood refresh on resume.
                    known = {}
                    for path in self._iter_source_files():
                        rel = str(path.relative_to(self.project_root)).replace("\\", "/")
                        try:
                            known[rel] = self._file_hash(path)
                        except OSError:
                            continue
                    self._save_hashes(known)
                    self._write_watch_status(state="running", pause_until=None)

                current: Dict[str, str] = {}
                for path in self._iter_source_files():
                    rel = str(path.relative_to(self.project_root)).replace("\\", "/")
                    try:
                        digest = self._file_hash(path)
                    except OSError:
                        continue
                    current[rel] = digest
                    if known.get(rel) != digest:
                        print(f"change: {rel}")
                        n = self.refresh_path(path)
                        print(f"  refreshed {n} symbol(s)")
                # deletions
                for rel in list(known.keys()):
                    if rel not in current:
                        print(f"deleted: {rel}")
                        symbols = [s for s in self._load_symbols() if s.relative_path != rel]
                        hashes = self._load_hashes()
                        hashes.pop(rel, None)
                        if symbols:
                            embeddings = self._embed([s.search_blob() for s in symbols])
                            index, engine_meta = self._build_faiss(embeddings)
                            faiss.write_index(index, str(self.faiss_path))
                            self._save_symbols(symbols, engine_meta)
                        else:
                            self._save_symbols([])
                            if self.faiss_path.exists():
                                self.faiss_path.unlink()
                        self._save_hashes(hashes)
                known = current
                self._write_watch_status(state="running", pause_until=None)
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("\nStopped watching.")
        finally:
            self._clear_watch_status()
            if self.watch_cmd_path.exists():
                try:
                    self.watch_cmd_path.unlink()
                except OSError:
                    pass
