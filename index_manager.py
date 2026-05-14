from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from encoder import SigLIP2Encoder


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 64


class IndexManager:
    """管理图片索引、标签索引，以及双向检索逻辑。"""

    def __init__(self, base_dir: str | Path, encoder: SigLIP2Encoder):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.labels_file = self.base_dir / "labels.txt"

        self.image_index_path = self.base_dir / "image_index.faiss"
        self.image_paths_path = self.base_dir / "image_paths.json"
        self.image_embeddings_path = self.base_dir / "image_embeddings.npy"
        self.label_index_path = self.base_dir / "label_index.faiss"
        self.labels_json_path = self.base_dir / "labels.json"
        self.label_embeddings_path = self.base_dir / "label_embeddings.npy"

        self.encoder = encoder
        self.image_index: faiss.IndexFlatIP | None = None
        self.image_paths: list[str] = []
        self.label_index: faiss.IndexFlatIP | None = None
        self.labels: list[str] = []
        self.progress: dict = {
            "running": False,
            "stage": "idle",
            "done": 0,
            "total": 0,
            "message": "索引空闲。",
        }

        self.images_dir.mkdir(parents=True, exist_ok=True)

    def ensure_indexes(self) -> dict:
        image_status = self.load_image_index()
        label_status = self.load_label_index()
        return {"image_index": image_status, "label_index": label_status}

    def build_image_index(self) -> dict:
        image_files = self._list_image_files()
        valid_paths: list[str] = []
        embeddings: list[np.ndarray] = []
        skipped: list[str] = []
        total = len(image_files)

        self._set_progress("image_index", 0, total, "正在构建图片索引。", running=True)

        for start in tqdm(range(0, total, IMAGE_BATCH_SIZE), desc="Building image index", unit="batch"):
            batch_paths = image_files[start : start + IMAGE_BATCH_SIZE]
            batch_images: list[Image.Image] = []
            batch_valid_paths: list[Path] = []

            for image_path in batch_paths:
                try:
                    with Image.open(image_path) as image:
                        batch_images.append(image.convert("RGB").copy())
                    batch_valid_paths.append(image_path)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    skipped.append(f"{image_path.name}: {exc}")

            if batch_images:
                try:
                    batch_vectors = self.encoder.encode_images(batch_images)
                    embeddings.extend(batch_vectors)
                    valid_paths.extend(
                        path.relative_to(self.images_dir).as_posix()
                        for path in batch_valid_paths
                    )
                except Exception as exc:
                    for image_path in batch_valid_paths:
                        skipped.append(f"{image_path.name}: {exc}")

            done = min(start + IMAGE_BATCH_SIZE, total)
            self._set_progress(
                "image_index",
                done,
                total,
                f"正在构建图片索引：{done}/{total}",
                running=True,
            )

        self.image_paths = valid_paths

        if embeddings:
            matrix = np.vstack(embeddings).astype(np.float32)
            np.save(self.image_embeddings_path, matrix)
            self.image_index = faiss.IndexFlatIP(matrix.shape[1])
            self.image_index.add(matrix)
            faiss.write_index(self.image_index, str(self.image_index_path))
        else:
            self.image_index = None
            self._delete_file(self.image_index_path)
            self._delete_file(self.image_embeddings_path)

        self._write_json(self.image_paths_path, self.image_paths)
        self._set_progress("image_index", total, total, "图片索引构建完成。", running=False)
        return {
            "status": "built",
            "count": len(self.image_paths),
            "skipped": skipped,
            "index_file": str(self.image_index_path),
        }

    def load_image_index(self) -> dict:
        if not self.image_index_path.exists() or not self.image_paths_path.exists():
            return self.build_image_index()

        try:
            self.image_index = faiss.read_index(str(self.image_index_path))
            self.image_paths = self._read_json_list(self.image_paths_path)
            if self.image_index.ntotal != len(self.image_paths):
                return self.build_image_index()
            embeddings_status = self._ensure_embeddings_file(
                self.image_index,
                self.image_embeddings_path,
            )
            return {
                "status": "loaded",
                "count": len(self.image_paths),
                "embeddings": embeddings_status,
            }
        except Exception:
            return self.build_image_index()

    def build_label_index(self) -> dict:
        labels = self._read_labels_txt()
        self.labels = labels
        total = len(labels)
        self._set_progress("label_index", 0, total, "正在构建标签索引。", running=True)

        if labels:
            vectors = []
            for start in tqdm(range(0, total, TEXT_BATCH_SIZE), desc="Building label index", unit="batch"):
                batch = labels[start : start + TEXT_BATCH_SIZE]
                vectors.append(self.encoder.encode_text(batch))
                done = min(start + TEXT_BATCH_SIZE, total)
                self._set_progress(
                    "label_index",
                    done,
                    total,
                    f"正在构建标签索引：{done}/{total}",
                    running=True,
                )

            matrix = np.vstack(vectors).astype(np.float32)
            np.save(self.label_embeddings_path, matrix)
            self.label_index = faiss.IndexFlatIP(matrix.shape[1])
            self.label_index.add(matrix)
            faiss.write_index(self.label_index, str(self.label_index_path))
        else:
            self.label_index = None
            self._delete_file(self.label_index_path)
            self._delete_file(self.label_embeddings_path)

        self._write_json(self.labels_json_path, self.labels)
        self._set_progress("label_index", total, total, "标签索引构建完成。", running=False)
        return {
            "status": "built",
            "count": len(self.labels),
            "index_file": str(self.label_index_path),
        }

    def load_label_index(self) -> dict:
        if not self.label_index_path.exists() or not self.labels_json_path.exists():
            return self.build_label_index()

        try:
            self.label_index = faiss.read_index(str(self.label_index_path))
            self.labels = self._read_json_list(self.labels_json_path)
            if self.label_index.ntotal != len(self.labels):
                return self.build_label_index()
            embeddings_status = self._ensure_embeddings_file(
                self.label_index,
                self.label_embeddings_path,
            )
            return {
                "status": "loaded",
                "count": len(self.labels),
                "embeddings": embeddings_status,
            }
        except Exception:
            return self.build_label_index()

    def search_images_by_text(
        self,
        query: str,
        top_k: int = 8,
        base_url: str = "http://localhost:8000",
    ) -> list[dict]:
        if self.image_index is None or not self.image_paths:
            return []

        top_k = self._normalize_top_k(top_k, default=8, maximum=len(self.image_paths))
        query_vector = self.encoder.encode_text([query])
        scores, indices = self.image_index.search(query_vector, top_k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            filename = self.image_paths[int(index)]
            image_url = quote(filename, safe="/")
            results.append(
                {
                    "filename": filename,
                    "url": f"{base_url.rstrip('/')}/images/{image_url}",
                    "score": round(float(score), 6),
                }
            )
        return results

    def search_labels_by_image(self, file, top_k: int = 5) -> list[dict]:
        if self.label_index is None or not self.labels:
            return []

        top_k = self._normalize_top_k(top_k, default=5, maximum=len(self.labels))
        image_vector = self.encoder.encode_uploaded_image(file)
        scores, indices = self.label_index.search(image_vector, top_k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(
                {
                    "label": self.labels[int(index)],
                    "score": round(float(score), 6),
                }
            )
        return results

    def rebuild_all_indexes(self) -> dict:
        self._set_progress("rebuild", 0, 2, "正在重建全部索引。", running=True)
        image_status = self.build_image_index()
        self._set_progress("rebuild", 1, 2, "图片索引已完成，正在构建标签索引。", running=True)
        label_status = self.build_label_index()
        self._set_progress("rebuild", 2, 2, "全部索引重建完成。", running=False)
        return {"image_index": image_status, "label_index": label_status}

    def get_progress(self) -> dict:
        progress = dict(self.progress)
        total = progress.get("total") or 0
        done = progress.get("done") or 0
        progress["percent"] = round(done / total * 100, 2) if total else 0
        progress["image_count"] = len(self.image_paths)
        progress["label_count"] = len(self.labels)
        return progress

    def _list_image_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _read_labels_txt(self) -> list[str]:
        if not self.labels_file.exists():
            self.labels_file.write_text("", encoding="utf-8")
            return []

        labels = []
        seen = set()
        for line in self.labels_file.read_text(encoding="utf-8").splitlines():
            label = line.strip()
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
        return labels

    @staticmethod
    def _normalize_top_k(top_k: int, default: int, maximum: int) -> int:
        try:
            value = int(top_k)
        except (TypeError, ValueError):
            value = default
        value = max(1, value)
        return min(value, max(1, maximum))

    @staticmethod
    def _write_json(path: Path, data: list[str]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json_list(path: Path) -> list[str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError(f"{path.name} 格式错误。")
        return data

    @staticmethod
    def _delete_file(path: Path) -> None:
        if path.exists():
            path.unlink()

    def _set_progress(self, stage: str, done: int, total: int, message: str, running: bool) -> None:
        self.progress = {
            "running": running,
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "message": message,
        }

    @staticmethod
    def _ensure_embeddings_file(index: faiss.Index, path: Path) -> str:
        if path.exists():
            return "loaded"

        if index.ntotal == 0:
            np.save(path, np.empty((0, 0), dtype=np.float32))
            return "created_empty"

        if not hasattr(index, "reconstruct_n"):
            return "missing"

        matrix = index.reconstruct_n(0, index.ntotal).astype(np.float32)
        np.save(path, matrix)
        return "reconstructed_from_faiss"
