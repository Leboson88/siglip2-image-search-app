from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from encoder import SigLIP2Encoder
from index_manager import IndexManager


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"

app = FastAPI(title="SigLIP-2 Image Search App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder: SigLIP2Encoder | None = None
index_manager: IndexManager | None = None
startup_status: dict = {}


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索文本")
    top_k: int = Field(8, ge=1, le=50, description="返回图片数量")


def get_index_manager() -> IndexManager:
    if index_manager is None:
        raise HTTPException(status_code=503, detail="模型或索引尚未初始化完成。")
    return index_manager


@app.on_event("startup")
def startup_event() -> None:
    global encoder, index_manager, startup_status
    encoder = SigLIP2Encoder()
    index_manager = IndexManager(BASE_DIR, encoder)
    startup_status = index_manager.ensure_indexes()


@app.get("/health")
def health() -> dict:
    manager = get_index_manager()
    return {
        "status": "ok",
        "device": str(manager.encoder.device),
        "image_count": len(manager.image_paths),
        "label_count": len(manager.labels),
        "startup": startup_status,
    }


@app.get("/index_status")
def index_status() -> dict:
    manager = get_index_manager()
    return manager.get_progress()


@app.post("/search_text")
def search_text(payload: TextSearchRequest) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空。")

    manager = get_index_manager()
    results = manager.search_images_by_text(
        query=query,
        top_k=payload.top_k,
        base_url="http://localhost:8000",
    )
    return {"results": results}


@app.post("/search_image")
def search_image(file: UploadFile = File(...), top_k: int = Form(5)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="请上传一张图片。")

    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传文件必须是图片。")

    manager = get_index_manager()
    try:
        results = manager.search_labels_by_image(file=file, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"图片解析或检索失败：{exc}") from exc
    return {"results": results}


@app.post("/rebuild_index")
def rebuild_index() -> dict:
    manager = get_index_manager()
    status = manager.rebuild_all_indexes()
    return {"message": "索引已重新构建。", "status": status}


@app.get("/images/{filename:path}")
def get_image(filename: str):
    if not filename:
        raise HTTPException(status_code=404, detail="图片不存在。")

    image_path = (IMAGES_DIR / filename).resolve()
    images_root = IMAGES_DIR.resolve()
    if images_root not in image_path.parents and image_path != images_root:
        raise HTTPException(status_code=403, detail="非法图片路径。")

    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="图片不存在。")

    return FileResponse(image_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
