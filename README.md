# SigLIP-2 图文双向检索 App

这是一个基于 **SigLIP-2** 的图文双向检索系统，适合作为课程设计、项目展示或多模态检索 Demo。

系统支持两类检索：

- **Text-to-Image Retrieval**：输入文字，从本地图片库中检索最相关图片。
- **Image-to-Text Label Retrieval**：上传图片，从预设标签库中检索最相关文字标签。

后端使用 **Python + FastAPI + Hugging Face Transformers + FAISS**，模型为 `google/siglip2-base-patch16-224`。前端使用 **React + Vite + axios**。

## 项目结构

```text
siglip2-image-search-app/
├── backend/
│   ├── main.py
│   ├── encoder.py
│   ├── index_manager.py
│   ├── requirements.txt
│   ├── images/
│   ├── labels.txt
│   ├── image_index.faiss
│   ├── image_paths.json
│   ├── image_embeddings.npy
│   ├── label_index.faiss
│   ├── labels.json
│   └── label_embeddings.npy
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   └── style.css
└── README.md
```

说明：`image_index.faiss`、`image_paths.json`、`image_embeddings.npy`、`label_index.faiss`、`labels.json`、`label_embeddings.npy` 是运行时自动生成的索引和特征文件。第一次启动后端或调用 `/rebuild_index` 后会生成。

## 环境安装

### 后端

```bash
cd backend
pip install -r requirements.txt
```

如果你有 NVIDIA GPU，请根据本机 CUDA 版本优先从 PyTorch 官网安装对应 GPU 版 `torch`，然后再安装其他依赖。代码会自动选择：

- 有 GPU：`cuda`
- 无 GPU：`cpu`

### 前端

```bash
cd frontend
npm install
```

## 数据准备

把本地图片放到：

```text
backend/images/
```

支持格式：

- `jpg`
- `jpeg`
- `png`
- `webp`

把候选文字标签写到：

```text
backend/labels.txt
```

每行一个标签，例如：

```text
cat
dog
car
sunset
beach
mountain
一只猫
一条狗
海边
雪山
城市街道
```

### 可选：从 Hugging Face 自动导入 COCO 小样本

项目提供了 `backend/prepare_coco.py`，可以直接从 Hugging Face datasets 下载/流式读取 COCO 小样本，自动保存图片并生成标签。

默认使用 `detection-datasets/coco` 的 `val` split，适合课程展示：

```bash
cd backend
python prepare_coco.py --max-images 100
```

也可以调大样本数量：

```bash
python prepare_coco.py --split train --max-images 1000
```

如果想让 80 个 COCO 类别更均衡，例如每类 100 张，推荐使用 `train` split：

```bash
python prepare_coco.py --split train --balanced --per-category 100
```

这会尽量准备约 `80 × 100 = 8000` 张图片，并按类别子目录保存到 `backend/images/`，例如：

```text
backend/images/person/
backend/images/bicycle/
backend/images/car/
...
```

如果只是先测试流程，可以限制扫描数量：

```bash
python prepare_coco.py --split train --balanced --per-category 10 --max-scan 2000
```

脚本会：

- 把图片保存到 `backend/images/`
- 从 COCO 的 object category 中提取标签并追加写入 `backend/labels.txt`
- 保留已有标签并自动去重

导入完成后，启动后端或调用 `POST /rebuild_index` 重新构建 FAISS 索引。

## 启动后端

```bash
cd backend
python main.py
```

或者：

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Windows 一键启动

项目根目录提供了启动脚本：

```text
start_app.bat
start_app.ps1
```

推荐直接双击 `start_app.bat`，它会：

- 启动后端窗口
- 启动前端窗口
- 自动打开 `http://localhost:5173/`

脚本默认使用 conda 环境名：

```text
siglip2
```

如果你的环境名不同，请编辑 `start_app.bat` 里的 `ENV_NAME`。

第一次启动时，后端会自动加载 SigLIP-2 模型，并在索引不存在时构建：

- `backend/image_index.faiss`
- `backend/image_paths.json`
- `backend/image_embeddings.npy`
- `backend/label_index.faiss`
- `backend/labels.json`
- `backend/label_embeddings.npy`

构建索引时，后端终端会显示进度条，例如 `Building image index`。图片索引使用 batch 推理，默认每批 32 张，更适合 GPU。

图片特征和标签特征会额外保存为 `.npy` 文件。后续启动时会直接加载已有 FAISS 索引和元数据，不会重复编码整个图片库；只有输入查询文本或上传图片需要过一次 SigLIP-2 编码器。

## 启动前端

```bash
cd frontend
npm run dev
```

默认前端会请求：

```text
http://localhost:8000
```

如需修改后端地址，可以在启动前端时设置环境变量：

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Windows PowerShell：

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

## 使用方式

1. 打开前端页面。
2. 在 **Text to Image Search** 区域输入文字，例如：
   - `a dog running on the grass`
   - `一只猫坐在沙发上`
   - `海边日落风景`
3. 点击 **Search Images**，系统会返回最相关的图片、文件名和相似度分数。
4. 在 **Image to Label Search** 区域上传图片。
5. 点击 **Search Labels**，系统会返回最相关的文字标签和相似度分数。
6. 修改 `backend/images/` 或 `backend/labels.txt` 后，可以点击页面右上角 **Rebuild Index** 或调用接口 `POST /rebuild_index` 重新构建索引。

## API 接口

### GET /health

查看服务、设备和索引状态。

### GET /index_status

查看当前索引构建进度。前端点击 **Rebuild Index** 时会自动轮询该接口并显示进度条。

### POST /search_text

请求：

```json
{
  "query": "一只猫坐在沙发上",
  "top_k": 8
}
```

返回：

```json
{
  "results": [
    {
      "filename": "cat_001.jpg",
      "url": "http://localhost:8000/images/cat_001.jpg",
      "score": 0.873
    }
  ]
}
```

### POST /search_image

请求格式：`multipart/form-data`

字段：

- `file`：用户上传的图片
- `top_k`：返回标签数量，默认 `5`

返回：

```json
{
  "results": [
    {
      "label": "cat",
      "score": 0.912
    },
    {
      "label": "一只猫",
      "score": 0.887
    }
  ]
}
```

### GET /images/{filename}

用于前端访问本地图片库中的图片。

### POST /rebuild_index

重新读取 `backend/images/` 和 `backend/labels.txt`，并重新构建图片索引和标签索引。

## 技术要点

- 使用 `from transformers import AutoProcessor, AutoModel` 加载 SigLIP-2。
- 使用 `model.get_text_features(**inputs)` 提取文本向量。
- 使用 `model.get_image_features(**inputs)` 提取图片向量。
- 图片向量和文本向量都做 L2 normalization。
- FAISS 使用 `IndexFlatIP`，在向量归一化后内积等价于 cosine similarity。
- 所有推理都在 `torch.no_grad()` 中执行。
- 模型启动后设置为 `eval()`。
