from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SigLIP2Encoder:
    """SigLIP-2 图文编码器，统一负责模型加载与特征归一化。"""

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            features = self._extract_feature_tensor(self.model.get_text_features(**inputs))
            features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            features = self._extract_feature_tensor(self.model.get_image_features(**inputs))
            features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32)

    def encode_image_path(self, image_path: str | Path) -> np.ndarray:
        with Image.open(image_path) as image:
            return self.encode_images([image])

    def encode_uploaded_image(self, file) -> np.ndarray:
        """兼容 FastAPI UploadFile、普通二进制文件对象和 bytes。"""
        image_bytes = self._read_uploaded_bytes(file)
        with Image.open(BytesIO(image_bytes)) as image:
            return self.encode_images([image])

    @staticmethod
    def _read_uploaded_bytes(file) -> bytes:
        if isinstance(file, bytes):
            return file

        if hasattr(file, "file"):
            stream: BinaryIO = file.file
            stream.seek(0)
            data = stream.read()
            stream.seek(0)
            return data

        if hasattr(file, "read"):
            data = file.read()
            if hasattr(file, "seek"):
                file.seek(0)
            return data

        raise ValueError("无法读取上传的图片文件。")

    @staticmethod
    def _extract_feature_tensor(model_output) -> torch.Tensor:
        """兼容不同 transformers 版本的 SigLIP/SigLIP-2 输出格式。"""
        if isinstance(model_output, torch.Tensor):
            return model_output

        if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
            return model_output.pooler_output

        if hasattr(model_output, "last_hidden_state") and model_output.last_hidden_state is not None:
            return model_output.last_hidden_state.mean(dim=1)

        if isinstance(model_output, (tuple, list)) and model_output:
            first_item = model_output[0]
            if isinstance(first_item, torch.Tensor):
                return first_item

        raise TypeError(f"无法从模型输出中提取特征向量：{type(model_output)!r}")
