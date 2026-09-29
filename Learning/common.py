"""오프라인 학습의 공통 경로와 재현성 식별자를 제공한다."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL = REPO / "models/xs/model.xml"
ARTIFACTS = ROOT / "artifacts"
JOINTS = tuple(f"J{i}" for i in range(1, 8)) + ("G_R",)
BODIES = ("shoulder_L", "upper_L", "forearm_L", "forearm_R", "upper_R",
          "shoulder_R", "gripper_R", "gripper_R_thumb")
GEOMS = ("gripper_R_geom_0", "gripper_R_thumb_geom_0", "ibeam_mesh")


def source_mesh_path(name):
    """생성 당시 절대 경로 대신 현재 XML에 연결된 원본 메시 경로를 반환한다."""
    xml = ET.parse(MODEL).getroot()
    geom = xml.find(f".//geom[@name='{name}']")
    asset = xml.find(f"./asset/mesh[@name='{geom.attrib['mesh']}']")
    return MODEL.parent / asset.attrib["file"]


def file_hash(path):
    """파일 내용의 SHA-256 식별자를 계산한다."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    """설정의 키 순서에 무관한 SHA-256 식별자를 계산한다."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    """유한한 값만 허용하는 JSON을 UTF-8로 저장한다."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
