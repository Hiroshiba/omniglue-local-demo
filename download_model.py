#!/usr/bin/env python3
"""OmniGlue用のモデルファイルをダウンロードするスクリプト"""

import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def main():
    """必要なモデルファイルをダウンロード"""
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    download_superpoint(models_dir)
    download_dinov2(models_dir)
    download_omniglue(models_dir)
    
    print("全てのモデルファイルのダウンロードが完了しました！")


def download_superpoint(models_dir: Path):
    """SuperPoint モデルをダウンロード"""
    sp_dir = models_dir / "sp_v6"
    if sp_dir.exists():
        print("SuperPoint モデルは既にダウンロード済みです")
        return
        
    print("SuperPoint モデルをダウンロード中...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        subprocess.run([
            "git", "clone", "https://github.com/rpautrat/SuperPoint.git", 
            str(tmpdir / "SuperPoint")
        ], check=True)
        
        tgz_src = tmpdir / "SuperPoint" / "pretrained_models" / "sp_v6.tgz"
        tgz_dst = models_dir / "sp_v6.tgz"
        shutil.move(str(tgz_src), str(tgz_dst))
        
        with tarfile.open(tgz_dst, 'r:gz') as tar:
            tar.extractall(models_dir)
        
        tgz_dst.unlink()
    
    print("SuperPoint モデルのダウンロード完了")


def download_dinov2(models_dir: Path):
    """DINOv2 モデルをダウンロード"""
    dino_path = models_dir / "dinov2_vitb14_pretrain.pth"
    if dino_path.exists():
        print("DINOv2 モデルは既にダウンロード済みです")
        return
        
    print("DINOv2 モデルをダウンロード中...")
    
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    urllib.request.urlretrieve(url, dino_path)
    
    print("DINOv2 モデルのダウンロード完了")


def download_omniglue(models_dir: Path):
    """OmniGlue モデルをダウンロード"""
    og_dir = models_dir / "og_export"
    if og_dir.exists():
        print("OmniGlue モデルは既にダウンロード済みです")
        return
        
    print("OmniGlue モデルをダウンロード中...")
    
    zip_path = models_dir / "og_export.zip"
    url = "https://storage.googleapis.com/omniglue/og_export.zip"
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(models_dir)
    
    zip_path.unlink()
    
    print("OmniGlue モデルのダウンロード完了")


if __name__ == "__main__":
    main()