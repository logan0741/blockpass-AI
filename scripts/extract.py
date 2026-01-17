#!/usr/bin/env python3
"""
압축 파일 해제 스크립트

사용법:
    python scripts/extract.py /경로/파일.zip
    python scripts/extract.py /경로/파일.zip --output /출력경로/
    python scripts/extract.py /경로/파일.tar.gz

지원 형식: zip, tar, tar.gz, tgz, tar.bz2, 7z, rar
"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract_file(archive_path: str, output_dir: str = None):
    """압축 파일 해제"""
    archive = Path(archive_path)

    if not archive.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {archive_path}")
        sys.exit(1)

    # 출력 디렉토리 설정
    if output_dir:
        output = Path(output_dir)
    else:
        output = archive.parent

    output.mkdir(parents=True, exist_ok=True)

    suffix = "".join(archive.suffixes).lower()
    name = archive.name

    print(f"📦 압축 해제 중: {name}")
    print(f"📂 출력 경로: {output}")

    try:
        # ZIP
        if name.endswith('.zip'):
            subprocess.run([
                'unzip', '-o', str(archive), '-d', str(output)
            ], check=True)

        # TAR.GZ / TGZ
        elif name.endswith('.tar.gz') or name.endswith('.tgz'):
            subprocess.run([
                'tar', '-xzf', str(archive), '-C', str(output)
            ], check=True)

        # TAR.BZ2
        elif name.endswith('.tar.bz2'):
            subprocess.run([
                'tar', '-xjf', str(archive), '-C', str(output)
            ], check=True)

        # TAR
        elif name.endswith('.tar'):
            subprocess.run([
                'tar', '-xf', str(archive), '-C', str(output)
            ], check=True)

        # 7Z
        elif name.endswith('.7z'):
            subprocess.run([
                '7z', 'x', str(archive), f'-o{output}', '-y'
            ], check=True)

        # RAR
        elif name.endswith('.rar'):
            subprocess.run([
                'unrar', 'x', '-o+', str(archive), str(output) + '/'
            ], check=True)

        else:
            print(f"❌ 지원하지 않는 형식: {suffix}")
            print("지원 형식: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .7z, .rar")
            sys.exit(1)

        print(f"✅ 압축 해제 완료!")

        # 결과 확인
        files = list(output.rglob('*'))
        file_count = len([f for f in files if f.is_file()])
        dir_count = len([f for f in files if f.is_dir()])
        print(f"📊 파일: {file_count}개, 폴더: {dir_count}개")

    except FileNotFoundError as e:
        print(f"❌ 필요한 프로그램이 없습니다. 설치해주세요:")
        if 'unzip' in str(e):
            print("   sudo apt install unzip")
        elif '7z' in str(e):
            print("   sudo apt install p7zip-full")
        elif 'unrar' in str(e):
            print("   sudo apt install unrar")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ 압축 해제 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="압축 파일 해제")
    parser.add_argument("file", help="압축 파일 경로")
    parser.add_argument("-o", "--output", help="출력 경로 (기본: 압축파일 위치)")

    args = parser.parse_args()
    extract_file(args.file, args.output)
