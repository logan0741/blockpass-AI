import subprocess
import os

def get_disk_usage(path):
    """지정된 경로가 속한 파일 시스템의 디스크 사용량을 가져옵니다."""
    try:
        # df 명령어-B1 옵션으로 바이트 단위의 파일 시스템 통계를 가져옵니다.
        df_proc = subprocess.run(['df', '-B1', path], capture_output=True, text=True, check=True, encoding='utf-8')
        # 출력 결과 파싱 (헤더 라인 제외)
        line = df_proc.stdout.strip().split('\n')[1]
        parts = line.split()
        
        total = int(parts[1])
        used = int(parts[2])
        available = int(parts[3])
        
        return total, used, available
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError) as e:
        print(f"디스크 사용량 확인 중 오류 발생: {e}")
        return None, None, None

def get_dir_size(dir_path):
    """지정된 디렉토리의 크기를 바이트 단위로 가져옵니다."""
    try:
        # 'du -sb'를 사용하여 바이트 단위의 총 크기를 가져옵니다.
        du_proc = subprocess.run(['du', '-sb', dir_path], capture_output=True, text=True, check=True, encoding='utf-8')
        size_bytes = int(du_proc.stdout.split()[0])
        return size_bytes
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError) as e:
        print(f"디렉토리 크기 확인 중 오류 발생: {e}")
        return None

def bytes_to_human_readable(size_bytes):
    """바이트를 사람이 읽기 쉬운 형태 (KB, MB, GB 등)로 변환합니다."""
    if size_bytes is None:
        return "N/A"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size_bytes >= power and n < len(power_labels) -1 :
        size_bytes /= power
        n += 1
    return f"{size_bytes:.2f}{power_labels[n]}B"

def main():
    # 분석할 대상 폴더와 해당 폴더가 속한 파일 시스템의 경로를 지정합니다.
    target_dir = '/home/gunhee/파인튜닝 데이터'
    mount_path = '/home/gunhee'

    # 1. 파일 시스템의 전체, 사용된, 남은 공간을 가져옵니다.
    total_space_bytes, used_space_bytes, available_space_bytes = get_disk_usage(mount_path)

    if total_space_bytes is None:
        return

    # 2. 대상 디렉토리의 크기를 가져옵니다.
    dir_size_bytes = get_dir_size(target_dir)

    dir_proportion = 0
    if dir_size_bytes is not None:
        # 3. 사용된 공간 대비 대상 디렉토리의 비중을 계산합니다.
        if used_space_bytes > 0:
            dir_proportion = (dir_size_bytes / used_space_bytes) * 100
    else:
        print(f"'{target_dir}'의 비중을 계산할 수 없습니다.")
            
    # 전체 공간 대비 사용된 공간의 비중을 계산합니다.
    total_usage_proportion = 0
    if total_space_bytes > 0:
        total_usage_proportion = (used_space_bytes / total_space_bytes) * 100

    # 4. 결과를 출력합니다.
    print(f"📊 디스크 용량 분석 (기준: {mount_path})")
    print("-------------------------------------------------")
    print(f"  - 전체 공간: {bytes_to_human_readable(total_space_bytes)}")
    print(f"  - 사용된 공간: {bytes_to_human_readable(used_space_bytes)} ({total_usage_proportion:.2f}%)")
    print(f"  - 남은 공간: {bytes_to_human_readable(available_space_bytes)}")
    print("-------------------------------------------------")
    if dir_size_bytes is not None:
        print(f"📂 '{os.path.basename(target_dir)}' 폴더 정보")
        print(f"  - 폴더 크기: {bytes_to_human_readable(dir_size_bytes)}")
        print(f"  - 사용된 공간 내 비중: {dir_proportion:.2f}%")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()
