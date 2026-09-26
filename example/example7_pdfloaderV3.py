"""
Example 7 V3: MinerU API 本地文件批量上传解析
功能：批量上传本地文件进行解析
"""

import os
import time
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

# ==================== 配置 ====================

# 优先读 .env 里的 MINERU_API_KEY，没有才用这里的默认值
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "sk-ecSxGAYv6k3Z3ojwRTmkt8EsENqYnRN8PTZ0bYSKKO8S2UZg")
MINERU_BATCH_URL = "https://mineru.net/api/v4/file-urls/batch"
MINERU_BATCH_RESULT_URL = "https://mineru.net/api/v4/extract-results/batch"
MODEL_VERSION = "vlm"

# ---------- 网络重试与超时 ----------
# 注意：mineru.net 的接口本身是通的，但“上传/下载文件”走的是阿里云 OSS
# （mineru.oss-cn-shanghai.aliyuncs.com）。该域名在部分网络下连接很不稳定：
# 实测同一个 IP 连续 3 次连接会出现 1 次 8s 超时，偶尔一次要等 7s 才连上。
# 原代码既没超时也没重试，任何一次抖动都会让整批任务失败。


def build_session() -> requests.Session:
    """带自动重试的 requests.Session：连接失败/超时/5xx 会退避重试。"""
    retry = Retry(
        total=5,
        connect=5,
        read=3,
        status=3,
        backoff_factor=1.5,  # 依次等待 1.5s、3s、6s、12s、24s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "PUT", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = build_session()

TIMEOUT_API = (10, 60)       # 普通接口：连接 10s / 读 60s
TIMEOUT_UPLOAD = (15, 600)   # 上传文件：放宽
TIMEOUT_DOWNLOAD = (15, 300)  # 下载结果 zip
UPLOAD_RETRIES = 3           # 每个文件最多尝试次数（在 session 重试之外再兜一层）


# ==================== 1. 批量申请上传链接 ====================

def apply_upload_urls(
    files: List[Dict[str, str]],
    model_version: str = MODEL_VERSION,
) -> Dict:
    """
    批量申请文件上传链接

    Args:
        files: 文件列表 [{"name": "demo.pdf", "data_id": "abcd"}, ...]
        model_version: 模型版本

    Returns:
        {"batch_id": "xxx", "file_urls": ["url1", "url2", ...]}
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_KEY}",
    }
    data = {
        "files": files,
        "model_version": model_version,
    }

    response = SESSION.post(MINERU_BATCH_URL, headers=headers, json=data, timeout=TIMEOUT_API)
    response.raise_for_status()

    result = response.json()
    if result.get("code") != 0:
        raise Exception(f"申请上传链接失败: {result.get('msg')}")

    return {
        "batch_id": result["data"]["batch_id"],
        "file_urls": result["data"]["file_urls"],
    }


# ==================== 2. 上传文件 ====================

def upload_file(file_path: str, upload_url: str) -> bool:
    """上传单个文件到指定 URL（上传链接是 OSS 预签名地址，不会返回错误体）"""
    with open(file_path, "rb") as f:
        response = SESSION.put(upload_url, data=f, timeout=TIMEOUT_UPLOAD)
    return response.status_code == 200


def upload_files(
    file_paths: List[str],
    file_urls: List[str],
) -> Dict[str, bool]:
    """
    批量上传文件（单个文件失败会重试，不会中断整批）

    Returns:
        {"file1.pdf": True, "file2.pdf": False, ...}
    """
    results = {}
    for file_path, url in zip(file_paths, file_urls):
        file_name = Path(file_path).name
        success = False

        for attempt in range(1, UPLOAD_RETRIES + 1):
            try:
                success = upload_file(file_path, url)
            except requests.exceptions.RequestException as e:
                print(f"  上传 {file_name} 第 {attempt} 次异常: {type(e).__name__}")
                success = False

            if success:
                break

            if attempt < UPLOAD_RETRIES:
                wait = 2 * attempt
                print(f"  上传 {file_name} 第 {attempt} 次失败，{wait}s 后重试...")
                time.sleep(wait)

        results[file_name] = success
        if success:
            print(f"  上传 {file_name}: 成功")
        else:
            print(f"  上传 {file_name}: 失败（已重试 {UPLOAD_RETRIES} 次）")

    return results


# ==================== 3. 批量查询任务结果 ====================

def get_batch_results(batch_id: str) -> Dict:
    """查询批量任务结果"""
    url = f"{MINERU_BATCH_RESULT_URL}/{batch_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_KEY}",
    }

    response = SESSION.get(url, headers=headers, timeout=TIMEOUT_API)
    response.raise_for_status()

    return response.json()


def wait_batch_completion(
    batch_id: str,
    poll_interval: int = 5,
    max_wait: int = 600,
) -> Dict:
    """等待批量任务完成（轮询期间的网络抖动不会直接中断）"""
    start_time = time.time()
    errors = 0

    while True:
        try:
            result = get_batch_results(batch_id)
            errors = 0
        except requests.exceptions.RequestException as e:
            errors += 1
            print(f"[等待] 查询失败（第 {errors} 次）: {type(e).__name__}")
            if errors >= 5:
                raise Exception(f"连续 {errors} 次查询失败，请检查网络: {e}") from e
            time.sleep(poll_interval)
            continue

        data = result.get("data", {})

        # API 返回: data.extract_result 是任务列表
        tasks = data.get("extract_result", [])

        if not tasks:
            print(f"[等待] 等待中...")
            time.sleep(poll_interval)
            continue

        # 统计状态
        total = len(tasks)
        done = sum(1 for t in tasks if t.get("state") == "done")
        failed = sum(1 for t in tasks if t.get("state") == "failed")
        running = sum(1 for t in tasks if t.get("state") == "running")
        pending = sum(1 for t in tasks if t.get("state") in ("pending", "waiting-file"))

        elapsed = time.time() - start_time
        print(f"[批量查询] 总: {total}, 完成: {done}, 失败: {failed}, 运行中: {running}, 等待: {pending}, 耗时: {elapsed:.0f}s")

        if done + failed == total:
            return result

        if elapsed > max_wait:
            raise Exception(f"等待超时（{max_wait}s）")

        time.sleep(poll_interval)


# ==================== 4. 下载并解压单个结果 ====================

def download_and_extract_single(zip_url: str, output_dir: str) -> str:
    """下载并解压单个结果，返回 markdown 内容"""
    print(f"  下载: {zip_url}")

    response = SESSION.get(zip_url, timeout=TIMEOUT_DOWNLOAD)
    response.raise_for_status()

    # 临时文件
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        zip_path = f.name

    try:
        with open(zip_path, "wb") as f:
            f.write(response.content)

        os.makedirs(output_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # 优先取 MinerU 生成的 full.md
    full_md = Path(output_dir) / "full.md"
    if full_md.exists():
        with open(full_md, "r", encoding="utf-8") as f:
            return f.read()

    md_files = sorted(Path(output_dir).glob("**/*.md"))
    if md_files:
        with open(md_files[0], "r", encoding="utf-8") as f:
            return f.read()

    return ""


# ==================== 5. 完整批量解析流程 ====================

def batch_parse_local_files(
    file_paths: List[str],
    model_version: str = MODEL_VERSION,
    output_dir: str = "./data/mineru_batch",
    wait_completion: bool = True,
) -> Dict:
    """
    批量解析本地文件

    Args:
        file_paths: 本地文件路径列表
        model_version: 模型版本
        output_dir: 输出目录
        wait_completion: 是否等待完成

    Returns:
        包含 batch_id、results 等信息的字典
    """
    print("=" * 50)
    print("MinerU 本地文件批量解析")
    print("=" * 50)
    print(f"文件数量: {len(file_paths)}")
    print(f"模型版本: {model_version}")

    # 1. 准备文件列表
    files = []
    for i, path in enumerate(file_paths):
        file_name = Path(path).name
        files.append({"name": file_name, "data_id": f"file_{i}"})

    # 2. 申请上传链接
    print("\n[1] 申请上传链接...")
    batch_info = apply_upload_urls(files, model_version)
    batch_id = batch_info["batch_id"]
    file_urls = batch_info["file_urls"]
    print(f"  batch_id: {batch_id}")
    print(f"  获取到 {len(file_urls)} 个上传链接")

    # 3. 上传文件
    print("\n[2] 上传文件...")
    upload_results = upload_files(file_paths, file_urls)
    print(f"  成功: {sum(upload_results.values())}/{len(upload_results)}")

    if not any(upload_results.values()):
        raise Exception("所有文件都上传失败，请检查网络后重试（上传链接有效期约 1 小时）")

    if not wait_completion:
        return {"batch_id": batch_id, "upload_results": upload_results}

    # 4. 等待解析完成
    print("\n[3] 等待解析完成...")
    batch_results = wait_batch_completion(batch_id)
    tasks = batch_results.get("data", {}).get("extract_result", [])

    # 5. 下载结果
    print("\n[4] 下载解析结果...")
    results = {}
    for task in tasks:
        data_id = task.get("data_id", "unknown")
        state = task.get("state")
        file_output_dir = os.path.join(output_dir, data_id)

        if state == "done":
            zip_url = task.get("full_zip_url", "")
            if zip_url:
                content = download_and_extract_single(zip_url, file_output_dir)
                results[data_id] = {
                    "state": state,
                    "content": content,
                    "output_dir": file_output_dir,
                }
            else:
                results[data_id] = {"state": state}
        else:
            results[data_id] = {"state": state, "err_msg": task.get("err_msg", "")}

    print("\n" + "=" * 50)
    print("批量解析完成！")
    print("=" * 50)

    return {
        "batch_id": batch_id,
        "results": results,
    }


# ==================== 6. 示例用法 ====================

if __name__ == "__main__":
    # 示例：解析 data 目录下的所有 PDF 文件
    # 用 __file__ 推导路径，这样在任意目录下运行都不会找不到文件
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_dir = PROJECT_ROOT / "data"
    file_paths = sorted(str(p) for p in data_dir.glob("*.pdf"))[:5]  # 最多 5 个文件

    if not file_paths:
        print("未找到可解析的文件")
    else:
        result = batch_parse_local_files(
            file_paths=file_paths,
            model_version="vlm",
            output_dir=str(PROJECT_ROOT / "data" / "mineru_batch"),
        )

        print(f"\nbatch_id: {result['batch_id']}")
        for data_id, res in result["results"].items():
            state = res.get("state")
            content = res.get("content", "")
            print(f"\n{data_id}: {state}")
            if content:
                print(f"  内容预览: {content[:200]}...")