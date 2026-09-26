"""
Example 7 V2: MinerU API 解析 PDF 文件
功能：通过 MinerU API 解析 PDF，支持 pdf、doc、ppt、图片等多种格式
"""

import os
import time
import requests
import zipfile
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==================== 配置 ====================

MINERU_API_KEY = os.getenv("MINERU_API_KEY")
MINERU_URL = "https://mineru.net/api/v4/extract/task"
MODEL_VERSION = "vlm"


# ==================== 1. 创建解析任务 ====================

def create_parse_task(file_url: str, model_version: str = MODEL_VERSION) -> str:
    """创建文档解析任务，返回 task_id"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_KEY}",
    }
    data = {
        "url": file_url,
        "model_version": model_version,
    }

    response = requests.post(MINERU_URL, headers=headers, json=data)
    response.raise_for_status()

    result = response.json()
    if result.get("code") != 0:
        raise Exception(f"创建任务失败: {result.get('msg')}")

    task_id = result["data"]["task_id"]
    print(f"[创建任务] task_id: {task_id}")
    return task_id


# ==================== 2. 查询任务状态 ====================

def get_task_result(task_id: str) -> dict:
    """查询任务结果"""
    url = f"https://mineru.net/api/v4/extract/task/{task_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_KEY}",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()


def wait_for_completion(task_id: str, poll_interval: int = 5, max_wait: int = 600) -> dict:
    """等待任务完成"""
    start_time = time.time()

    while True:
        result = get_task_result(task_id)
        state = result["data"]["state"]

        elapsed = time.time() - start_time
        print(f"[查询状态] {state}, 已等待 {elapsed:.0f}s")

        if state == "done":
            return result
        elif state == "failed":
            raise Exception(f"解析失败: {result['data'].get('err_msg', '未知错误')}")
        elif elapsed > max_wait:
            raise Exception(f"等待超时（{max_wait}s）")

        time.sleep(poll_interval)


# ==================== 3. 下载并解压结果 ====================

def download_and_extract(url: str, output_dir: str) -> str:
    """下载 zip 文件并解压，返回 markdown 内容"""
    print(f"[下载] {url}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        zip_path = f.name

    with open(zip_path, "wb") as f:
        f.write(response.content)

    print(f"[解压] {zip_path} -> {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # 删除临时 zip 文件
    os.remove(zip_path)

    # 查找 full.md 文件
    output_path = Path(output_dir)
    md_files = list(output_path.glob("*.md")) + list(output_path.glob("**/*.md"))

    if md_files:
        # 返回第一个 md 文件的内容
        md_file = md_files[0]
        print(f"[解析结果] {md_file}")
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    raise Exception("未找到 .md 文件")


# ==================== 4. 完整解析流程 ====================

def parse_document(
    file_url: str,
    output_dir: str = "./data/mineru_output",
    model_version: str = MODEL_VERSION,
    wait_completion: bool = True,
) -> dict:
    """
    解析文档的完整流程

    Args:
        file_url: 文件 URL（支持 pdf、doc、docx、ppt、pptx、图片、html）
        output_dir: 输出目录
        model_version: 模型版本，vlm 或 pipeline，HTML 用 MinerU-HTML
        wait_completion: 是否等待完成，False 则只创建任务返回 task_id

    Returns:
        包含 task_id、state、content 等信息的字典
    """
    print("=" * 50)
    print("MinerU 文档解析")
    print("=" * 50)
    print(f"文件 URL: {file_url}")
    print(f"模型版本: {model_version}")

    # 1. 创建任务
    print("\n[1] 创建解析任务...")
    task_id = create_parse_task(file_url, model_version)

    if not wait_completion:
        return {"task_id": task_id, "state": "pending"}

    # 2. 等待完成
    print("\n[2] 等待解析完成...")
    result = wait_for_completion(task_id)

    # 3. 下载结果
    print("\n[3] 下载解析结果...")
    full_zip_url = result["data"]["full_zip_url"]
    content = download_and_extract(full_zip_url, output_dir)

    print("\n" + "=" * 50)
    print("解析完成！")
    print("=" * 50)

    return {
        "task_id": task_id,
        "state": "done",
        "content": content,
        "output_dir": output_dir,
    }


# ==================== 5. 示例用法 ====================

if __name__ == "__main__":
    # 示例文件 URL（可以使用你自己的文件 URL）
    file_url = "https://cdn-mineru.openxlab.org.cn/demo/example.pdf"

    # 也可以使用本地文件的 URL（需要先上传到可访问的服务器）
    # file_url = "https://your-server.com/your-file.pdf"

    # 解析文档
    result = parse_document(
        file_url=file_url,
        output_dir="./data/mineru_output",
        model_version="vlm",
    )

    # 打印结果
    print(f"\n任务 ID: {result['task_id']}")
    print(f"状态: {result['state']}")
    print(f"\n解析内容（前 500 字符）:\n{result['content'][:500]}...")