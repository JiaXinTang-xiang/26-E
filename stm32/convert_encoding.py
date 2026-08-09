# -*- coding: utf-8 -*-
"""
将 GBK 编码的 C 源文件转换为 UTF-8 编码
运行方式: python convert_encoding.py
"""
import os
import codecs

# 需要转换的目录
dirs_to_convert = [
    'Core/Src',
    'Core/Inc',
]

# 需要转换的文件扩展名
extensions = ('.c', '.h')

def convert_file(filepath):
    """将单个文件从 GBK 转换为 UTF-8"""
    try:
        # 先尝试用 GBK 读取
        with open(filepath, 'rb') as f:
            raw = f.read()

        # 检测是否已经是 UTF-8
        try:
            raw.decode('utf-8')
            print(f'  [跳过] {filepath} (已经是 UTF-8)')
            return False
        except UnicodeDecodeError:
            pass

        # 用 GBK 解码
        try:
            text = raw.decode('gbk')
        except UnicodeDecodeError:
            # 如果 GBK 失败，尝试 gb18030（GBK 的超集）
            try:
                text = raw.decode('gb18030')
            except UnicodeDecodeError:
                print(f'  [失败] {filepath} (无法解码)')
                return False

        # 写回 UTF-8 编码
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f'  [完成] {filepath}')
        return True

    except Exception as e:
        print(f'  [错误] {filepath}: {e}')
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    converted = 0
    skipped = 0
    failed = 0

    print("=" * 60)
    print("GBK → UTF-8 编码转换工具")
    print("=" * 60)

    for dir_name in dirs_to_convert:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"\n[警告] 目录不存在: {dir_path}")
            continue

        print(f"\n处理目录: {dir_name}")
        print("-" * 40)

        for filename in os.listdir(dir_path):
            if filename.endswith(extensions):
                filepath = os.path.join(dir_path, filename)
                result = convert_file(filepath)
                if result is True:
                    converted += 1
                elif result is False:
                    skipped += 1
                else:
                    failed += 1

    print("\n" + "=" * 60)
    print(f"转换完成! 已转换: {converted}, 跳过: {skipped}, 失败: {failed}")
    print("=" * 60)

if __name__ == '__main__':
    main()
