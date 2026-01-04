import os
import shutil

def clean_target_position_dir():
    """清理target_position目录"""
    target_dir = 'target_position'
    
    # 检查目录是否存在
    if os.path.exists(target_dir):
        # 删除目录下所有文件
        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"已清理{target_dir}目录下所有文件")
    else:
        # 如果目录不存在，创建它
        os.makedirs(target_dir)
        print(f"已创建{target_dir}目录")

if __name__ == "__main__":
    clean_target_position_dir()