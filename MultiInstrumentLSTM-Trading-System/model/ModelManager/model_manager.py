import os
import json
import pickle
import datetime
import tensorflow as tf

class ModelManager:
    def __init__(self, model_dir='models'):
        """
        初始化模型管理模块
        
        参数:
        - model_dir: 模型保存目录
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model, model_name, metadata=None):
        """
        保存模型
        
        参数:
        - model: 模型实例
        - model_name: 模型名称
        - metadata: 模型元数据
        
        返回:
        - model_path: 模型保存路径
        """
        # 创建模型保存目录
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_dir = os.path.join(self.model_dir, f'{model_name}_{timestamp}')
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 保存模型权重
        model.save_model(model_save_dir)
        
        # 保存元数据
        if metadata:
            # 分离出scalers，单独保存
            scalers = metadata.pop('scalers', None)
            if scalers:
                scalers_path = os.path.join(model_save_dir, 'scalers.pkl')
                with open(scalers_path, 'wb') as f:
                    pickle.dump(scalers, f)
            
            # 将Timestamp对象转换为字符串
            def convert_timestamps(obj):
                if isinstance(obj, list):
                    return [convert_timestamps(item) for item in obj]
                elif hasattr(obj, 'isoformat'):  # Handle datetime/Timestamp objects
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {key: convert_timestamps(value) for key, value in obj.items()}
                return obj
            
            # 转换元数据中的Timestamp对象
            json_metadata = convert_timestamps(metadata)
            
            # 保存剩余元数据为JSON
            metadata_path = os.path.join(model_save_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
        
        return model_save_dir
    
    def load_model(self, model_path, model_class):
        """
        加载模型
        
        参数:
        - model_path: 模型保存路径
        - model_class: 模型类
        
        返回:
        - model: 模型实例
        - metadata: 模型元数据
        """
        # 加载模型
        model = model_class(None, None)  # 先创建空模型
        model.load_model(model_path)
        
        # 加载元数据
        metadata = {}
        
        # 加载JSON元数据
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # 加载scalers
        scalers_path = os.path.join(model_path, 'scalers.pkl')
        if os.path.exists(scalers_path):
            with open(scalers_path, 'rb') as f:
                metadata['scalers'] = pickle.load(f)
        
        return model, metadata
    
    def get_latest_model(self, model_name):
        """
        获取最新保存的模型
        
        参数:
        - model_name: 模型名称
        
        返回:
        - latest_model_path: 最新模型路径
        """
        # 列出所有模型目录
        model_dirs = []
        for dir_name in os.listdir(self.model_dir):
            if dir_name.startswith(model_name):
                model_dirs.append(dir_name)
        
        if not model_dirs:
            return None
        
        # 按时间戳排序，获取最新的模型
        model_dirs.sort(reverse=True)
        latest_model_path = os.path.join(self.model_dir, model_dirs[0])
        
        return latest_model_path
    
    def list_models(self, model_name=None):
        """
        列出所有模型
        
        参数:
        - model_name: 模型名称（可选）
        
        返回:
        - model_list: 模型列表
        """
        model_list = []
        
        for dir_name in os.listdir(self.model_dir):
            if model_name is None or dir_name.startswith(model_name):
                model_path = os.path.join(self.model_dir, dir_name)
                if os.path.isdir(model_path):
                    # 获取模型创建时间
                    timestamp = dir_name.split('_')[-1]
                    try:
                        create_time = datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                    except ValueError:
                        create_time = None
                    
                    model_list.append({
                        'name': dir_name,
                        'path': model_path,
                        'create_time': create_time
                    })
        
        # 按创建时间排序
        model_list.sort(key=lambda x: x['create_time'], reverse=True)
        
        return model_list
    
    def delete_model(self, model_path):
        """
        删除模型
        
        参数:
        - model_path: 模型路径
        """
        if os.path.exists(model_path):
            tf.io.gfile.rmtree(model_path)
