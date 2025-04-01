class ModelManager:
    """
    只管理 Transformer 预训练模型的加载状态，不负责实际的模型加载。
    """
    _models = {}  # 存储已加载的模型
    _tokenizers = {}  # 存储已加载的 tokenizer

    @classmethod
    def is_model_loaded(cls, model_name):
        """
        检查模型是否已加载。

        参数：
        - model_name: str，例如 "gpt2-medium", "bert-base-uncased"

        返回：
        - bool: 如果模型已加载，返回 True，否则返回 False
        """
        return model_name in cls._models

    @classmethod
    def register_model(cls, model_name, model, tokenizer):
        """
        将已加载的模型注册到 ModelManager。

        参数：
        - model_name: str
        - model: 预训练模型
        - tokenizer: 预训练模型的 tokenizer
        """
        if model_name not in cls._models:
            cls._models[model_name] = model
            cls._tokenizers[model_name] = tokenizer
            print(f"Model {model_name} registered successfully.")

    @classmethod
    def get_model(cls, model_name):
        """
        获取已注册的模型和 tokenizer。

        参数：
        - model_name: str

        返回：
        - (model, tokenizer): 如果模型已注册，返回模型和 tokenizer
        - 如果模型未注册，抛出 ValueError
        """
        if model_name not in cls._models:
            raise ValueError(f"Model {model_name} has not been loaded yet. Please load it first.")
        return cls._models[model_name], cls._tokenizers[model_name]
