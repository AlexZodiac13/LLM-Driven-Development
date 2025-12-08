import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class AttentionExtractor:
    """Класс для извлечения и обработки attention весов из трансформер моделей"""
    
    def __init__(self, model_name, device='cpu'):
        """
        Инициализация экстрактора
        Args:
            model_name: Название модели на Hugging Face
            device: 'cpu' или 'cuda'
        """
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            output_attentions=True
        ).to(device)
        self.model.eval()
    
    def get_attention_weights(self, text):
        """
        Получить attention веса для текста
        Args:
            text: Входной текст
        Returns:
            dict с attention информацией
        """
        inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs[0])
        
        with torch.no_grad():
            outputs = self.model(inputs)
            attention = outputs[-1]
        
        return {
            'attention': attention,
            'tokens': tokens,
            'input_ids': inputs[0]
        }
    
    def get_layer_attention(self, text, layer_idx=0, head_idx=None):
        """
        Получить attention для конкретного слоя/головы
        Args:
            text: Входной текст
            layer_idx: Индекс слоя
            head_idx: Индекс головы (если None, усредняет по всем головам)
        Returns:
            numpy array attention матрица
        """
        result = self.get_attention_weights(text)
        attention = result['attention'][layer_idx][0]  # [batch, head, seq, seq]
        
        if head_idx is None:
            attention = attention.mean(dim=0)  # Усреднить по головам
        else:
            attention = attention[head_idx]
        
        return attention.cpu().numpy()
    
    def get_all_heads_attention(self, text, layer_idx=0):
        """Получить attention для всех голов конкретного слоя"""
        result = self.get_attention_weights(text)
        attention = result['attention'][layer_idx][0]  # [head, seq, seq]
        return attention.cpu().numpy(), result['tokens']
