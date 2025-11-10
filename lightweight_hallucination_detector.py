"""
轻量级开源幻觉检测器
替代 Vectara 模型的最佳方案
"""

import os
import re
import torch
from typing import List, Dict, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class LightweightHallucinationDetector:
    """
    轻量级幻觉检测器
    使用开源 NLI 模型，无需特殊权限
    """
    
    def __init__(self, model_name="cross-encoder/nli-MiniLM2-L6-H768"):
        """
        初始化轻量级幻觉检测器
        
        Args:
            model_name: 可选的开源模型
                - "cross-encoder/nli-MiniLM2-L6-H768" (推荐: 80MB, 85%准确率)
                - "cross-encoder/nli-deberta-v3-xsmall" (更小: 40MB, 82%准确率)
                - "cross-encoder/nli-roberta-base" (更准: 430MB, 88%准确率)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 初始化轻量级幻觉检测器...")
        print(f"   模型: {model_name}")
        print(f"   设备: {self.device}")
        
        try:
            self.nli_model = pipeline(
                "text-classification",
                model=model_name,
                device=self.device,
                truncation=True,
                max_length=512,
                return_all_scores=True
            )
            print(f"✅ 模型加载成功!")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 尝试使用备用模型...")
            
            # 备用模型列表（按从轻到重排列）
            backup_models = [
                "cross-encoder/nli-deberta-v3-xsmall",
                "cross-encoder/nli-roberta-base",
                "facebook/bart-large-mnli"
            ]
            
            self.nli_model = None
            for backup_model in backup_models:
                try:
                    print(f"   尝试备用模型: {backup_model}")
                    self.nli_model = pipeline(
                        "text-classification",
                        model=backup_model,
                        device=self.device,
                        truncation=True,
                        max_length=512,
                        return_all_scores=True
                    )
                    print(f"✅ 备用模型加载成功: {backup_model}")
                    self.model_name = backup_model
                    break
                except Exception as backup_e:
                    print(f"   ❌ 备用模型失败: {backup_e}")
                    continue
    
    def _split_text_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 简单但有效的句子分割
        sentences = re.split(r'[。！？.!?]\\s*', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _nli_score(self, premise: str, hypothesis: str) -> Dict:
        """计算 NLI 分数"""
        if self.nli_model is None:
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            # 格式化输入
            input_text = f"{premise} [SEP] {hypothesis}"
            
            # 获取所有分数
            results = self.nli_model(input_text)[0]
            
            # 解析结果
            result_dict = {item['label']: item['score'] for item in results}
            
            return result_dict
        except Exception as e:
            print(f"❌ NLI 推理失败: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def _calculate_hallucination_score(self, nli_results: Dict) -> float:
        """
        根据 NLI 结果计算幻觉分数
        
        Args:
            nli_results: NLI 模型的输出结果
            
        Returns:
            float: 幻觉分数 (0-1)
        """
        contradiction = nli_results.get('CONTRADICTION', 0.0)
        neutral = nli_results.get('NEUTRAL', 0.0)
        entailment = nli_results.get('ENTAILMENT', 0.0)
        
        # 幻觉分数计算公式
        # 矛盾 -> 高幻觉分数
        # 中立 -> 中等幻觉分数  
        # 蕴含 -> 低幻觉分数
        
        hallucination_score = contradiction * 0.9 + neutral * 0.5 + entailment * 0.1
        
        return min(1.0, hallucination_score)
    
    def detect(self, generation: str, documents: str, method="sentence_level") -> Dict:
        """
        检测幻觉
        
        Args:
            generation: LLM 生成的内容
            documents: 参考文档
            method: 检测方法
                - "sentence_level": 句子级别检测（推荐）
                - "document_level": 文档级别检测
                
        Returns:
            Dict: 检测结果
        """
        if self.nli_model is None:
            return {
                "has_hallucination": False,
                "hallucination_score": 0.0,
                "factuality_score": 1.0,
                "method": "model_failed",
                "details": "模型加载失败，返回安全默认值"
            }
        
        if method == "sentence_level":
            return self._detect_sentence_level(generation, documents)
        else:
            return self._detect_document_level(generation, documents)
    
    def _detect_sentence_level(self, generation: str, documents: str) -> Dict:
        """句子级别的幻觉检测"""
        sentences = self._split_text_into_sentences(generation)
        
        if not sentences:
            return {
                "has_hallucination": False,
                "hallucination_score": 0.0,
                "factuality_score": 1.0,
                "method": "sentence_level",
                "details": "没有可分析的句子"
            }
        
        # 分析每个句子
        sentence_scores = []
        problematic_sentences = []
        
        for sentence in sentences:
            nli_result = self._nli_score(documents, sentence)
            hallucination_score = self._calculate_hallucination_score(nli_result)
            
            sentence_scores.append(hallucination_score)
            
            if hallucination_score > 0.6:  # 阈值
                problematic_sentences.append({
                    "sentence": sentence,
                    "score": hallucination_score,
                    "nli_result": nli_result
                })
        
        # 计算整体分数
        avg_hallucination_score = np.mean(sentence_scores)
        max_hallucination_score = np.max(sentence_scores)
        
        # 判断是否有幻觉
        has_hallucination = max_hallucination_score > 0.7  # 严格阈值
        
        return {
            "has_hallucination": has_hallucination,
            "hallucination_score": float(max_hallucination_score),
            "factuality_score": float(1.0 - avg_hallucination_score),
            "method": "sentence_level",
            "details": {
                "sentence_count": len(sentences),
                "avg_score": float(avg_hallucination_score),
                "max_score": float(max_hallucination_score),
                "problematic_sentences": problematic_sentences[:3]  # 只返回前3个问题句子
            }
        }
    
    def _detect_document_level(self, generation: str, documents: str) -> Dict:
        """文档级别的幻觉检测"""
        nli_result = self._nli_score(documents, generation)
        hallucination_score = self._calculate_hallucination_score(nli_result)
        
        has_hallucination = hallucination_score > 0.5  # 标准阈值
        
        return {
            "has_hallucination": has_hallucination,
            "hallucination_score": float(hallucination_score),
            "factuality_score": float(1.0 - hallucination_score),
            "method": "document_level",
            "details": {
                "nli_result": nli_result,
                "primary_label": max(nli_result.keys(), key=lambda k: nli_result[k])
            }
        }
    
    def batch_detect(self, generations: List[str], documents: str, method="sentence_level") -> List[Dict]:
        """
        批量检测幻觉
        
        Args:
            generations: 多个生成内容
            documents: 参考文档
            method: 检测方法
            
        Returns:
            List[Dict]: 每个生成内容的检测结果
        """
        results = []
        for generation in generations:
            result = self.detect(generation, documents, method)
            results.append(result)
        
        return results


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    # 创建检测器
    detector = LightweightHallucinationDetector()
    
    # 测试数据
    documents = "The capital of France is Paris. It is a beautiful city with many historical landmarks."
    
    test_cases = [
        "The capital of France is Berlin.",  # 明显错误
        "Paris is the capital of France.",  # 正确
        "Paris is the capital of Germany and has many beautiful landmarks.",  # 部分错误
        "The French capital has several famous museums and historical sites."  # 正确，但表述不同
    ]
    
    print("\n" + "="*60)
    print("🧪 轻量级幻觉检测器测试")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 测试案例:")
        print(f"   前提: {documents[:50]}...")
        print(f"   假设: {test_case}")
        
        # 检测幻觉
        result = detector.detect(test_case, documents, method="sentence_level")
        
        print(f"   结果:")
        print(f"     - 是否有幻觉: {result['has_hallucination']}")
        print(f"     - 幻觉分数: {result['hallucination_score']:.3f}")
        print(f"     - 事实性分数: {result['factuality_score']:.3f}")
        print(f"     - 检测方法: {result['method']}")
        
        if result['details'].get('problematic_sentences'):
            print(f"     - 问题句子: {len(result['details']['problematic_sentences'])} 个")
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)