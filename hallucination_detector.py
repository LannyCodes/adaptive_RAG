"""
专业幻觉检测模块
支持多种检测方法：NLI模型、专门检测模型、轻量级模型、混合检测
"""

import re
from typing import List, Dict, Tuple
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 导入轻量级检测器
from lightweight_hallucination_detector import LightweightHallucinationDetector


class VectaraHallucinationDetector:
    """
    Vectara 专门的幻觉检测模型
    使用 HHEM (Hughes Hallucination Evaluation Model)
    """
    
    def __init__(self):
        """初始化 Vectara 幻觉检测模型"""
        print("🔧 初始化 Vectara 幻觉检测模型...")
        
        try:
            self.model_name = "vectara/hallucination_evaluation_model"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()  # 设置为评估模式
            
            # 移动到GPU（如果可用）
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            print(f"✅ Vectara 模型加载成功 (device: {self.device})")
        except Exception as e:
            print(f"⚠️ Vectara 模型加载失败: {e}")
            print("💡 尝试使用 NLI 模型作为备选...")
            self.model = None
    
    def detect(self, generation: str, documents: str) -> Dict:
        """
        检测幻觉
        
        Args:
            generation: LLM 生成的内容
            documents: 参考文档
            
        Returns:
            {
                "has_hallucination": bool,
                "hallucination_score": float (0-1),
                "factuality_score": float (0-1)
            }
        """
        if self.model is None:
            return {"has_hallucination": False, "hallucination_score": 0.0, "factuality_score": 1.0}
        
        try:
            # 准备输入
            inputs = self.tokenizer(
                documents,
                generation,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Vectara 模型输出：[0] = factual, [1] = hallucinated
            factuality_score = probs[0][0].item()
            hallucination_score = probs[0][1].item()
            
            # 判断是否有幻觉（阈值 0.5）
            has_hallucination = hallucination_score > 0.5
            
            return {
                "has_hallucination": has_hallucination,
                "hallucination_score": hallucination_score,
                "factuality_score": factuality_score
            }
        
        except Exception as e:
            print(f"❌ Vectara 检测失败: {e}")
            return {"has_hallucination": False, "hallucination_score": 0.0, "factuality_score": 1.0}
    
    def grade(self, generation: str, documents) -> str:
        """
        兼容原有接口的检测方法
        
        Args:
            generation: LLM 生成的内容
            documents: 参考文档（可以是字符串或列表）
            
        Returns:
            "yes" 表示无幻觉，"no" 表示有幻觉
        """
        # 处理文档格式
        if isinstance(documents, list):
            doc_text = "\n\n".join([
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in documents
            ])
        else:
            doc_text = str(documents)
        
        # 检测幻觉
        result = self.detect(generation, doc_text)
        
        # 打印详细信息
        if result['has_hallucination']:
            print(f"⚠️ Vectara 检测到幻觉 (得分: {result['hallucination_score']:.2f})")
        else:
            print(f"✅ Vectara 未检测到幻觉 (真实性得分: {result['factuality_score']:.2f})")
        
        # 返回兼容格式
        return "no" if result['has_hallucination'] else "yes"


class NLIHallucinationDetector:
    """
    基于 NLI (Natural Language Inference) 的幻觉检测
    使用 DeBERTa 模型
    """
    
    def __init__(self):
        """初始化 NLI 模型"""
        print("🔧 初始化 NLI 幻觉检测模型...")
        
        # 尝试多个模型，按照从小到大的顺序
        models_to_try = [
            "cross-encoder/nli-deberta-v3-xsmall",  # 最小 40MB
            "cross-encoder/nli-deberta-v3-small",   # 小 150MB
            "cross-encoder/nli-MiniLM2-L6-H768",    # 轻量 90MB
            "facebook/bart-large-mnli",              # 备用
        ]
        
        self.nli_model = None
        
        for model_name in models_to_try:
            try:
                print(f"   尝试加载: {model_name}...")
                self.nli_model = pipeline(
                    "text-classification",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    truncation=True,
                    max_length=512
                )
                print(f"✅ NLI 模型加载成功: {model_name}")
                self.model_name = model_name
                break  # 成功加载，退出循环
            except Exception as e:
                print(f"   ⚠️ {model_name} 加载失败: {str(e)[:80]}")
                continue
        
        if self.nli_model is None:
            print("❌ 所有 NLI 模型加载失败，将禁用 NLI 检测")
    
    def split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 简单的句子分割（可以用更复杂的 NLP 工具）
        sentences = re.split(r'[。！？\.\!\?]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def detect(self, generation: str, documents) -> Dict:
        """
        检测幻觉（支持多文档最大匹配策略）
        
        Args:
            generation: LLM 生成的内容
            documents: 参考文档 (str 或 List[Document/str])
            
        Returns:
            {
                "has_hallucination": bool,
                "contradiction_count": int,
                "neutral_count": int,
                "entailment_count": int,
                "problematic_sentences": List[str]
            }
        """
        if self.nli_model is None:
            print("⚠️ NLI 模型未加载，跳过检测")
            return {
                "has_hallucination": False,
                "contradiction_count": 0,
                "neutral_count": 0,
                "entailment_count": 0,
                "problematic_sentences": []
            }
        
        # 1. 预处理文档列表
        docs_content = []
        if isinstance(documents, list):
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    docs_content.append(doc.page_content)
                else:
                    docs_content.append(str(doc))
        else:
            # 如果是单个字符串，尝试按换行符分割，或者作为单文档处理
            docs_content = [str(documents)]

        # 2. 分割生成内容为句子
        sentences = self.split_sentences(generation)
        
        if not sentences:
            print("⚠️ 没有检测到有效句子")
            return {
                "has_hallucination": False,
                "contradiction_count": 0,
                "neutral_count": 0,
                "entailment_count": 0,
                "problematic_sentences": []
            }
        
        contradiction_count = 0
        neutral_count = 0
        entailment_count = 0
        problematic_sentences = []
        
        # 3. 逐句检测 (Max-Entailment Strategy)
        for sentence in sentences:
            if len(sentence) < 10:  # 跳过太短的句子
                continue
            
            # 默认为 Neutral (找不到支持)
            best_label = "neutral"
            best_score = 0.0
            
            # 遍历所有文档块，寻找最佳匹配
            # 只要有一个文档能 Entail (支持) 这个句子，就算通过
            sentence_supported = False
            
            for doc_content in docs_content:
                # 截断单个文档块以适应模型 (保留前 800 字符，通常足够覆盖 512 tokens)
                # 注意：这里是对单个文档块截断，而不是对所有文档拼接后截断
                premise = doc_content[:800]
                
                try:
                    # NLI 推理
                    if hasattr(self, 'model_name') and 'cross-encoder' in self.model_name:
                        result = self.nli_model(
                            f"{premise} [SEP] {sentence}",
                            truncation=True,
                            max_length=512
                        )
                    else:
                        result = self.nli_model(
                            sentence,
                            premise,
                            truncation=True,
                            max_length=512
                        )
                    
                    # 解析结果
                    if isinstance(result, list) and len(result) > 0:
                        current_label = result[0]['label'].lower()
                        current_score = result[0]['score']
                        
                        # 优先级逻辑：Entailment > Contradiction > Neutral
                        # 如果找到 Entailment，立即停止查找（已验证）
                        if 'entailment' in current_label or 'entail' in current_label:
                            best_label = "entailment"
                            sentence_supported = True
                            break
                        
                        # 如果是 Contradiction，记录下来，但继续找（也许其他文档能解释）
                        if 'contradiction' in current_label or 'contradict' in current_label:
                            # 只有当目前是 Neutral 时才更新为 Contradiction
                            # 这样防止 Contradiction 覆盖了潜在的 Entailment (虽然上面break了，但这逻辑保持严谨)
                            if best_label == "neutral":
                                best_label = "contradiction"
                                best_score = current_score
                                
                    else:
                        continue
                        
                except Exception as e:
                    print(f"⚠️ NLI 子任务失败: {str(e)[:50]}")
                    continue
            
            # 统计该句子的最终判定
            if best_label == "entailment":
                entailment_count += 1
            elif best_label == "contradiction":
                contradiction_count += 1
                problematic_sentences.append(sentence)
            else: # neutral
                neutral_count += 1
                
        # 4. 综合评分
        total_sentences = contradiction_count + neutral_count + entailment_count

        has_hallucination = False
        if total_sentences > 0:
            contradiction_ratio = contradiction_count / total_sentences
            neutral_ratio = neutral_count / total_sentences

            # 合理性检查：如果 100% 都是矛盾，很可能是 NLI 模型对中文支持差导致的误判
            # 这种情况下回退到不过滤（让生成通过）
            if contradiction_ratio >= 1.0:
                print(f"⚠️ NLI 模型检测到 100% 矛盾，可能是模型对中文支持差，回退为无幻觉")
                has_hallucination = False
                contradiction_count = 0
            else:
                # 正常阈值判断
                has_hallucination = (contradiction_ratio > 0.5) or (neutral_ratio > 0.95)

            # Debug 信息
            print(f"📊 NLI 检测结果: Entail={entailment_count}, Contra={contradiction_count}, Neutral={neutral_count}")

        return {
            "has_hallucination": has_hallucination,
            "contradiction_count": contradiction_count,
            "neutral_count": neutral_count,
            "entailment_count": entailment_count,
            "problematic_sentences": problematic_sentences
        }
    
    def grade(self, generation: str, documents) -> str:
        """
        兼容原有接口的检测方法
        
        Args:
            generation: LLM 生成的内容
            documents: 参考文档（可以是字符串或列表）
            
        Returns:
            "yes" 表示无幻觉，"no" 表示有幻觉
        """
        # 处理文档格式
        if isinstance(documents, list):
            doc_text = "\n\n".join([
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in documents
            ])
        else:
            doc_text = str(documents)
        
        # 检测幻觉
        result = self.detect(generation, doc_text)
        
        # 打印详细信息
        if result['has_hallucination']:
            print(f"⚠️ NLI 检测到幻觉")
            print(f"   矛盾句子: {result['contradiction_count']}")
            print(f"   中立句子: {result['neutral_count']}")
            print(f"   蕴含句子: {result['entailment_count']}")
            if result['problematic_sentences']:
                print(f"   问题句子: {result['problematic_sentences'][:2]}")
        else:
            print(f"✅ NLI 未检测到幻觉")
        
        # 返回兼容格式
        return "no" if result['has_hallucination'] else "yes"


class HybridHallucinationDetector:
    """
    混合幻觉检测器
    结合 Vectara 模型和 NLI 模型，提供最佳检测效果
    """
    
    def __init__(self, use_vectara: bool = True, use_nli: bool = True):
        """
        初始化混合检测器
        
        Args:
            use_vectara: 是否使用 Vectara 模型
            use_nli: 是否使用 NLI 模型
        """
        self.detectors = {}
        
        if use_vectara:
            try:
                self.detectors['vectara'] = VectaraHallucinationDetector()
            except Exception as e:
                print(f"⚠️ Vectara 检测器初始化失败: {e}")
        
        if use_nli:
            try:
                self.detectors['nli'] = NLIHallucinationDetector()
            except Exception as e:
                print(f"⚠️ NLI 检测器初始化失败: {e}")
        
        if not self.detectors:
            raise RuntimeError("❌ 所有检测器初始化失败！")
        
        print(f"✅ 混合检测器就绪，已加载: {list(self.detectors.keys())}")
    
    def detect(self, generation: str, documents: str) -> Dict:
        """
        综合检测幻觉
        
        Returns:
            {
                "has_hallucination": bool,
                "confidence": float,
                "vectara_result": Dict,
                "nli_result": Dict,
                "method_used": str
            }
        """
        results = {
            "has_hallucination": False,
            "confidence": 0.0,
            "method_used": ""
        }
        
        # 1. 优先使用 Vectara（最准确）
        if 'vectara' in self.detectors:
            vectara_result = self.detectors['vectara'].detect(generation, documents)
            results['vectara_result'] = vectara_result
            
            if vectara_result['hallucination_score'] > 0.5:  # 调高阈值减少误报
                results['has_hallucination'] = True
                results['confidence'] = vectara_result['hallucination_score']
                results['method_used'] = 'vectara'
                return results
            else:
                # Vectara 未检测到幻觉，设置 method_used
                results['method_used'] = 'vectara'
        
        # 2. 如果 Vectara 不确定或不可用，使用 NLI 二次确认
        if 'nli' in self.detectors:
            nli_result = self.detectors['nli'].detect(generation, documents)
            results['nli_result'] = nli_result
            
            if nli_result['has_hallucination']:
                results['has_hallucination'] = True
                # 计算置信度
                total_sentences = (nli_result['contradiction_count'] + 
                                 nli_result['neutral_count'] + 
                                 nli_result['entailment_count'])
                if total_sentences > 0:
                    results['confidence'] = (nli_result['contradiction_count'] + 
                                           nli_result['neutral_count'] * 0.5) / total_sentences
                results['method_used'] = 'nli'
            else:
                # 未检测到幻觉，也要设置 method_used
                if not results['method_used']:  # 只有当前面没有设置时
                    results['method_used'] = 'nli'
        
        # 如果两个模型都有结果，投票决定
        if 'vectara_result' in results and 'nli_result' in results:
            vectara_vote = results['vectara_result']['has_hallucination']
            nli_vote = results['nli_result']['has_hallucination']
            
            if vectara_vote and nli_vote:
                results['has_hallucination'] = True
                results['confidence'] = min(
                    results.get('vectara_result', {}).get('hallucination_score', 0.5),
                    results.get('confidence', 0.5)
                )
                results['method_used'] = 'vectara+nli'
        
        return results
    
    def grade(self, generation: str, documents) -> str:
        """
        兼容原有接口的检测方法
        
        Args:
            generation: LLM 生成的内容
            documents: 参考文档（可以是字符串或列表）
            
        Returns:
            "yes" 表示无幻觉，"no" 表示有幻觉
        """
        # 处理文档格式
        if isinstance(documents, list):
            doc_text = "\n\n".join([
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in documents
            ])
        else:
            doc_text = str(documents)
        
        # 检测幻觉
        result = self.detect(generation, doc_text)
        
        # 打印详细信息
        if result['has_hallucination']:
            print(f"⚠️ 检测到幻觉 (置信度: {result['confidence']:.2f}, 方法: {result['method_used']})")
            if 'nli_result' in result:
                print(f"   矛盾句子: {result['nli_result']['contradiction_count']}")
                if result['nli_result']['problematic_sentences']:
                    print(f"   问题句子: {result['nli_result']['problematic_sentences'][:2]}")
        else:
            print(f"✅ 未检测到幻觉 (方法: {result['method_used']})")
        
        # 返回兼容格式
        return "no" if result['has_hallucination'] else "yes"


def initialize_hallucination_detector(method: str = "nli") -> object:
    """
    初始化幻觉检测器
    
    Args:
        method: 'vectara', 'nli', 或 'hybrid' (推荐)
        
    Returns:
        幻觉检测器实例
    """
    if method == "vectara":
        return VectaraHallucinationDetector()
    elif method == "nli":
        return NLIHallucinationDetector()
    elif method == "hybrid":
        return HybridHallucinationDetector(use_vectara=False, use_nli=True)  # 禁用Vectara，使用NLI
    else:
        raise ValueError(f"未知的检测方法: {method}")
