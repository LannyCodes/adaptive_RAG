#!/usr/bin/env python3
"""
轻量级幻觉检测器测试脚本
测试效果与性能，替代 Vectara 模型
"""

import time
from lightweight_hallucination_detector import LightweightHallucinationDetector

def test_performance():
    """测试不同模型的性能和效果"""
    print("="*70)
    print("🚀 轻量级幻觉检测器性能测试")
    print("="*70)
    
    # 测试不同模型
    models_to_test = [
        "cross-encoder/nli-MiniLM2-L6-H768",  # 推荐轻量方案
        "cross-encoder/nli-deberta-v3-xsmall",  # 超轻量方案
        "cross-encoder/nli-roberta-base",  # 高准确率方案
    ]
    
    # 测试数据
    documents = "巴黎是法国的首都，这是一座美丽的城市，拥有许多历史地标和博物馆。"
    
    test_cases = [
        ("完全正确", "巴黎是法国的首都。"),
        ("事实错误", "柏林是法国的首都。"),
        ("部分正确", "巴黎是德国的首都，但很美丽。"),
        ("语义等价", "法国的首都是巴黎。"),
        ("无关信息", "纽约是美国的一个大城市。"),
    ]
    
    results = []
    
    for model_name in models_to_test:
        print(f"\n📊 测试模型: {model_name}")
        print("-" * 50)
        
        try:
            detector = LightweightHallucinationDetector(model_name)
            
            model_results = {
                "model": model_name,
                "tests": []
            }
            
            for test_name, test_case in test_cases:
                start_time = time.time()
                result = detector.detect(test_case, documents)
                end_time = time.time()
                
                print(f"  {test_name}:")
                print(f"    假设: {test_case}")
                print(f"    是否幻觉: {result['has_hallucination']}")
                print(f"    幻觉分数: {result['hallucination_score']:.3f}")
                print(f"    推理时间: {end_time - start_time:.3f}秒")
                print()
                
                model_results["tests"].append({
                    "name": test_name,
                    "case": test_case,
                    "result": result,
                    "time": end_time - start_time
                })
            
            results.append(model_results)
            
        except Exception as e:
            print(f"  ❌ 模型测试失败: {e}")
    
    # 总结
    print("\n" + "="*70)
    print("📋 测试总结")
    print("="*70)
    
    for model_result in results:
        model = model_result["model"]
        tests = model_result["tests"]
        
        avg_time = sum(t["time"] for t in tests) / len(tests)
        correct_count = 0
        
        # 评估准确性
        expected_results = [False, True, True, False, False]  # 预期结果
        for i, test in enumerate(tests):
            if test["result"]["has_hallucination"] == expected_results[i]:
                correct_count += 1
        
        accuracy = correct_count / len(tests) * 100
        
        print(f"\n🤖 {model}:")
        print(f"  ⚡ 平均推理时间: {avg_time:.3f}秒")
        print(f"  🎯 准确率: {accuracy:.1f}% ({correct_count}/{len(tests)})")
        print(f"  📊 幻觉检测评分: {sum(t['result']['hallucination_score'] for t in tests):.2f}")

def test_rag_scenarios():
    """测试RAG场景下的幻觉检测"""
    print("\n" + "="*70)
    print("🔍 RAG场景测试")
    print("="*70)
    
    # RAG测试数据
    rag_documents = """
    产品信息：iPhone 14 Pro 是苹果公司在2022年9月发布的旗舰智能手机。
    主要特性：配备6.1英寸Super Retina XDR显示屏，A16仿生芯片，4800万像素主摄像头。
    电池续航：视频播放最长可达23小时，支持20W有线快充。
    价格：起售价为799美元。
    """
    
    rag_test_cases = [
        ("准确信息", "iPhone 14 Pro配备了A16仿生芯片和4800万像素摄像头。"),
        ("规格错误", "iPhone 14 Pro配备A15仿生芯片和1200万像素摄像头。"),
        ("价格错误", "iPhone 14 Pro的起售价为999美元。"),
        ("无关信息", "iPhone 14 Pro支持手写笔输入。"),
        ("混合信息", "iPhone 14 Pro配备A16芯片，起售价999美元，支持手写笔。"),
    ]
    
    detector = LightweightHallucinationDetector()
    
    print("🧪 RAG幻觉检测测试：\n")
    
    for test_name, test_case in rag_test_cases:
        result = detector.detect(test_case, rag_documents, method="sentence_level")
        
        print(f"📋 {test_name}:")
        print(f"   生成内容: {test_case}")
        print(f"   检测结果: {'🚨 检测到幻觉' if result['has_hallucination'] else '✅ 未检测到幻觉'}")
        print(f"   幻觉分数: {result['hallucination_score']:.3f}")
        print(f"   事实性分数: {result['factuality_score']:.3f}")
        
        if result['details'].get('problematic_sentences'):
            print(f"   问题句子数: {len(result['details']['problematic_sentences'])}")
            for i, prob in enumerate(result['details']['problematic_sentences'], 1):
                print(f"     {i}. {prob['sentence']} (分数: {prob['score']:.3f})")
        
        print()

if __name__ == "__main__":
    # 1. 性能测试
    test_performance()
    
    # 2. RAG场景测试
    test_rag_scenarios()
    
    print("\n" + "="*70)
    print("💡 使用建议:")
    print("1. 生产环境推荐使用 cross-encoder/nli-MiniLM2-L6-H768")
    print("2. 资源受限环境可使用 cross-encoder/nli-deberta-v3-xsmall")
    print("3. 高准确率需求可使用 cross-encoder/nli-roberta-base")
    print("4. 建议设置幻觉分数阈值为 0.6-0.7")
    print("="*70)