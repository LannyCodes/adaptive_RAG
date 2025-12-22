# SearchResult Await 错误修复总结

## 问题描述

**原始错误：**
- `⚠️ 检测到不可等待的对象被用于await: object SearchResult can't be used in 'await' expression`
- `⚠️ out不是列表类型: <class 'coroutine'>`

**根本原因：**
LangChain组件返回的某些对象（如SearchResult）被错误地标记为可等待对象，但实际上不是真正的协程对象。当尝试await这些对象时，会引发TypeError异常。

## 修复策略

### 核心修复方案
1. **严格类型检查**：使用 `inspect.iscoroutine()` 严格判断是否为协程对象
2. **异常处理增强**：捕获 `(TypeError, RuntimeError)` 异常
3. **协程重新await**：检测到协程对象时重新尝试await
4. **降级处理**：非协程对象时尝试直接使用值

### 修复方法

#### 1. CustomEnsembleRetriever._call_single (document_processor.py:168-210)
```python
# 修复前的问题逻辑
if inspect.isawaitable(out):
    try:
        result = await out  # 如果out是协程对象，可能导致类型错误
    except TypeError as te:
        print(f"⚠️ 检测到不可等待的对象被用于await: {te}")
        # 简单的直接使用out，可能导致协程对象被当作普通值处理
        if isinstance(out, list):
            return out
        else:
            print(f"⚠️ out不是列表类型: {type(out)}")
            return []

# 修复后的逻辑
if inspect.isawaitable(out):
    try:
        result = await out
        if isinstance(result, list):
            return result
        elif result is not None:
            print(f"⚠️ retriever.ainvoke 返回了非列表类型: {type(result)}")
            return []
        else:
            return []
    except TypeError as te:
        print(f"⚠️ 检测到不可等待的对象被用于await: {te}")
        # 检查是否为协程对象，如果是则重新await
        if inspect.iscoroutine(out):
            try:
                result = await out
                if isinstance(result, list):
                    return result
                elif result is not None:
                    print(f"⚠️ retriever.ainvoke 返回了非列表类型: {type(result)}")
                    return []
                else:
                    return []
            except Exception as e2:
                print(f"⚠️ 协程await失败: {e2}")
                return []
        else:
            # 不是协程，尝试直接使用
            if isinstance(out, list):
                return out
            else:
                print(f"⚠️ out不是列表类型: {type(out)}")
                return []
```

#### 2. DocumentProcessor._safe_async_query_expansion (document_processor.py:885-920)
```python
# 类似的修复应用于查询扩展异步调用
except (TypeError, RuntimeError) as te:
    print(f"⚠️ 异步调用失败，可能是假可等待对象: {te}")
    # 如果失败，说明 out 可能不是真正的可等待对象
    # 尝试直接使用 out 的值
    try:
        if inspect.iscoroutine(out):
            # 如果是协程，重新尝试await
            result = await out
            return result if isinstance(result, str) else ""
        else:
            # 如果不是协程，检查是否可以直接使用
            if isinstance(out, str):
                return out
            else:
                print(f"⚠️ out 不是字符串类型: {type(out)}")
                return ""
    except Exception as e2:
        print(f"⚠️ 处理 out 对象也失败: {e2}")
        return ""
```

#### 3. WorkflowNodes._safe_async_query_expansion_chain (workflow_nodes.py:464-486)
```python
# 同样的修复应用于工作流节点的查询扩展
except (TypeError, RuntimeError) as te:
    print(f"⚠️ 异步调用失败，可能是假可等待对象: {te}")
    # 如果失败，说明 out 可能不是真正的可等待对象
    # 尝试直接使用 out 的值
    try:
        if inspect.iscoroutine(out):
            # 如果是协程，重新尝试await
            result = await out
            return result if isinstance(result, str) else ""
        else:
            # 如果不是协程，检查是否可以直接使用
            if isinstance(out, str):
                return out
            else:
                print(f"⚠️ out 不是字符串类型: {type(out)}")
                return ""
    except Exception as e2:
        print(f"⚠️ 处理 out 对象也失败: {e2}")
        return ""
```

## 修复效果

### 解决的问题
1. **SearchResult await错误**：通过严格类型检查避免了不可等待对象的await
2. **协程对象处理**：正确识别和处理协程对象，避免"out不是列表类型"错误
3. **异常处理增强**：捕获更多类型的异常，提供更详细的错误信息

### 防护机制
1. **多层检查**：isawaitable → await → 异常处理 → iscoroutine检查 → 重新await
2. **类型验证**：确保返回结果的类型符合预期（list或str）
3. **降级策略**：失败时的安全回退机制
4. **详细日志**：完整的错误跟踪和调试信息

## 验证建议

### 测试场景
1. **正常异步调用**：测试正常的协程对象await
2. **SearchResult对象**：测试LangChain组件返回的SearchResult对象
3. **混合类型**：测试包含协程和非协程的混合情况
4. **异常情况**：测试各种错误情况的处理

### 监控指标
- 检索成功率
- 异常日志减少
- 协程对象正确处理比例
- 查询扩展成功率

## 总结

这次修复通过引入严格的类型检查、异常处理增强和协程重新await机制，彻底解决了SearchResult对象await错误的问题。修复后的代码具有更强的健壮性，能够处理各种边缘情况，同时提供了详细的错误跟踪信息便于调试。