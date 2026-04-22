"""
LangSmith 集成模块
为自适应 RAG 系统提供可观测性和追踪功能

功能列表:
1. 基础追踪: 自动追踪所有 LangChain/LangGraph 操作
2. 自定义追踪: 记录检索、生成、评分等自定义事件
3. 性能监控: 收集和追踪各阶段性能指标
4. 告警系统: 设置阈值并在性能异常时触发告警
5. 统计分析: 提供查询统计和性能分析报告
"""

import os
import time
import json
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from langchain.callbacks import LangChainTracer
from langchain_core.runnables import RunnableConfig
import langsmith
from enum import Enum


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    name: str
    value: float
    unit: str = "ms"
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    threshold: float
    operator: str  # ">", "<", ">=", "<=", "=="
    level: AlertLevel
    enabled: bool = True
    cooldown_seconds: int = 300  # 冷却时间，避免重复告警


class LangSmithManager:
    """LangSmith 集成管理器 - 增强版"""
    
    def __init__(
        self,
        project_name: str = "adaptive-rag-project",
        api_key: Optional[str] = None,
        tracing_enabled: bool = True,
        enable_performance_monitoring: bool = True,
        enable_alerts: bool = True
    ):
        """
        初始化 LangSmith 管理器
        
        Args:
            project_name: LangSmith 项目名称
            api_key: LangSmith API 密钥
            tracing_enabled: 是否启用基础追踪
            enable_performance_monitoring: 是否启用性能监控
            enable_alerts: 是否启用告警系统
        """
        self.project_name = project_name
        self.tracing_enabled = tracing_enabled
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_alerts = enable_alerts
        self.tracer: Optional[LangChainTracer] = None
        
        # 性能指标存储
        self.metrics: List[PerformanceMetric] = []
        self.query_history: List[Dict] = []
        
        # 告警系统
        self.alert_rules: List[AlertRule] = []
        self.alert_callbacks: List[Callable] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # 从环境变量或参数获取 API 密钥
        self.api_key = api_key or os.environ.get("LANGSMITH_API_KEY")
        # 确保 LANGCHAIN_API_KEY 环境变量被设置（新版 LangChainTracer 依赖此变量）
        if self.api_key and not os.environ.get("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
        
        # 从环境变量读取配置
        self.auto_flush = os.environ.get("LANGSMITH_AUTO_FLUSH", "true").lower() == "true"
        self.flush_interval = int(os.environ.get("LANGSMITH_FLUSH_INTERVAL", "60"))
        
        if self.tracing_enabled and self.api_key:
            self._setup_tracer()
            self._setup_default_alert_rules()
        elif self.tracing_enabled:
            print("⚠️ 未找到 LangSmith API 密钥，追踪功能将不可用")
    
    def _setup_tracer(self):
        """设置 LangChain 追踪器"""
        try:
            self.tracer = LangChainTracer(
                project_name=self.project_name,
                api_key=self.api_key
            )
            print(f"✅ LangSmith 追踪已启用，项目: {self.project_name}")
        except TypeError:
            # 新版 LangChainTracer 不接受 api_key 参数，从环境变量读取
            try:
                self.tracer = LangChainTracer(project_name=self.project_name)
                print(f"✅ LangSmith 追踪已启用，项目: {self.project_name}")
            except Exception as e:
                print(f"❌ LangSmith 追踪设置失败: {e}")
                self.tracer = None
        except Exception as e:
            print(f"❌ LangSmith 追踪设置失败: {e}")
            self.tracer = None
    
    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        # 默认告警规则：检索超时、生成超时、Token使用过高等
        self.add_alert_rule(AlertRule(
            name="检索超时告警",
            metric_name="retrieve_latency",
            threshold=30000,  # 30秒
            operator=">",
            level=AlertLevel.WARNING
        ))
        
        self.add_alert_rule(AlertRule(
            name="生成超时告警",
            metric_name="generate_latency",
            threshold=60000,  # 60秒
            operator=">",
            level=AlertLevel.WARNING
        ))
        
        self.add_alert_rule(AlertRule(
            name="总响应时间告警",
            metric_name="total_latency",
            threshold=120000,  # 120秒
            operator=">",
            level=AlertLevel.ERROR
        ))
        
        self.add_alert_rule(AlertRule(
            name="Token使用过高告警",
            metric_name="total_tokens",
            threshold=4000,
            operator=">",
            level=AlertLevel.INFO
        ))
    
    def get_callback_config(self) -> RunnableConfig:
        """
        获取回调配置，用于 LangGraph 编译
        
        Returns:
            RunnableConfig: 包含回调的配置
        """
        if self.tracer and self.tracing_enabled:
            return {
                "callbacks": [self.tracer]
            }
        return {}
    
    # ==================== 自定义追踪功能 ====================
    
    def log_custom_event(
        self,
        name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        记录自定义事件到 LangSmith
        
        Args:
            name: 事件名称
            inputs: 输入数据
            outputs: 输出数据（可选）
            metadata: 元数据（可选）
        """
        if not (self.tracing_enabled and self.api_key):
            return
        
        try:
            client = langsmith.Client(api_key=self.api_key)
            
            run_data = {
                "name": name,
                "inputs": inputs,
                "outputs": outputs or {},
                "metadata": metadata or {},
                "run_type": "chain"
            }
            
            # 使用create_run记录事件
            client.create_run(
                project_name=self.project_name,
                **run_data
            )
            
        except Exception as e:
            print(f"⚠️ 记录自定义事件失败: {e}")
    
    def log_retrieval_event(
        self,
        query: str,
        documents_count: int,
        retrieval_time: float,
        top_k: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        记录检索事件
        
        Args:
            query: 查询内容
            documents_count: 检索到的文档数量
            retrieval_time: 检索耗时（毫秒）
            top_k: 请求的top-k值
            metadata: 额外元数据
        """
        if not self.enable_performance_monitoring:
            return
        
        # 记录性能指标
        self.record_metric("retrieve_latency", retrieval_time, "ms", {
            "query_length": str(len(query)),
            "documents_count": str(documents_count),
            "top_k": str(top_k)
        })
        
        # 记录检索事件
        self.log_custom_event(
            name="retrieval",
            inputs={"query": query, "top_k": top_k},
            outputs={
                "documents_count": documents_count,
                "retrieval_time_ms": retrieval_time
            },
            metadata=metadata
        )
    
    def log_generation_event(
        self,
        prompt: str,
        generation: str,
        generation_time: float,
        tokens_used: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        记录生成事件
        
        Args:
            prompt: 提示词
            generation: 生成的文本
            generation_time: 生成耗时（毫秒）
            tokens_used: 使用的token数量
            metadata: 额外元数据
        """
        if not self.enable_performance_monitoring:
            return
        
        # 记录性能指标
        self.record_metric("generate_latency", generation_time, "ms", {
            "prompt_length": str(len(prompt)),
            "generation_length": str(len(generation)),
            "tokens_used": str(tokens_used)
        })
        
        self.record_metric("generation_length", len(generation), "chars")
        self.record_metric("total_tokens", tokens_used, "tokens")
        
        # 记录生成事件
        self.log_custom_event(
            name="generation",
            inputs={"prompt": prompt[:500] if prompt else ""},  # 限制长度
            outputs={
                "generation_length": len(generation),
                "generation_time_ms": generation_time,
                "tokens_used": tokens_used
            },
            metadata=metadata
        )
    
    def log_query_complete(
        self,
        question: str,
        answer: str,
        total_latency: float,
        routing_decision: str,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        记录完整查询完成事件
        
        Args:
            question: 用户问题
            answer: 生成答案
            total_latency: 总耗时（毫秒）
            routing_decision: 路由决策
            metrics: 评估指标
        """
        if not self.enable_performance_monitoring:
            return
        
        # 记录总耗时
        self.record_metric("total_latency", total_latency, "ms", {
            "routing_decision": routing_decision,
            "question_length": str(len(question)),
            "answer_length": str(len(answer))
        })
        
        # 记录查询历史
        query_record = {
            "question": question,
            "answer": answer,
            "total_latency_ms": total_latency,
            "routing_decision": routing_decision,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {}
        }
        self.query_history.append(query_record)
        
        # 限制历史记录大小
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-500:]
        
        # 记录完整查询事件
        self.log_custom_event(
            name="query_complete",
            inputs={"question": question, "routing_decision": routing_decision},
            outputs={
                "answer_length": len(answer),
                "total_latency_ms": total_latency,
                "metrics": metrics or {}
            }
        )
    
    # ==================== 性能监控功能 ====================
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "ms",
        tags: Optional[Dict[str, str]] = None
    ):
        """
        记录性能指标
        
        Args:
            name: 指标名称
            value: 指标值
            unit: 单位
            tags: 标签
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        self.metrics.append(metric)
        
        # 限制指标数量
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-5000:]
        
        # 检查告警规则
        if self.enable_alerts:
            self._check_alert_rules(metric)
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """
        获取指定指标的统计信息
        
        Args:
            metric_name: 指标名称
            
        Returns:
            Dict: 包含count, sum, avg, min, max, p50, p95, p99的字典
        """
        relevant_metrics = [m for m in self.metrics if m.name == metric_name]
        
        if not relevant_metrics:
            return {
                "count": 0,
                "sum": 0,
                "avg": 0,
                "min": 0,
                "max": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0
            }
        
        values = sorted([m.value for m in relevant_metrics])
        count = len(values)
        total = sum(values)
        
        def percentile(p: float) -> float:
            if not values:
                return 0
            idx = int(len(values) * p / 100)
            return values[min(idx, len(values) - 1)]
        
        return {
            "count": count,
            "sum": total,
            "avg": total / count,
            "min": min(values),
            "max": max(values),
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99)
        }
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        生成性能报告

        Args:
            hours: 统计最近多少小时的数据，默认 24

        Returns:
            Dict: 性能报告
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self.query_history),
            "total_metrics": len(self.metrics),
            "metrics_summary": {},
            "query_stats": {},
            "alerts_triggered": len(self.last_alert_times)
        }
        
        # 关键指标统计
        key_metrics = [
            "retrieve_latency",
            "generate_latency",
            "total_latency",
            "total_tokens"
        ]
        
        for metric_name in key_metrics:
            stats = self.get_metric_stats(metric_name)
            if stats["count"] > 0:
                report["metrics_summary"][metric_name] = stats
        
        # 查询统计
        if self.query_history:
            latencies = [q["total_latency_ms"] for q in self.query_history if "total_latency_ms" in q]
            if latencies:
                report["query_stats"] = {
                    "total_queries": len(self.query_history),
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
                    "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)]
                }
        
        return report
    
    # ==================== 告警系统 ====================
    
    def add_alert_rule(self, rule: AlertRule):
        """
        添加告警规则
        
        Args:
            rule: 告警规则
        """
        self.alert_rules.append(rule)
        print(f"✅ 添加告警规则: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """
        移除告警规则
        
        Args:
            rule_name: 规则名称
        """
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
    
    def add_alert_callback(self, callback: Callable):
        """
        添加告警回调函数
        
        Args:
            callback: 回调函数，接收 (AlertRule, PerformanceMetric) 参数
        """
        self.alert_callbacks.append(callback)
    
    def _check_alert_rules(self, metric: PerformanceMetric):
        """
        检查告警规则
        
        Args:
            metric: 性能指标
        """
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            if rule.metric_name != metric.name:
                continue
            
            # 检查是否在冷却期内
            if rule.name in self.last_alert_times:
                elapsed = (datetime.now() - self.last_alert_times[rule.name]).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue
            
            # 检查阈值
            triggered = False
            if rule.operator == ">" and metric.value > rule.threshold:
                triggered = True
            elif rule.operator == "<" and metric.value < rule.threshold:
                triggered = True
            elif rule.operator == ">=" and metric.value >= rule.threshold:
                triggered = True
            elif rule.operator == "<=" and metric.value <= rule.threshold:
                triggered = True
            elif rule.operator == "==" and metric.value == rule.threshold:
                triggered = True
            
            if triggered:
                self._trigger_alert(rule, metric)
                self.last_alert_times[rule.name] = datetime.now()
    
    def _trigger_alert(self, rule: AlertRule, metric: PerformanceMetric):
        """
        触发告警
        
        Args:
            rule: 触发的规则
            metric: 触发告警的指标
        """
        alert_data = {
            "rule_name": rule.name,
            "level": rule.level.value,
            "metric_name": metric.name,
            "value": metric.value,
            "threshold": rule.threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        # 打印告警信息
        level_str = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        
        print(f"{level_str[rule.level]} [告警] {rule.name}")
        print(f"   指标: {metric.name} = {metric.value:.2f}{metric.unit} (阈值: {rule.threshold})")
        
        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(rule, metric)
            except Exception as e:
                print(f"⚠️ 告警回调失败: {e}")
        
        # 记录到 LangSmith
        if self.tracing_enabled and self.api_key:
            self.log_custom_event(
                name="alert_triggered",
                inputs=alert_data,
                outputs={},
                metadata={"level": rule.level.value}
            )
    
    # ==================== 统计和报告功能 ====================
    
    def get_query_history(self, limit: int = 100) -> List[Dict]:
        """
        获取查询历史
        
        Args:
            limit: 返回的最大记录数
            
        Returns:
            List: 查询历史记录
        """
        return self.query_history[-limit:]
    
    def export_metrics(self, format: str = "json") -> str:
        """
        导出指标数据
        
        Args:
            format: 导出格式 ("json" 或 "csv")
            
        Returns:
            str: 格式化后的指标数据
        """
        if format == "json":
            return json.dumps([{
                "name": m.name,
                "value": m.value,
                "unit": m.unit,
                "timestamp": m.timestamp.isoformat(),
                "tags": m.tags
            } for m in self.metrics], indent=2, ensure_ascii=False)
        
        elif format == "csv":
            lines = ["name,value,unit,timestamp,tags"]
            for m in self.metrics:
                tags_str = "|".join([f"{k}={v}" for k, v in m.tags.items()])
                lines.append(f"{m.name},{m.value},{m.unit},{m.timestamp.isoformat()},{tags_str}")
            return "\n".join(lines)
        
        return ""
    
    def print_performance_summary(self):
        """打印性能摘要"""
        report = self.get_performance_report()
        
        print("\n" + "=" * 60)
        print("📊 LangSmith 性能监控摘要")
        print("=" * 60)
        
        print(f"总查询数: {report['total_queries']}")
        print(f"总指标数: {report['total_metrics']}")
        print(f"触发告警数: {report['alerts_triggered']}")
        
        if report["query_stats"]:
            stats = report["query_stats"]
            print(f"\n查询延迟统计:")
            print(f"  平均延迟: {stats['avg_latency_ms']:.2f}ms")
            print(f"  最小延迟: {stats['min_latency_ms']:.2f}ms")
            print(f"  最大延迟: {stats['max_latency_ms']:.2f}ms")
            print(f"  P50延迟: {stats['p50_latency_ms']:.2f}ms")
            print(f"  P95延迟: {stats['p95_latency_ms']:.2f}ms")
        
        print("\n关键指标统计:")
        for metric_name, stats in report["metrics_summary"].items():
            if stats["count"] > 0:
                print(f"  {metric_name}:")
                print(f"    平均值: {stats['avg']:.2f}")
                print(f"    最小值: {stats['min']:.2f}")
                print(f"    最大值: {stats['max']:.2f}")
                print(f"    P95: {stats['p95']:.2f}")
        
        print("=" * 60)


def setup_langsmith(
    project_name: str = "adaptive-rag-project",
    enable_tracing: bool = None,
    enable_performance_monitoring: bool = True,
    enable_alerts: bool = True
) -> LangSmithManager:
    """
    设置 LangSmith 集成
    
    Args:
        project_name: 项目名称
        enable_tracing: 是否启用追踪，如果为 None 则从环境变量读取
        enable_performance_monitoring: 是否启用性能监控
        enable_alerts: 是否启用告警系统
    
    Returns:
        LangSmithManager 实例
    """
    # 从环境变量读取配置
    if enable_tracing is None:
        enable_tracing = os.environ.get("LANGSMITH_TRACING", "true").lower() == "true"
    
    api_key = os.environ.get("LANGSMITH_API_KEY")
    
    # 如果没有 API 密钥且追踪已启用，发出警告
    if enable_tracing and not api_key:
        print("⚠️ LANGSMITH_TRACING=true 但未设置 LANGSMITH_API_KEY")
        print("   请在 .env 文件中设置 LANGSMITH_API_KEY")
        enable_tracing = False
    
    manager = LangSmithManager(
        project_name=project_name,
        api_key=api_key,
        tracing_enabled=enable_tracing,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_alerts=enable_alerts
    )
    
    return manager