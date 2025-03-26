"""
Analytics and monitoring system for tracking system performance and usage.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
import psutil
import numpy as np

from app.utils.logger import setup_logger
from app.config.settings import config

logger = setup_logger(__name__, "logs/analytics.log")

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, float]

@dataclass
class QueryMetrics:
    """Query performance metrics."""
    timestamp: datetime
    query: str
    response_time: float
    num_results: int
    avg_similarity: float
    error: Optional[str] = None

class AnalyticsMonitor:
    """System for tracking analytics and monitoring performance."""
    
    def __init__(self, metrics_dir: str = "metrics"):
        """
        Initialize the analytics monitor.
        
        Args:
            metrics_dir: Directory to store metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.system_metrics: List[SystemMetrics] = []
        self.query_metrics: List[QueryMetrics] = []
        
        logger.info("Initialized AnalyticsMonitor")
    
    def record_system_metrics(self, operation: str, metadata: Dict[str, Any] = None):
        """
        Record current system metrics.
        
        Args:
            operation: The operation being performed (e.g., 'indexing', 'querying')
            metadata: Optional metadata about the operation
        """
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        # Create metrics object
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            network_io={
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv
            }
        )
        
        # Store metrics with operation and metadata
        self.system_metrics.append(metrics)
        
        # Save to file
        self._save_system_metrics()
        
        logger.debug(f"Recorded system metrics for {operation}: CPU={cpu_percent}%, Memory={memory.percent}%")
    
    def record_query_metrics(
        self,
        query: str,
        response_time: float,
        num_results: int,
        avg_similarity: float,
        error: Optional[str] = None
    ):
        """
        Record metrics for a query.
        
        Args:
            query: The query string
            response_time: Time taken to process query
            num_results: Number of results returned
            avg_similarity: Average similarity score
            error: Optional error message
        """
        # Create metrics object
        metrics = QueryMetrics(
            timestamp=datetime.now(),
            query=query,
            response_time=response_time,
            num_results=num_results,
            avg_similarity=avg_similarity,
            error=error
        )
        
        # Store metrics
        self.query_metrics.append(metrics)
        
        # Save to file
        self._save_query_metrics()
        
        logger.debug(f"Recorded query metrics: response_time={response_time}s, results={num_results}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        if not self.system_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in self.system_metrics]
        memory_values = [m.memory_percent for m in self.system_metrics]
        disk_values = [m.disk_usage_percent for m in self.system_metrics]
        
        stats = {
            "cpu": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values)
            },
            "memory": {
                "mean": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values)
            },
            "disk": {
                "mean": np.mean(disk_values),
                "max": np.max(disk_values),
                "min": np.min(disk_values)
            },
            "total_metrics": len(self.system_metrics),
            "time_range": {
                "start": self.system_metrics[0].timestamp,
                "end": self.system_metrics[-1].timestamp
            }
        }
        
        logger.info("Generated system statistics")
        return stats
    
    def get_query_stats(self) -> Dict[str, Any]:
        """
        Get query statistics.
        
        Returns:
            Dict[str, Any]: Query statistics
        """
        if not self.query_metrics:
            return {}
        
        # Calculate statistics
        response_times = [m.response_time for m in self.query_metrics]
        num_results = [m.num_results for m in self.query_metrics]
        similarities = [m.avg_similarity for m in self.query_metrics]
        errors = [m.error for m in self.query_metrics if m.error]
        
        stats = {
            "response_time": {
                "mean": np.mean(response_times),
                "max": np.max(response_times),
                "min": np.min(response_times)
            },
            "results": {
                "mean": np.mean(num_results),
                "max": np.max(num_results),
                "min": np.min(num_results)
            },
            "similarity": {
                "mean": np.mean(similarities),
                "max": np.max(similarities),
                "min": np.min(similarities)
            },
            "errors": {
                "count": len(errors),
                "rate": len(errors) / len(self.query_metrics)
            },
            "total_queries": len(self.query_metrics),
            "time_range": {
                "start": self.query_metrics[0].timestamp,
                "end": self.query_metrics[-1].timestamp
            }
        }
        
        logger.info("Generated query statistics")
        return stats
    
    def _save_system_metrics(self):
        """Save system metrics to file."""
        file_path = self.metrics_dir / "system_metrics.json"
        metrics_data = [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "disk_usage_percent": m.disk_usage_percent,
                "network_io": m.network_io
            }
            for m in self.system_metrics
        ]
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f)
    
    def _save_query_metrics(self):
        """Save query metrics to file."""
        file_path = self.metrics_dir / "query_metrics.json"
        metrics_data = [
            {
                "timestamp": m.timestamp.isoformat(),
                "query": m.query,
                "response_time": m.response_time,
                "num_results": m.num_results,
                "avg_similarity": m.avg_similarity,
                "error": m.error
            }
            for m in self.query_metrics
        ]
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f)
    
    def load_metrics(self):
        """Load metrics from files."""
        # Load system metrics
        system_file = self.metrics_dir / "system_metrics.json"
        if system_file.exists():
            with open(system_file, 'r') as f:
                data = json.load(f)
                self.system_metrics = [
                    SystemMetrics(
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        cpu_percent=m["cpu_percent"],
                        memory_percent=m["memory_percent"],
                        disk_usage_percent=m["disk_usage_percent"],
                        network_io=m["network_io"]
                    )
                    for m in data
                ]
        
        # Load query metrics
        query_file = self.metrics_dir / "query_metrics.json"
        if query_file.exists():
            with open(query_file, 'r') as f:
                data = json.load(f)
                self.query_metrics = [
                    QueryMetrics(
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        query=m["query"],
                        response_time=m["response_time"],
                        num_results=m["num_results"],
                        avg_similarity=m["avg_similarity"],
                        error=m.get("error")
                    )
                    for m in data
                ]
        
        logger.info("Loaded metrics from files") 