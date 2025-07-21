"""
Supervisor agent for system monitoring and health checks.
Implements performance metrics collection, health monitoring, and exception handling.
"""
import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from pydantic import BaseModel, Field

from config.settings import settings
from agents.database import database_agent
from agents.web_scraper import web_scraper_agent
from agents.verification import verification_agent
from agents.extractor import extraction_agent
from agents.summarizer import summarizer_agent
from agents.connector import connection_agent
from agents.writer import writer_agent
from agents.quality_assurance import quality_assurance_agent
from agents.telegram_bot import telegram_bot_agent


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AgentStatus(BaseModel):
    """Status information for an individual agent."""
    
    agent_name: str = Field(..., description="Name of the agent")
    status: HealthStatus = Field(..., description="Current health status")
    last_check: datetime = Field(default_factory=datetime.now, description="Last health check time")
    
    # Performance metrics
    response_time: Optional[float] = Field(None, description="Last response time in seconds")
    error_count: int = Field(0, description="Number of errors in current period")
    success_count: int = Field(0, description="Number of successful operations")
    
    # Status details
    status_message: str = Field("", description="Human-readable status message")
    last_error: Optional[str] = Field(None, description="Last error message")
    uptime: Optional[float] = Field(None, description="Agent uptime in hours")
    
    # Metrics
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")


class SystemHealthReport(BaseModel):
    """Comprehensive system health report."""
    
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique report ID")
    generated_at: datetime = Field(default_factory=datetime.now, description="Report generation time")
    
    # Overall system status
    overall_status: HealthStatus = Field(..., description="Overall system health")
    system_uptime: float = Field(..., description="System uptime in hours")
    
    # Agent statuses
    agent_statuses: List[AgentStatus] = Field(default_factory=list, description="Individual agent statuses")
    
    # System metrics
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="System-wide metrics")
    database_metrics: Dict[str, Any] = Field(default_factory=dict, description="Database metrics")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    # Issues and recommendations
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues found")
    warnings: List[str] = Field(default_factory=list, description="Warning conditions")
    recommendations: List[str] = Field(default_factory=list, description="System recommendations")
    
    def get_healthy_agent_count(self) -> int:
        """Get number of healthy agents."""
        return sum(1 for agent in self.agent_statuses if agent.status == HealthStatus.HEALTHY)
    
    def get_critical_agent_count(self) -> int:
        """Get number of agents in critical state."""
        return sum(1 for agent in self.agent_statuses if agent.status == HealthStatus.CRITICAL)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    
    start_time: datetime = field(default_factory=datetime.now)
    
    # Operation counters
    articles_scraped: int = 0
    articles_processed: int = 0
    articles_generated: int = 0
    articles_published: int = 0
    
    # Error counters
    scraping_errors: int = 0
    processing_errors: int = 0
    generation_errors: int = 0
    distribution_errors: int = 0
    
    # Performance metrics
    avg_scraping_time: float = 0.0
    avg_processing_time: float = 0.0
    avg_generation_time: float = 0.0
    
    # Resource usage
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    
    def get_uptime_hours(self) -> float:
        """Get system uptime in hours."""
        return (datetime.now() - self.start_time).total_seconds() / 3600
    
    def get_total_operations(self) -> int:
        """Get total operations performed."""
        return (self.articles_scraped + self.articles_processed + 
                self.articles_generated + self.articles_published)
    
    def get_total_errors(self) -> int:
        """Get total error count."""
        return (self.scraping_errors + self.processing_errors + 
                self.generation_errors + self.distribution_errors)
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        total_ops = self.get_total_operations()
        if total_ops == 0:
            return 1.0
        return (total_ops - self.get_total_errors()) / total_ops


class HealthChecker:
    """Performs health checks on individual agents."""
    
    def __init__(self):
        self.check_timeout = 10.0  # seconds
    
    async def check_database_agent(self) -> AgentStatus:
        """Check database agent health."""
        start_time = time.time()
        status = AgentStatus(agent_name="Database Agent")
        
        try:
            # Test database connectivity and get stats
            db_stats = await asyncio.wait_for(
                database_agent.get_database_stats(),
                timeout=self.check_timeout
            )
            
            response_time = time.time() - start_time
            
            if db_stats:
                status.status = HealthStatus.HEALTHY
                status.status_message = f"Database operational. {db_stats.get('total_articles', 0)} articles stored."
                status.response_time = response_time
                status.success_count = 1
            else:
                status.status = HealthStatus.WARNING
                status.status_message = "Database responded but returned no stats"
                status.response_time = response_time
                
        except asyncio.TimeoutError:
            status.status = HealthStatus.CRITICAL
            status.status_message = f"Database health check timed out after {self.check_timeout}s"
            status.error_count = 1
            status.last_error = "Timeout"
        except Exception as e:
            status.status = HealthStatus.CRITICAL
            status.status_message = f"Database health check failed: {str(e)}"
            status.error_count = 1
            status.last_error = str(e)
        
        return status
    
    async def check_web_scraper_agent(self) -> AgentStatus:
        """Check web scraper agent health."""
        start_time = time.time()
        status = AgentStatus(agent_name="Web Scraper Agent")
        
        try:
            # Check scraper status by getting batch statuses
            batch_statuses = web_scraper_agent.get_all_batch_statuses()
            
            response_time = time.time() - start_time
            status.response_time = response_time
            
            if len(batch_statuses) == 0:
                status.status = HealthStatus.HEALTHY
                status.status_message = "Web scraper idle, ready for operations"
                status.success_count = 1
            else:
                # Check if any batches are running
                running_batches = [b for b in batch_statuses if not b.is_complete()]
                if running_batches:
                    status.status = HealthStatus.HEALTHY
                    status.status_message = f"Web scraper active: {len(running_batches)} batches running"
                else:
                    status.status = HealthStatus.HEALTHY
                    status.status_message = f"Web scraper completed {len(batch_statuses)} recent batches"
                status.success_count = 1
                
        except Exception as e:
            status.status = HealthStatus.WARNING
            status.status_message = f"Web scraper check failed: {str(e)}"
            status.error_count = 1
            status.last_error = str(e)
        
        return status
    
    async def check_telegram_bot_agent(self) -> AgentStatus:
        """Check Telegram bot agent health."""
        start_time = time.time()
        status = AgentStatus(agent_name="Telegram Bot Agent")
        
        try:
            # Get delivery statistics
            delivery_stats = await asyncio.wait_for(
                telegram_bot_agent.get_delivery_statistics(),
                timeout=self.check_timeout
            )
            
            response_time = time.time() - start_time
            status.response_time = response_time
            
            if delivery_stats:
                success_rate = delivery_stats.get('success_rate', 0.0)
                total_messages = delivery_stats.get('total_messages', 0)
                
                if success_rate >= 0.9:
                    status.status = HealthStatus.HEALTHY
                elif success_rate >= 0.7:
                    status.status = HealthStatus.WARNING
                else:
                    status.status = HealthStatus.CRITICAL
                
                status.status_message = f"Telegram bot: {total_messages} messages, {success_rate:.1%} success rate"
                status.success_count = delivery_stats.get('successful_deliveries', 0)
                status.error_count = delivery_stats.get('failed_deliveries', 0)
            else:
                status.status = HealthStatus.WARNING
                status.status_message = "Telegram bot responded but no statistics available"
                
        except asyncio.TimeoutError:
            status.status = HealthStatus.CRITICAL
            status.status_message = f"Telegram bot check timed out after {self.check_timeout}s"
            status.error_count = 1
            status.last_error = "Timeout"
        except Exception as e:
            status.status = HealthStatus.WARNING
            status.status_message = f"Telegram bot check failed: {str(e)}"
            status.error_count = 1
            status.last_error = str(e)
        
        return status
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            
            # Get network stats (if available)
            try:
                network = psutil.net_io_counters()
                network_stats = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            except Exception:
                network_stats = {}
            
            return {
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent
                },
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'percent': (disk.used / disk.total) * 100
                },
                'network': network_stats
            }
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {}


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.alert_thresholds = {
            'memory_usage_percent': 85.0,
            'cpu_usage_percent': 80.0,
            'disk_usage_percent': 90.0,
            'error_rate_threshold': 0.1,
            'response_time_threshold': 30.0  # seconds
        }
        self.alert_history: List[Dict[str, Any]] = []
    
    def check_alerts(self, health_report: SystemHealthReport) -> List[Dict[str, Any]]:
        """Check for alert conditions and generate alerts."""
        alerts = []
        
        # Check agent health
        critical_agents = [agent for agent in health_report.agent_statuses 
                          if agent.status == HealthStatus.CRITICAL]
        if critical_agents:
            alerts.append({
                'level': 'critical',
                'type': 'agent_health',
                'message': f"{len(critical_agents)} agents in critical state",
                'affected_agents': [agent.agent_name for agent in critical_agents],
                'timestamp': datetime.now()
            })
        
        # Check system resources
        system_metrics = health_report.system_metrics
        if system_metrics:
            memory_usage = system_metrics.get('memory', {}).get('percent', 0)
            if memory_usage > self.alert_thresholds['memory_usage_percent']:
                alerts.append({
                    'level': 'warning',
                    'type': 'high_memory_usage',
                    'message': f"High memory usage: {memory_usage:.1f}%",
                    'value': memory_usage,
                    'threshold': self.alert_thresholds['memory_usage_percent'],
                    'timestamp': datetime.now()
                })
            
            cpu_usage = system_metrics.get('cpu', {}).get('percent', 0)
            if cpu_usage > self.alert_thresholds['cpu_usage_percent']:
                alerts.append({
                    'level': 'warning',
                    'type': 'high_cpu_usage',
                    'message': f"High CPU usage: {cpu_usage:.1f}%",
                    'value': cpu_usage,
                    'threshold': self.alert_thresholds['cpu_usage_percent'],
                    'timestamp': datetime.now()
                })
            
            disk_usage = system_metrics.get('disk', {}).get('percent', 0)
            if disk_usage > self.alert_thresholds['disk_usage_percent']:
                alerts.append({
                    'level': 'critical',
                    'type': 'high_disk_usage',
                    'message': f"High disk usage: {disk_usage:.1f}%",
                    'value': disk_usage,
                    'threshold': self.alert_thresholds['disk_usage_percent'],
                    'timestamp': datetime.now()
                })
        
        # Check performance metrics
        performance_metrics = health_report.performance_metrics
        if performance_metrics:
            error_rate = performance_metrics.get('error_rate', 0)
            if error_rate > self.alert_thresholds['error_rate_threshold']:
                alerts.append({
                    'level': 'warning',
                    'type': 'high_error_rate',
                    'message': f"High error rate: {error_rate:.1%}",
                    'value': error_rate,
                    'threshold': self.alert_thresholds['error_rate_threshold'],
                    'timestamp': datetime.now()
                })
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > cutoff_time
        ]
        
        return alerts


class SupervisorAgent:
    """
    Supervisor agent for system monitoring and health checks.
    Implements performance metrics collection, health monitoring, and exception handling.
    """
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.performance_metrics = PerformanceMetrics()
        self.health_check_interval = 300  # 5 minutes
        self.monitoring_active = False
        self.last_health_report: Optional[SystemHealthReport] = None
    
    async def generate_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report."""
        logger.info("Generating system health report...")
        
        try:
            # Check individual agents
            agent_checks = await asyncio.gather(
                self.health_checker.check_database_agent(),
                self.health_checker.check_web_scraper_agent(),
                self.health_checker.check_telegram_bot_agent(),
                return_exceptions=True
            )
            
            # Filter out exceptions and create agent statuses
            agent_statuses = []
            for check_result in agent_checks:
                if isinstance(check_result, AgentStatus):
                    agent_statuses.append(check_result)
                else:
                    # Create error status for failed check
                    error_status = AgentStatus(
                        agent_name="Unknown Agent",
                        status=HealthStatus.CRITICAL,
                        status_message=f"Health check failed: {str(check_result)}",
                        error_count=1,
                        last_error=str(check_result)
                    )
                    agent_statuses.append(error_status)
            
            # Check system resources
            system_metrics = await self.health_checker.check_system_resources()
            
            # Get database metrics
            database_metrics = await self._get_database_metrics()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            # Determine overall system status
            overall_status = self._determine_overall_status(agent_statuses, system_metrics)
            
            # Generate issues and recommendations
            critical_issues, warnings, recommendations = self._analyze_issues(
                agent_statuses, system_metrics, performance_metrics
            )
            
            # Create health report
            health_report = SystemHealthReport(
                overall_status=overall_status,
                system_uptime=self.performance_metrics.get_uptime_hours(),
                agent_statuses=agent_statuses,
                system_metrics=system_metrics,
                database_metrics=database_metrics,
                performance_metrics=performance_metrics,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
            
            # Check for alerts
            alerts = self.alert_manager.check_alerts(health_report)
            if alerts:
                logger.warning(f"Generated {len(alerts)} system alerts")
                for alert in alerts:
                    if alert['level'] == 'critical':
                        logger.error(f"CRITICAL ALERT: {alert['message']}")
                    else:
                        logger.warning(f"WARNING ALERT: {alert['message']}")
            
            self.last_health_report = health_report
            
            logger.info(f"Health report generated: {overall_status.value} status, "
                       f"{health_report.get_healthy_agent_count()}/{len(agent_statuses)} agents healthy")
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            # Return minimal error report
            return SystemHealthReport(
                overall_status=HealthStatus.CRITICAL,
                system_uptime=0.0,
                critical_issues=[f"Health report generation failed: {str(e)}"]
            )
    
    async def _get_database_metrics(self) -> Dict[str, Any]:
        """Get database-specific metrics."""
        try:
            db_stats = await database_agent.get_database_stats()
            return {
                'total_articles': db_stats.get('total_articles', 0),
                'processed_articles': db_stats.get('processed_articles', 0),
                'total_extractions': db_stats.get('total_extractions', 0),
                'approved_articles': db_stats.get('approved_articles', 0),
                'processing_rate': (
                    db_stats.get('processed_articles', 0) / 
                    max(db_stats.get('total_articles', 1), 1)
                )
            }
        except Exception as e:
            logger.error(f"Error getting database metrics: {e}")
            return {}
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics."""
        return {
            'uptime_hours': self.performance_metrics.get_uptime_hours(),
            'total_operations': self.performance_metrics.get_total_operations(),
            'total_errors': self.performance_metrics.get_total_errors(),
            'success_rate': self.performance_metrics.get_success_rate(),
            'articles_scraped': self.performance_metrics.articles_scraped,
            'articles_processed': self.performance_metrics.articles_processed,
            'articles_generated': self.performance_metrics.articles_generated,
            'articles_published': self.performance_metrics.articles_published,
            'avg_scraping_time': self.performance_metrics.avg_scraping_time,
            'avg_processing_time': self.performance_metrics.avg_processing_time,
            'avg_generation_time': self.performance_metrics.avg_generation_time,
            'peak_memory_usage': self.performance_metrics.peak_memory_usage,
            'avg_cpu_usage': self.performance_metrics.avg_cpu_usage
        }
    
    def _determine_overall_status(
        self, 
        agent_statuses: List[AgentStatus], 
        system_metrics: Dict[str, Any]
    ) -> HealthStatus:
        """Determine overall system health status."""
        # Check for critical agents
        critical_agents = [agent for agent in agent_statuses if agent.status == HealthStatus.CRITICAL]
        if critical_agents:
            return HealthStatus.CRITICAL
        
        # Check for warning agents
        warning_agents = [agent for agent in agent_statuses if agent.status == HealthStatus.WARNING]
        
        # Check system resources
        resource_warnings = 0
        if system_metrics:
            memory_usage = system_metrics.get('memory', {}).get('percent', 0)
            cpu_usage = system_metrics.get('cpu', {}).get('percent', 0)
            disk_usage = system_metrics.get('disk', {}).get('percent', 0)
            
            if memory_usage > 85:
                resource_warnings += 1
            if cpu_usage > 80:
                resource_warnings += 1
            if disk_usage > 90:
                return HealthStatus.CRITICAL  # Disk space is critical
            elif disk_usage > 80:
                resource_warnings += 1
        
        # Determine status based on warnings
        if resource_warnings > 1 or len(warning_agents) > 2:
            return HealthStatus.WARNING
        elif len(warning_agents) > 0 or resource_warnings > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _analyze_issues(
        self,
        agent_statuses: List[AgentStatus],
        system_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> tuple[List[str], List[str], List[str]]:
        """Analyze system state and generate issues and recommendations."""
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Analyze agent issues
        for agent in agent_statuses:
            if agent.status == HealthStatus.CRITICAL:
                critical_issues.append(f"{agent.agent_name}: {agent.status_message}")
            elif agent.status == HealthStatus.WARNING:
                warnings.append(f"{agent.agent_name}: {agent.status_message}")
        
        # Analyze system resources
        if system_metrics:
            memory_usage = system_metrics.get('memory', {}).get('percent', 0)
            if memory_usage > 90:
                critical_issues.append(f"Critical memory usage: {memory_usage:.1f}%")
                recommendations.append("Consider increasing system memory or optimizing memory usage")
            elif memory_usage > 80:
                warnings.append(f"High memory usage: {memory_usage:.1f}%")
                recommendations.append("Monitor memory usage and consider optimization")
            
            disk_usage = system_metrics.get('disk', {}).get('percent', 0)
            if disk_usage > 90:
                critical_issues.append(f"Critical disk usage: {disk_usage:.1f}%")
                recommendations.append("Free up disk space immediately")
            elif disk_usage > 80:
                warnings.append(f"High disk usage: {disk_usage:.1f}%")
                recommendations.append("Clean up old logs and temporary files")
        
        # Analyze performance
        if performance_metrics:
            success_rate = performance_metrics.get('success_rate', 1.0)
            if success_rate < 0.7:
                critical_issues.append(f"Low success rate: {success_rate:.1%}")
                recommendations.append("Investigate and fix recurring errors")
            elif success_rate < 0.9:
                warnings.append(f"Reduced success rate: {success_rate:.1%}")
                recommendations.append("Monitor error patterns and consider improvements")
            
            total_errors = performance_metrics.get('total_errors', 0)
            if total_errors > 10:
                warnings.append(f"High error count: {total_errors} errors")
                recommendations.append("Review error logs and implement fixes")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System is operating normally")
        
        return critical_issues, warnings, recommendations
    
    async def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting continuous system monitoring...")
        
        try:
            while self.monitoring_active:
                # Generate health report
                health_report = await self.generate_health_report()
                
                # Log health status
                logger.info(f"System health check completed: {health_report.overall_status.value}")
                
                # Wait for next check
                await asyncio.sleep(self.health_check_interval)
                
        except asyncio.CancelledError:
            logger.info("System monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")
        finally:
            self.monitoring_active = False
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        logger.info("Stopping system monitoring...")
    
    def update_performance_metrics(self, operation_type: str, success: bool, duration: float = 0.0):
        """Update performance metrics for an operation."""
        if operation_type == "scraping":
            if success:
                self.performance_metrics.articles_scraped += 1
                if duration > 0:
                    # Update average scraping time
                    current_avg = self.performance_metrics.avg_scraping_time
                    count = self.performance_metrics.articles_scraped
                    self.performance_metrics.avg_scraping_time = (
                        (current_avg * (count - 1) + duration) / count
                    )
            else:
                self.performance_metrics.scraping_errors += 1
        
        elif operation_type == "processing":
            if success:
                self.performance_metrics.articles_processed += 1
                if duration > 0:
                    current_avg = self.performance_metrics.avg_processing_time
                    count = self.performance_metrics.articles_processed
                    self.performance_metrics.avg_processing_time = (
                        (current_avg * (count - 1) + duration) / count
                    )
            else:
                self.performance_metrics.processing_errors += 1
        
        elif operation_type == "generation":
            if success:
                self.performance_metrics.articles_generated += 1
                if duration > 0:
                    current_avg = self.performance_metrics.avg_generation_time
                    count = self.performance_metrics.articles_generated
                    self.performance_metrics.avg_generation_time = (
                        (current_avg * (count - 1) + duration) / count
                    )
            else:
                self.performance_metrics.generation_errors += 1
        
        elif operation_type == "distribution":
            if success:
                self.performance_metrics.articles_published += 1
            else:
                self.performance_metrics.distribution_errors += 1
        
        # Update resource usage
        try:
            memory = psutil.virtual_memory()
            current_memory_mb = memory.used / (1024 * 1024)
            if current_memory_mb > self.performance_metrics.peak_memory_usage:
                self.performance_metrics.peak_memory_usage = current_memory_mb
            
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 0:
                # Update average CPU usage
                current_avg = self.performance_metrics.avg_cpu_usage
                total_ops = self.performance_metrics.get_total_operations()
                if total_ops > 0:
                    self.performance_metrics.avg_cpu_usage = (
                        (current_avg * (total_ops - 1) + cpu_percent) / total_ops
                    )
                else:
                    self.performance_metrics.avg_cpu_usage = cpu_percent
        except Exception:
            pass  # Ignore resource monitoring errors
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of monitoring data."""
        return {
            'monitoring_active': self.monitoring_active,
            'last_health_check': self.last_health_report.generated_at if self.last_health_report else None,
            'system_uptime_hours': self.performance_metrics.get_uptime_hours(),
            'total_operations': self.performance_metrics.get_total_operations(),
            'success_rate': self.performance_metrics.get_success_rate(),
            'recent_alerts': len(self.alert_manager.alert_history),
            'overall_status': self.last_health_report.overall_status if self.last_health_report else HealthStatus.UNKNOWN
        }


# Global supervisor agent instance
supervisor_agent = SupervisorAgent()