"""
Scheduler agent for coordinating automated 4-hour cycles and task management.
Implements APScheduler integration with persistent scheduling and agent coordination.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
import uuid
import json

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
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
from agents.supervisor import supervisor_agent


logger = logging.getLogger(__name__)


class ScheduledTask(BaseModel):
    """Represents a scheduled task."""

    task_id: str = Field(..., description="Unique task identifier")
    task_name: str = Field(..., description="Human-readable task name")
    task_type: str = Field(
        ..., description="Type of task (workflow, maintenance, etc.)"
    )

    # Scheduling
    schedule_type: str = Field(..., description="Schedule type: interval, cron, once")
    schedule_config: Dict[str, Any] = Field(..., description="Schedule configuration")

    # Execution tracking
    created_at: datetime = Field(
        default_factory=datetime.now, description="When task was created"
    )
    last_run: Optional[datetime] = Field(None, description="Last execution time")
    next_run: Optional[datetime] = Field(None, description="Next scheduled execution")
    run_count: int = Field(0, description="Number of times task has run")

    # Status
    enabled: bool = Field(True, description="Whether task is enabled")
    status: str = Field("scheduled", description="Task status")
    last_result: Optional[str] = Field(None, description="Result of last execution")

    # Configuration
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout_seconds: int = Field(3600, description="Task timeout in seconds")
    retry_delay: int = Field(300, description="Delay between retries in seconds")


class WorkflowExecution(BaseModel):
    """Represents a workflow execution instance."""

    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID"
    )
    workflow_type: str = Field(..., description="Type of workflow")

    # Timing
    started_at: datetime = Field(
        default_factory=datetime.now, description="Execution start time"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Execution completion time"
    )

    # Progress tracking
    total_steps: int = Field(0, description="Total number of steps")
    completed_steps: int = Field(0, description="Number of completed steps")
    current_step: str = Field("", description="Current step being executed")

    # Results
    success: bool = Field(False, description="Whether execution was successful")
    results: Dict[str, Any] = Field(
        default_factory=dict, description="Execution results"
    )
    errors: List[str] = Field(default_factory=list, description="Errors encountered")

    # Metrics
    articles_scraped: int = Field(0, description="Articles scraped in this execution")
    articles_processed: int = Field(0, description="Articles processed")
    articles_generated: int = Field(0, description="Articles generated")
    articles_published: int = Field(0, description="Articles published")

    def duration_seconds(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def progress_percentage(self) -> float:
        """Get execution progress as percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100


class WorkflowManager:
    """Manages workflow execution and coordination between agents."""

    def __init__(self):
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        self.max_history_size = 50

    async def execute_full_research_workflow(
        self, max_articles_per_source: int = 10, execution_id: Optional[str] = None
    ) -> WorkflowExecution:
        """Execute the complete legal research workflow."""
        if execution_id is None:
            execution_id = str(uuid.uuid4())

        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_type="full_research_cycle",
            total_steps=7,
        )

        self.active_executions[execution_id] = execution

        try:
            logger.info(f"Starting full research workflow: {execution_id}")

            # Step 1: Web Scraping
            execution.current_step = "web_scraping"
            execution.completed_steps = 1
            supervisor_agent.update_performance_metrics("scraping", True)

            logger.info("Step 1/7: Web scraping legal news sources...")
            scraping_start = datetime.now()

            scraping_batches = await web_scraper_agent.scrape_all_sources(
                max_articles_per_source
            )
            execution.articles_scraped = sum(
                batch.successful_urls for batch in scraping_batches
            )

            scraping_duration = (datetime.now() - scraping_start).total_seconds()
            supervisor_agent.update_performance_metrics(
                "scraping", True, scraping_duration
            )

            execution.results["scraping"] = {
                "batches": len(scraping_batches),
                "articles_scraped": execution.articles_scraped,
                "duration_seconds": scraping_duration,
            }

            if execution.articles_scraped == 0:
                raise Exception("No articles were successfully scraped")

            # Step 2: Data Extraction
            execution.current_step = "data_extraction"
            execution.completed_steps = 2

            logger.info("Step 2/7: Extracting structured data from articles...")
            extraction_start = datetime.now()

            processed_ids = await extraction_agent.process_unprocessed_articles(
                limit=execution.articles_scraped
            )
            execution.articles_processed = len(processed_ids)

            extraction_duration = (datetime.now() - extraction_start).total_seconds()
            supervisor_agent.update_performance_metrics(
                "processing", True, extraction_duration
            )

            execution.results["extraction"] = {
                "articles_processed": execution.articles_processed,
                "duration_seconds": extraction_duration,
            }

            if execution.articles_processed == 0:
                raise Exception("No articles were successfully processed")

            # Step 3: Summarization
            execution.current_step = "summarization"
            execution.completed_steps = 3

            logger.info("Step 3/7: Generating article summaries...")
            summaries = await summarizer_agent.batch_summarize_articles(
                processed_ids[:5]
            )

            execution.results["summarization"] = {
                "summaries_generated": len(summaries),
                "articles_summarized": len(summaries),
            }

            if len(summaries) == 0:
                raise Exception("No summaries were generated")

            # Step 4: Connection Analysis
            execution.current_step = "connection_analysis"
            execution.completed_steps = 4

            logger.info("Step 4/7: Analyzing connections between articles...")
            article_ids = [summary.article_id for summary in summaries]
            connections = await connection_agent.analyze_article_connections(
                article_ids
            )

            execution.results["connections"] = {
                "connections_found": connections.connection_count,
                "clusters_identified": connections.cluster_count,
                "trends_detected": connections.trend_count,
            }

            # Step 5: Article Generation
            execution.current_step = "article_generation"
            execution.completed_steps = 5

            logger.info("Step 5/7: Generating comprehensive legal blog article...")
            generation_start = datetime.now()

            # Use top summaries for article generation
            top_summaries = summaries[:3]
            generated_article = await writer_agent.write_article(
                summaries=top_summaries,
                connections=connections,
                trends=connections.identified_trends,
            )

            generation_duration = (datetime.now() - generation_start).total_seconds()
            supervisor_agent.update_performance_metrics(
                "generation", True, generation_duration
            )

            execution.articles_generated = 1
            execution.results["generation"] = {
                "articles_generated": 1,
                "word_count": generated_article.word_count,
                "quality_score": generated_article.quality_score,
                "duration_seconds": generation_duration,
            }

            # Step 6: Quality Assurance
            execution.current_step = "quality_assurance"
            execution.completed_steps = 6

            logger.info("Step 6/7: Quality assurance validation...")
            validation_result = await quality_assurance_agent.validate_article(
                generated_article
            )

            execution.results["quality_assurance"] = {
                "validation_passed": validation_result.passed,
                "overall_score": validation_result.overall_score,
                "originality_score": validation_result.originality_score,
                "critical_issues": len(validation_result.critical_issues),
            }

            # Step 7: Distribution (if validation passed)
            execution.current_step = "distribution"
            execution.completed_steps = 7

            if validation_result.passed:
                logger.info("Step 7/7: Distributing article via Telegram...")

                try:
                    distribution_result = await telegram_bot_agent.distribute_article(
                        generated_article, validation_result
                    )

                    if distribution_result.success:
                        execution.articles_published = 1
                        supervisor_agent.update_performance_metrics(
                            "distribution", True
                        )
                    else:
                        supervisor_agent.update_performance_metrics(
                            "distribution", False
                        )

                    execution.results["distribution"] = {
                        "distribution_success": distribution_result.success,
                        "telegram_message_id": distribution_result.telegram_message_id,
                        "error_message": distribution_result.error_message,
                    }

                except Exception as e:
                    logger.error(f"Distribution failed: {e}")
                    execution.errors.append(f"Distribution failed: {str(e)}")
                    execution.results["distribution"] = {
                        "distribution_success": False,
                        "error_message": str(e),
                    }
            else:
                logger.warning(
                    "Article failed quality validation, skipping distribution"
                )
                execution.results["distribution"] = {
                    "distribution_success": False,
                    "error_message": "Article failed quality validation",
                }

            # Mark execution as successful
            execution.success = True
            execution.completed_at = datetime.now()

            logger.info(
                f"Full research workflow completed successfully: {execution_id}"
            )
            logger.info(
                f"Results: {execution.articles_scraped} scraped, "
                f"{execution.articles_processed} processed, "
                f"{execution.articles_generated} generated, "
                f"{execution.articles_published} published"
            )

            return execution

        except Exception as e:
            execution.success = False
            execution.completed_at = datetime.now()
            execution.errors.append(str(e))

            logger.error(f"Full research workflow failed: {e}")
            return execution

        finally:
            # Move execution to history
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]

            self.execution_history.append(execution)

            # Limit history size
            if len(self.execution_history) > self.max_history_size:
                self.execution_history = self.execution_history[
                    -self.max_history_size :
                ]

    async def execute_maintenance_workflow(self) -> WorkflowExecution:
        """Execute maintenance tasks."""
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id, workflow_type="maintenance", total_steps=3
        )

        self.active_executions[execution_id] = execution

        try:
            logger.info(f"Starting maintenance workflow: {execution_id}")

            # Step 1: System health check
            execution.current_step = "health_check"
            execution.completed_steps = 1

            health_report = await supervisor_agent.generate_health_report()
            execution.results["health_check"] = {
                "overall_status": health_report.overall_status.value,
                "healthy_agents": health_report.get_healthy_agent_count(),
                "critical_issues": len(health_report.critical_issues),
            }

            # Step 2: Database cleanup
            execution.current_step = "database_cleanup"
            execution.completed_steps = 2

            # This would implement database cleanup tasks
            # For now, just get database stats
            db_stats = await database_agent.get_database_stats()
            execution.results["database_cleanup"] = {
                "total_articles": db_stats.get("total_articles", 0),
                "cleanup_performed": False,  # Placeholder
            }

            # Step 3: Log rotation and cleanup
            execution.current_step = "log_cleanup"
            execution.completed_steps = 3

            # This would implement log cleanup
            execution.results["log_cleanup"] = {"logs_cleaned": False}  # Placeholder

            execution.success = True
            execution.completed_at = datetime.now()

            logger.info(f"Maintenance workflow completed: {execution_id}")
            return execution

        except Exception as e:
            execution.success = False
            execution.completed_at = datetime.now()
            execution.errors.append(str(e))

            logger.error(f"Maintenance workflow failed: {e}")
            return execution

        finally:
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            self.execution_history.append(execution)

    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get status of a workflow execution."""
        return self.active_executions.get(execution_id)

    def get_recent_executions(self, limit: int = 10) -> List[WorkflowExecution]:
        """Get recent workflow executions."""
        return self.execution_history[-limit:]


class SchedulerAgent:
    """
    Scheduler agent for coordinating automated cycles and task management.
    Implements APScheduler integration with persistent scheduling and workflow coordination.
    """

    def __init__(self):
        # Configure APScheduler
        jobstores = {"default": MemoryJobStore()}
        executors = {"default": AsyncIOExecutor()}
        job_defaults = {"coalesce": False, "max_instances": 1}

        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone="UTC",
        )

        self.workflow_manager = WorkflowManager()
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.is_running = False

        # Set up event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        self.scheduler.add_listener(self._job_missed, EVENT_JOB_MISSED)

    async def start_scheduler(self):
        """Start the scheduler and set up default tasks."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        logger.info("Starting scheduler agent...")

        try:
            self.scheduler.start()
            self.is_running = True

            # Add default scheduled tasks
            await self._setup_default_tasks()

            logger.info("Scheduler started successfully")

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    async def stop_scheduler(self):
        """Stop the scheduler."""
        if not self.is_running:
            return

        logger.info("Stopping scheduler...")

        try:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("Scheduler stopped")

        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

    async def _setup_default_tasks(self):
        """Set up default scheduled tasks."""
        # Main research workflow - every 4 hours
        await self.schedule_research_workflow(
            interval_hours=settings.scraping_interval_hours, max_articles_per_source=5
        )

        # Health monitoring - every 5 minutes
        await self.schedule_health_monitoring(interval_minutes=5)

        # Maintenance tasks - daily at 2 AM UTC
        await self.schedule_maintenance_tasks(hour=2, minute=0)

        logger.info("Default scheduled tasks configured")

    async def schedule_research_workflow(
        self,
        interval_hours: int = 4,
        max_articles_per_source: int = 10,
        task_id: Optional[str] = None,
    ) -> str:
        """Schedule the main research workflow to run at regular intervals."""
        if task_id is None:
            task_id = f"research_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task = ScheduledTask(
            task_id=task_id,
            task_name="Legal Research Workflow",
            task_type="workflow",
            schedule_type="interval",
            schedule_config={
                "hours": interval_hours,
                "max_articles_per_source": max_articles_per_source,
            },
        )

        # Add job to scheduler
        job = self.scheduler.add_job(
            func=self._execute_research_workflow,
            trigger=IntervalTrigger(hours=interval_hours),
            args=[max_articles_per_source],
            id=task_id,
            name=task.task_name,
            replace_existing=True,
        )

        task.next_run = job.next_run_time
        self.scheduled_tasks[task_id] = task

        logger.info(
            f"Scheduled research workflow: every {interval_hours} hours, "
            f"max {max_articles_per_source} articles per source"
        )

        return task_id

    async def schedule_health_monitoring(
        self, interval_minutes: int = 5, task_id: Optional[str] = None
    ) -> str:
        """Schedule health monitoring checks."""
        if task_id is None:
            task_id = f"health_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task = ScheduledTask(
            task_id=task_id,
            task_name="System Health Monitoring",
            task_type="monitoring",
            schedule_type="interval",
            schedule_config={"minutes": interval_minutes},
        )

        job = self.scheduler.add_job(
            func=self._execute_health_check,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id=task_id,
            name=task.task_name,
            replace_existing=True,
        )

        task.next_run = job.next_run_time
        self.scheduled_tasks[task_id] = task

        logger.info(f"Scheduled health monitoring: every {interval_minutes} minutes")

        return task_id

    async def schedule_maintenance_tasks(
        self, hour: int = 2, minute: int = 0, task_id: Optional[str] = None
    ) -> str:
        """Schedule daily maintenance tasks."""
        if task_id is None:
            task_id = f"maintenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task = ScheduledTask(
            task_id=task_id,
            task_name="Daily Maintenance Tasks",
            task_type="maintenance",
            schedule_type="cron",
            schedule_config={"hour": hour, "minute": minute},
        )

        job = self.scheduler.add_job(
            func=self._execute_maintenance,
            trigger=CronTrigger(hour=hour, minute=minute),
            id=task_id,
            name=task.task_name,
            replace_existing=True,
        )

        task.next_run = job.next_run_time
        self.scheduled_tasks[task_id] = task

        logger.info(
            f"Scheduled maintenance tasks: daily at {hour:02d}:{minute:02d} UTC"
        )

        return task_id

    async def _execute_research_workflow(self, max_articles_per_source: int):
        """Execute the research workflow (called by scheduler)."""
        try:
            logger.info("Executing scheduled research workflow...")

            execution = await self.workflow_manager.execute_full_research_workflow(
                max_articles_per_source=max_articles_per_source
            )

            if execution.success:
                logger.info(
                    f"Scheduled research workflow completed successfully: {execution.execution_id}"
                )
            else:
                logger.error(f"Scheduled research workflow failed: {execution.errors}")

        except Exception as e:
            logger.error(f"Error in scheduled research workflow: {e}")

    async def _execute_health_check(self):
        """Execute health monitoring (called by scheduler)."""
        try:
            health_report = await supervisor_agent.generate_health_report()

            # Log critical issues
            if health_report.critical_issues:
                logger.error(
                    f"Health check found critical issues: {health_report.critical_issues}"
                )
            elif health_report.warnings:
                logger.warning(f"Health check warnings: {health_report.warnings}")
            else:
                logger.debug("Health check completed - system healthy")

        except Exception as e:
            logger.error(f"Error in scheduled health check: {e}")

    async def _execute_maintenance(self):
        """Execute maintenance tasks (called by scheduler)."""
        try:
            logger.info("Executing scheduled maintenance tasks...")

            execution = await self.workflow_manager.execute_maintenance_workflow()

            if execution.success:
                logger.info(
                    f"Scheduled maintenance completed successfully: {execution.execution_id}"
                )
            else:
                logger.error(f"Scheduled maintenance failed: {execution.errors}")

        except Exception as e:
            logger.error(f"Error in scheduled maintenance: {e}")

    def _job_executed(self, event):
        """Handle job execution events."""
        job_id = event.job_id
        if job_id in self.scheduled_tasks:
            task = self.scheduled_tasks[job_id]
            task.run_count += 1
            task.last_run = datetime.now()
            task.status = "completed"
            task.last_result = "success"

            # Update next run time
            job = self.scheduler.get_job(job_id)
            if job:
                task.next_run = job.next_run_time

        logger.debug(f"Job executed successfully: {job_id}")

    def _job_error(self, event):
        """Handle job error events."""
        job_id = event.job_id
        if job_id in self.scheduled_tasks:
            task = self.scheduled_tasks[job_id]
            task.status = "error"
            task.last_result = f"error: {str(event.exception)}"

        logger.error(f"Job execution failed: {job_id}, error: {event.exception}")

    def _job_missed(self, event):
        """Handle missed job events."""
        job_id = event.job_id
        if job_id in self.scheduled_tasks:
            task = self.scheduled_tasks[job_id]
            task.status = "missed"
            task.last_result = "missed execution"

        logger.warning(f"Job execution missed: {job_id}")

    async def execute_workflow_now(
        self, workflow_type: str = "full_research", max_articles_per_source: int = 5
    ) -> str:
        """Execute a workflow immediately (not scheduled)."""
        if workflow_type == "full_research":
            execution = await self.workflow_manager.execute_full_research_workflow(
                max_articles_per_source=max_articles_per_source
            )
        elif workflow_type == "maintenance":
            execution = await self.workflow_manager.execute_maintenance_workflow()
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        return execution.execution_id

    def get_scheduled_tasks(self) -> List[ScheduledTask]:
        """Get all scheduled tasks."""
        return list(self.scheduled_tasks.values())

    def get_active_executions(self) -> List[WorkflowExecution]:
        """Get currently active workflow executions."""
        return list(self.workflow_manager.active_executions.values())

    def get_execution_history(self, limit: int = 10) -> List[WorkflowExecution]:
        """Get recent execution history."""
        return self.workflow_manager.get_recent_executions(limit)

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        jobs = self.scheduler.get_jobs()

        return {
            "running": self.is_running,
            "total_jobs": len(jobs),
            "scheduled_tasks": len(self.scheduled_tasks),
            "active_executions": len(self.workflow_manager.active_executions),
            "execution_history_count": len(self.workflow_manager.execution_history),
            "next_jobs": [
                {"job_id": job.id, "name": job.name, "next_run": job.next_run_time}
                for job in jobs[:5]  # Next 5 jobs
            ],
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        try:
            self.scheduler.remove_job(task_id)
            if task_id in self.scheduled_tasks:
                self.scheduled_tasks[task_id].enabled = False
                self.scheduled_tasks[task_id].status = "cancelled"

            logger.info(f"Cancelled scheduled task: {task_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False


# Global scheduler agent instance
scheduler_agent = SchedulerAgent()
