"""
Integration tests for the complete legal research workflow.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from main import LegalResearchOrchestrator
from agents.scheduler import SchedulerAgent, WorkflowManager
from agents.supervisor import SupervisorAgent


class TestLegalResearchOrchestrator:
    """Test the main orchestrator integration."""
    
    @pytest.fixture
    async def orchestrator(self, mock_settings):
        """Create a LegalResearchOrchestrator instance."""
        with patch('main.settings', mock_settings):
            orchestrator = LegalResearchOrchestrator()
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_initialize_system_success(self, orchestrator):
        """Test successful system initialization."""
        # Mock database initialization
        with patch('main.database_agent.initialize') as mock_db_init:
            mock_db_init.return_value = None
            
            # Mock LLM test
            with patch('main.llm_client.generate') as mock_llm_test:
                mock_llm_test.return_value = MagicMock(success=True)
                
                result = await orchestrator.initialize_system()
                
                assert result is True
                assert orchestrator.system_initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_system_db_failure(self, orchestrator):
        """Test system initialization with database failure."""
        # Mock database initialization to fail
        with patch('main.database_agent.initialize') as mock_db_init:
            mock_db_init.side_effect = Exception("Database connection failed")
            
            result = await orchestrator.initialize_system()
            
            assert result is False
            assert orchestrator.system_initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_system_llm_failure(self, orchestrator):
        """Test system initialization with LLM failure."""
        # Mock database initialization success
        with patch('main.database_agent.initialize') as mock_db_init:
            mock_db_init.return_value = None
            
            # Mock LLM test failure
            with patch('main.llm_client.generate') as mock_llm_test:
                mock_llm_test.return_value = MagicMock(success=False, error="API key invalid")
                
                result = await orchestrator.initialize_system()
                
                assert result is False
    
    @pytest.mark.asyncio
    async def test_run_complete_workflow_success(self, orchestrator):
        """Test successful complete workflow execution."""
        orchestrator.system_initialized = True
        
        # Mock all the agent operations
        with patch('main.web_scraper_agent.scrape_all_sources') as mock_scraper:
            # Mock scraping results
            mock_batch = MagicMock()
            mock_batch.successful_urls = 5
            mock_scraper.return_value = [mock_batch]
            
            with patch('main.extraction_agent.process_unprocessed_articles') as mock_extractor:
                mock_extractor.return_value = ['article1', 'article2', 'article3']
                
                with patch('main.summarizer_agent.batch_summarize_articles') as mock_summarizer:
                    mock_summaries = [MagicMock(article_id=f'article{i}') for i in range(3)]
                    mock_summarizer.return_value = mock_summaries
                    
                    with patch('main.connection_agent.analyze_article_connections') as mock_connector:
                        mock_connections = MagicMock()
                        mock_connections.connection_count = 2
                        mock_connections.trend_count = 1
                        mock_connections.identified_trends = []
                        mock_connector.return_value = mock_connections
                        
                        with patch('main.writer_agent.write_article') as mock_writer:
                            mock_article = MagicMock()
                            mock_article.title = "Test Article"
                            mock_article.word_count = 550
                            mock_article.quality_score = 0.85
                            mock_writer.return_value = mock_article
                            
                            with patch('main.quality_assurance_agent.validate_article') as mock_qa:
                                mock_validation = MagicMock()
                                mock_validation.passed = True
                                mock_validation.overall_score = 8.5
                                mock_qa.return_value = mock_validation
                                
                                with patch('main.database_agent.get_database_stats') as mock_db_stats:
                                    mock_db_stats.return_value = {'total_articles': 10}
                                    
                                    results = await orchestrator.run_complete_workflow(max_articles_per_source=2)
                                    
                                    assert results['articles_scraped'] == 5
                                    assert results['articles_processed'] == 3
                                    assert results['articles_generated'] == 1
                                    assert 'validation_result' in results
                                    assert 'generated_article' in results
    
    @pytest.mark.asyncio
    async def test_run_complete_workflow_no_articles_scraped(self, orchestrator):
        """Test workflow when no articles are scraped."""
        orchestrator.system_initialized = True
        
        # Mock scraping to return no articles
        with patch('main.web_scraper_agent.scrape_all_sources') as mock_scraper:
            mock_batch = MagicMock()
            mock_batch.successful_urls = 0
            mock_scraper.return_value = [mock_batch]
            
            results = await orchestrator.run_complete_workflow()
            
            assert results['articles_scraped'] == 0
            assert 'start_time' in results
            assert 'end_time' in results
    
    @pytest.mark.asyncio
    async def test_run_scraping_only(self, orchestrator):
        """Test scraping-only workflow."""
        orchestrator.system_initialized = True
        
        # Mock scraping and extraction
        with patch('main.web_scraper_agent.scrape_all_sources') as mock_scraper:
            mock_batch = MagicMock()
            mock_batch.successful_urls = 3
            mock_scraper.return_value = [mock_batch]
            
            with patch('main.extraction_agent.process_unprocessed_articles') as mock_extractor:
                mock_extractor.return_value = ['article1', 'article2']
                
                with patch('main.verification_agent.get_verification_stats') as mock_verification:
                    mock_verification.return_value = {'verification_rate': 0.8}
                    
                    results = await orchestrator.run_scraping_only(max_articles=5)
                    
                    assert results['articles_scraped'] == 3
                    assert results['articles_processed'] == 2
                    assert results['sources_scraped'] == 1
                    assert 'verification_stats' in results
    
    @pytest.mark.asyncio
    async def test_shutdown_system(self, orchestrator):
        """Test system shutdown."""
        # Mock database close
        with patch('main.database_agent.close') as mock_close:
            mock_close.return_value = None
            
            await orchestrator.shutdown_system()
            
            mock_close.assert_called_once()


class TestWorkflowManager:
    """Test the WorkflowManager integration."""
    
    @pytest.fixture
    def workflow_manager(self):
        """Create a WorkflowManager instance."""
        return WorkflowManager()
    
    @pytest.mark.asyncio
    async def test_execute_full_research_workflow_success(self, workflow_manager):
        """Test successful full research workflow execution."""
        # Mock all agent operations
        with patch('agents.scheduler.web_scraper_agent.scrape_all_sources') as mock_scraper:
            mock_batch = MagicMock()
            mock_batch.successful_urls = 5
            mock_scraper.return_value = [mock_batch]
            
            with patch('agents.scheduler.extraction_agent.process_unprocessed_articles') as mock_extractor:
                mock_extractor.return_value = ['article1', 'article2', 'article3']
                
                with patch('agents.scheduler.summarizer_agent.batch_summarize_articles') as mock_summarizer:
                    mock_summaries = [MagicMock(article_id=f'article{i}') for i in range(3)]
                    mock_summarizer.return_value = mock_summaries
                    
                    with patch('agents.scheduler.connection_agent.analyze_article_connections') as mock_connector:
                        mock_connections = MagicMock()
                        mock_connections.connection_count = 2
                        mock_connections.trend_count = 1
                        mock_connections.identified_trends = []
                        mock_connector.return_value = mock_connections
                        
                        with patch('agents.scheduler.writer_agent.write_article') as mock_writer:
                            mock_article = MagicMock()
                            mock_article.word_count = 550
                            mock_article.quality_score = 0.85
                            mock_writer.return_value = mock_article
                            
                            with patch('agents.scheduler.quality_assurance_agent.validate_article') as mock_qa:
                                mock_validation = MagicMock()
                                mock_validation.passed = True
                                mock_validation.overall_score = 8.5
                                mock_qa.return_value = mock_validation
                                
                                with patch('agents.scheduler.telegram_bot_agent.distribute_article') as mock_telegram:
                                    mock_delivery = MagicMock()
                                    mock_delivery.success = True
                                    mock_delivery.telegram_message_id = 123
                                    mock_telegram.return_value = mock_delivery
                                    
                                    execution = await workflow_manager.execute_full_research_workflow(
                                        max_articles_per_source=5
                                    )
                                    
                                    assert execution.success
                                    assert execution.articles_scraped == 5
                                    assert execution.articles_processed == 3
                                    assert execution.articles_generated == 1
                                    assert execution.articles_published == 1
                                    assert execution.completed_steps == 7
                                    assert execution.progress_percentage() == 100.0
    
    @pytest.mark.asyncio
    async def test_execute_full_research_workflow_scraping_failure(self, workflow_manager):
        """Test workflow when scraping fails."""
        # Mock scraping to return no articles
        with patch('agents.scheduler.web_scraper_agent.scrape_all_sources') as mock_scraper:
            mock_batch = MagicMock()
            mock_batch.successful_urls = 0
            mock_scraper.return_value = [mock_batch]
            
            execution = await workflow_manager.execute_full_research_workflow()
            
            assert not execution.success
            assert execution.articles_scraped == 0
            assert len(execution.errors) > 0
            assert "No articles were successfully scraped" in execution.errors[0]
    
    @pytest.mark.asyncio
    async def test_execute_maintenance_workflow(self, workflow_manager):
        """Test maintenance workflow execution."""
        # Mock supervisor health check
        with patch('agents.scheduler.supervisor_agent.generate_health_report') as mock_health:
            mock_report = MagicMock()
            mock_report.overall_status.value = 'healthy'
            mock_report.get_healthy_agent_count.return_value = 8
            mock_report.critical_issues = []
            mock_health.return_value = mock_report
            
            with patch('agents.scheduler.database_agent.get_database_stats') as mock_db_stats:
                mock_db_stats.return_value = {'total_articles': 100}
                
                execution = await workflow_manager.execute_maintenance_workflow()
                
                assert execution.success
                assert execution.workflow_type == "maintenance"
                assert execution.completed_steps == 3
                assert 'health_check' in execution.results
                assert 'database_cleanup' in execution.results
    
    def test_get_execution_status(self, workflow_manager):
        """Test getting execution status."""
        # Should return None for non-existent execution
        status = workflow_manager.get_execution_status("non-existent")
        assert status is None
        
        # Test with active execution
        execution_id = "test-execution"
        mock_execution = MagicMock()
        workflow_manager.active_executions[execution_id] = mock_execution
        
        status = workflow_manager.get_execution_status(execution_id)
        assert status == mock_execution


class TestSchedulerAgent:
    """Test the SchedulerAgent integration."""
    
    @pytest.fixture
    def scheduler_agent(self):
        """Create a SchedulerAgent instance."""
        return SchedulerAgent()
    
    @pytest.mark.asyncio
    async def test_start_and_stop_scheduler(self, scheduler_agent):
        """Test starting and stopping the scheduler."""
        assert not scheduler_agent.is_running
        
        # Mock the APScheduler
        with patch.object(scheduler_agent.scheduler, 'start') as mock_start:
            with patch.object(scheduler_agent, '_setup_default_tasks') as mock_setup:
                mock_setup.return_value = None
                
                await scheduler_agent.start_scheduler()
                
                assert scheduler_agent.is_running
                mock_start.assert_called_once()
                mock_setup.assert_called_once()
        
        # Test stopping
        with patch.object(scheduler_agent.scheduler, 'shutdown') as mock_shutdown:
            await scheduler_agent.stop_scheduler()
            
            assert not scheduler_agent.is_running
            mock_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_schedule_research_workflow(self, scheduler_agent):
        """Test scheduling the research workflow."""
        # Mock the scheduler
        mock_job = MagicMock()
        mock_job.next_run_time = datetime.now()
        
        with patch.object(scheduler_agent.scheduler, 'add_job') as mock_add_job:
            mock_add_job.return_value = mock_job
            
            task_id = await scheduler_agent.schedule_research_workflow(
                interval_hours=4, 
                max_articles_per_source=10
            )
            
            assert task_id in scheduler_agent.scheduled_tasks
            task = scheduler_agent.scheduled_tasks[task_id]
            assert task.task_name == "Legal Research Workflow"
            assert task.schedule_config["hours"] == 4
            assert task.schedule_config["max_articles_per_source"] == 10
    
    @pytest.mark.asyncio
    async def test_execute_workflow_now(self, scheduler_agent):
        """Test immediate workflow execution."""
        # Mock the workflow manager
        with patch.object(scheduler_agent.workflow_manager, 'execute_full_research_workflow') as mock_execute:
            mock_execution = MagicMock()
            mock_execution.execution_id = "test-execution-123"
            mock_execute.return_value = mock_execution
            
            execution_id = await scheduler_agent.execute_workflow_now(
                workflow_type="full_research",
                max_articles_per_source=5
            )
            
            assert execution_id == "test-execution-123"
            mock_execute.assert_called_once_with(max_articles_per_source=5)
    
    def test_get_scheduler_status(self, scheduler_agent):
        """Test getting scheduler status."""
        # Mock scheduler jobs
        mock_job1 = MagicMock()
        mock_job1.id = "job1"
        mock_job1.name = "Test Job 1"
        mock_job1.next_run_time = datetime.now()
        
        mock_job2 = MagicMock()
        mock_job2.id = "job2"
        mock_job2.name = "Test Job 2"
        mock_job2.next_run_time = datetime.now()
        
        with patch.object(scheduler_agent.scheduler, 'get_jobs') as mock_get_jobs:
            mock_get_jobs.return_value = [mock_job1, mock_job2]
            
            status = scheduler_agent.get_scheduler_status()
            
            assert status['running'] == scheduler_agent.is_running
            assert status['total_jobs'] == 2
            assert len(status['next_jobs']) == 2
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, scheduler_agent):
        """Test cancelling a scheduled task."""
        # Add a task first
        task_id = "test-task-123"
        scheduler_agent.scheduled_tasks[task_id] = MagicMock()
        
        # Mock scheduler remove_job
        with patch.object(scheduler_agent.scheduler, 'remove_job') as mock_remove:
            mock_remove.return_value = None
            
            result = await scheduler_agent.cancel_task(task_id)
            
            assert result is True
            assert scheduler_agent.scheduled_tasks[task_id].enabled is False
            assert scheduler_agent.scheduled_tasks[task_id].status == "cancelled"


class TestSupervisorAgent:
    """Test the SupervisorAgent integration."""
    
    @pytest.fixture
    def supervisor_agent(self):
        """Create a SupervisorAgent instance."""
        return SupervisorAgent()
    
    @pytest.mark.asyncio
    async def test_generate_health_report(self, supervisor_agent):
        """Test health report generation."""
        # Mock agent health checks
        with patch.object(supervisor_agent.health_checker, 'check_database_agent') as mock_db_check:
            mock_db_check.return_value = MagicMock(status='healthy', agent_name='Database Agent')
            
            with patch.object(supervisor_agent.health_checker, 'check_web_scraper_agent') as mock_scraper_check:
                mock_scraper_check.return_value = MagicMock(status='healthy', agent_name='Web Scraper Agent')
                
                with patch.object(supervisor_agent.health_checker, 'check_telegram_bot_agent') as mock_telegram_check:
                    mock_telegram_check.return_value = MagicMock(status='healthy', agent_name='Telegram Bot Agent')
                    
                    with patch.object(supervisor_agent.health_checker, 'check_system_resources') as mock_system_check:
                        mock_system_check.return_value = {
                            'memory': {'percent': 50.0},
                            'cpu': {'percent': 25.0},
                            'disk': {'percent': 30.0}
                        }
                        
                        with patch.object(supervisor_agent, '_get_database_metrics') as mock_db_metrics:
                            mock_db_metrics.return_value = {'total_articles': 100}
                            
                            report = await supervisor_agent.generate_health_report()
                            
                            assert report.overall_status.value in ['healthy', 'warning', 'critical']
                            assert len(report.agent_statuses) == 3
                            assert report.system_uptime >= 0
                            assert 'memory' in report.system_metrics
    
    def test_update_performance_metrics(self, supervisor_agent):
        """Test performance metrics updates."""
        # Test scraping metrics
        supervisor_agent.update_performance_metrics("scraping", True, 2.5)
        assert supervisor_agent.performance_metrics.articles_scraped == 1
        assert supervisor_agent.performance_metrics.avg_scraping_time == 2.5
        
        supervisor_agent.update_performance_metrics("scraping", False)
        assert supervisor_agent.performance_metrics.scraping_errors == 1
        
        # Test processing metrics
        supervisor_agent.update_performance_metrics("processing", True, 1.5)
        assert supervisor_agent.performance_metrics.articles_processed == 1
        assert supervisor_agent.performance_metrics.avg_processing_time == 1.5
        
        # Test generation metrics
        supervisor_agent.update_performance_metrics("generation", True, 10.0)
        assert supervisor_agent.performance_metrics.articles_generated == 1
        assert supervisor_agent.performance_metrics.avg_generation_time == 10.0
        
        # Test distribution metrics
        supervisor_agent.update_performance_metrics("distribution", True)
        assert supervisor_agent.performance_metrics.articles_published == 1
    
    def test_get_monitoring_summary(self, supervisor_agent):
        """Test monitoring summary generation."""
        # Add some performance data
        supervisor_agent.performance_metrics.articles_scraped = 10
        supervisor_agent.performance_metrics.articles_processed = 8
        supervisor_agent.performance_metrics.articles_generated = 5
        
        summary = supervisor_agent.get_monitoring_summary()
        
        assert 'monitoring_active' in summary
        assert 'system_uptime_hours' in summary
        assert 'total_operations' in summary
        assert 'success_rate' in summary
        assert summary['total_operations'] == 23  # 10 + 8 + 5