"""
Main orchestration script for the Multi-Agent Legal Research System.
Coordinates all 11 agents in a complete workflow for autonomous legal content generation.
"""
import asyncio
import logging
import sys
from datetime import datetime
from typing import List, Optional

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
from agents.scheduler import scheduler_agent


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_research_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class LegalResearchOrchestrator:
    """
    Main orchestrator for the multi-agent legal research system.
    Coordinates the workflow from scraping to article generation and distribution.
    """
    
    def __init__(self):
        self.system_initialized = False
        self.workflow_stats = {
            'articles_scraped': 0,
            'articles_processed': 0,
            'articles_generated': 0,
            'articles_published': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def initialize_system(self) -> bool:
        """Initialize all agents and system components."""
        try:
            logger.info("🚀 Initializing Multi-Agent Legal Research System...")
            
            # Initialize database
            await database_agent.initialize()
            logger.info("✅ Database agent initialized")
            
            # Verify configuration
            if not self._verify_configuration():
                logger.error("❌ Configuration verification failed")
                return False
            
            logger.info("✅ System configuration verified")
            
            # Test LLM connectivity
            from tools.llm_client import llm_client
            test_response = await llm_client.generate("Test message", max_tokens=10)
            if not test_response.success:
                logger.error(f"❌ LLM connectivity test failed: {test_response.error}")
                return False
            
            logger.info("✅ LLM connectivity verified")
            
            self.system_initialized = True
            logger.info("🎉 System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def _verify_configuration(self) -> bool:
        """Verify that all required configuration is present."""
        required_settings = [
            'database_url',
            'llm_provider',
        ]
        
        for setting in required_settings:
            if not hasattr(settings, setting) or not getattr(settings, setting):
                logger.error(f"Missing required setting: {setting}")
                return False
        
        # Check LLM API keys based on provider
        if settings.llm_provider == 'openai' and not settings.openai_api_key:
            logger.error("OpenAI API key is required when using OpenAI provider")
            return False
        elif settings.llm_provider == 'anthropic' and not settings.anthropic_api_key:
            logger.error("Anthropic API key is required when using Anthropic provider")
            return False
        
        return True
    
    async def run_complete_workflow(self, max_articles_per_source: int = 10) -> dict:
        """
        Run the complete workflow from scraping to article generation.
        
        Args:
            max_articles_per_source: Maximum articles to scrape per source
            
        Returns:
            Dictionary with workflow statistics
        """
        if not self.system_initialized:
            logger.error("System not initialized. Call initialize_system() first.")
            return {}
        
        self.workflow_stats['start_time'] = datetime.now()
        logger.info("🔄 Starting complete legal research workflow...")
        
        try:
            # Step 1: Web Scraping
            logger.info("📰 Step 1: Web scraping legal news sources...")
            scraping_batches = await web_scraper_agent.scrape_all_sources(max_articles_per_source)
            
            total_scraped = sum(batch.successful_urls for batch in scraping_batches)
            self.workflow_stats['articles_scraped'] = total_scraped
            logger.info(f"✅ Scraped {total_scraped} articles from {len(scraping_batches)} sources")
            
            # Step 2: Data Extraction
            logger.info("🔍 Step 2: Extracting structured data from articles...")
            # Process existing articles even if no new ones were scraped
            limit = max(total_scraped, 10)  # Process at least 10 existing articles if no new ones
            processed_ids = await extraction_agent.process_unprocessed_articles(limit=limit)
            
            self.workflow_stats['articles_processed'] = len(processed_ids)
            logger.info(f"✅ Processed {len(processed_ids)} articles for data extraction")
            
            if len(processed_ids) == 0:
                logger.warning("⚠️ No articles were successfully processed")
                return self.workflow_stats
            
            # Step 3: Summarization
            logger.info("📝 Step 3: Generating article summaries...")
            summaries = await summarizer_agent.batch_summarize_articles(processed_ids[:5])  # Limit for demo
            
            logger.info(f"✅ Generated {len(summaries)} article summaries")
            
            if len(summaries) == 0:
                logger.warning("⚠️ No summaries were generated")
                return self.workflow_stats
            
            # Step 4: Connection Analysis
            logger.info("🔗 Step 4: Analyzing connections between articles...")
            article_ids = [summary.article_id for summary in summaries]
            connections = await connection_agent.analyze_article_connections(article_ids)
            
            logger.info(f"✅ Found {connections.connection_count} connections and {connections.trend_count} trends")
            
            # Step 5: Article Generation
            logger.info("✍️ Step 5: Generating comprehensive legal blog article...")
            
            # Use top summaries for article generation
            top_summaries = summaries[:3]  # Use top 3 summaries
            generated_article = await writer_agent.write_article(
                summaries=top_summaries,
                connections=connections,
                trends=connections.identified_trends
            )
            
            self.workflow_stats['articles_generated'] = 1
            logger.info(f"✅ Generated article: '{generated_article.title}' ({generated_article.word_count} words)")
            
            # Step 6: Quality Assurance
            logger.info("🔍 Step 6: Quality assurance validation...")
            validation_result = await quality_assurance_agent.validate_article(generated_article)
            
            logger.info(f"✅ Quality validation completed: passed={validation_result.passed}, score={validation_result.overall_score:.1f}")
            
            if validation_result.passed:
                # Step 7: Distribution via Telegram
                logger.info("📢 Step 7/7: Distributing article via Telegram...")
                
                try:
                    distribution_result = await telegram_bot_agent.distribute_article(
                        generated_article, validation_result
                    )
                    
                    if distribution_result.success:
                        self.workflow_stats['articles_published'] = 1
                        logger.info(f"🎉 Article published successfully! Telegram message ID: {distribution_result.telegram_message_id}")
                    else:
                        logger.warning(f"📢 Distribution failed: {distribution_result.error_message}")
                        
                except Exception as e:
                    logger.error(f"📢 Distribution error: {e}")
                    
            else:
                logger.warning(f"⚠️ Article failed quality validation: {validation_result.critical_issues}")
            
            # Step 8: System Health Check and Statistics
            logger.info("📊 Step 8: Generating system health report and statistics...")
            
            # Generate health report
            health_report = await supervisor_agent.generate_health_report()
            logger.info(f"🏥 System health: {health_report.overall_status.value}, "
                       f"{health_report.get_healthy_agent_count()}/{len(health_report.agent_statuses)} agents healthy")
            
            # Get database statistics
            db_stats = await database_agent.get_database_stats()
            
            logger.info("📈 Workflow Statistics:")
            logger.info(f"  • Articles scraped: {self.workflow_stats['articles_scraped']}")
            logger.info(f"  • Articles processed: {self.workflow_stats['articles_processed']}")
            logger.info(f"  • Summaries generated: {len(summaries)}")
            logger.info(f"  • Connections found: {connections.connection_count}")
            logger.info(f"  • Trends identified: {connections.trend_count}")
            logger.info(f"  • Articles generated: {self.workflow_stats['articles_generated']}")
            logger.info(f"  • Articles approved: {self.workflow_stats['articles_published']}")
            logger.info(f"  • Database total articles: {db_stats.get('total_articles', 0)}")
            
            return {
                **self.workflow_stats,
                'validation_result': validation_result.dict(),
                'generated_article': {
                    'title': generated_article.title,
                    'word_count': generated_article.word_count,
                    'quality_score': generated_article.quality_score,
                    'originality_score': generated_article.originality_score
                },
                'connections_summary': {
                    'connection_count': connections.connection_count,
                    'cluster_count': connections.cluster_count,
                    'trend_count': connections.trend_count
                },
                'database_stats': db_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            return self.workflow_stats
        
        finally:
            self.workflow_stats['end_time'] = datetime.now()
            duration = (self.workflow_stats['end_time'] - self.workflow_stats['start_time']).total_seconds()
            logger.info(f"⏱️ Workflow completed in {duration:.1f} seconds")
    
    async def run_scraping_only(self, max_articles: int = 5) -> dict:
        """Run only the scraping and initial processing workflow."""
        if not self.system_initialized:
            logger.error("System not initialized. Call initialize_system() first.")
            return {}
        
        logger.info("🔄 Starting scraping-only workflow...")
        
        try:
            # Web scraping
            scraping_batches = await web_scraper_agent.scrape_all_sources(max_articles)
            total_scraped = sum(batch.successful_urls for batch in scraping_batches)
            
            # Data extraction
            processed_ids = await extraction_agent.process_unprocessed_articles(limit=total_scraped)
            
            # Basic verification
            verification_stats = await verification_agent.get_verification_stats()
            
            return {
                'articles_scraped': total_scraped,
                'articles_processed': len(processed_ids),
                'verification_stats': verification_stats,
                'sources_scraped': len(scraping_batches)
            }
            
        except Exception as e:
            logger.error(f"❌ Scraping workflow failed: {e}")
            return {}
    
    async def shutdown_system(self):
        """Gracefully shutdown the system."""
        logger.info("🛑 Shutting down Multi-Agent Legal Research System...")
        
        try:
            # Close database connections
            await database_agent.close()
            logger.info("✅ Database connections closed")
            
            logger.info("✅ System shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


async def main():
    """Main entry point for the legal research system."""
    orchestrator = LegalResearchOrchestrator()
    
    try:
        # Initialize system
        success = await orchestrator.initialize_system()
        if not success:
            logger.error("Failed to initialize system")
            return
        
        # Run workflow based on command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == "scraping-only":
            logger.info("Running scraping-only workflow")
            results = await orchestrator.run_scraping_only(max_articles=3)
        else:
            logger.info("Running complete workflow")
            results = await orchestrator.run_complete_workflow(max_articles_per_source=2)
        
        # Print final results
        if results:
            logger.info("🎯 Final Results:")
            for key, value in results.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"    {sub_key}: {sub_value}")
                else:
                    logger.info(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Workflow interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        await orchestrator.shutdown_system()


if __name__ == "__main__":
    # Ensure logs directory exists
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Run the main workflow
    asyncio.run(main())