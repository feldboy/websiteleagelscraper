#!/usr/bin/env python3
"""
Create sample legal articles for testing the full workflow.
"""

import asyncio
from datetime import datetime
from pydantic import HttpUrl

from agents.database import database_agent
from models.scraped_data import ScrapedData


async def create_sample_data():
    """Create sample legal articles in the database."""
    
    # Initialize database
    await database_agent.initialize()
    
    # Sample legal articles
    sample_articles = [
        {
            "url": "https://example.com/legal-news/supreme-court-ruling-1",
            "source_name": "Legal News Sample",
            "title": "Supreme Court Issues Major Privacy Ruling on Digital Surveillance",
            "content": """
            The Supreme Court issued a landmark ruling today addressing the scope of digital surveillance powers under the Fourth Amendment. 
            In a 6-3 decision, the Court held that law enforcement agencies must obtain a warrant before accessing location data from cell phone carriers for extended periods. 
            The case, which has been closely watched by privacy advocates and law enforcement, establishes new boundaries for digital privacy rights.
            
            Justice Roberts, writing for the majority, emphasized that the "digital revolution" requires courts to adapt Fourth Amendment protections to modern technology. 
            The decision specifically addresses how long-term location tracking differs from traditional surveillance methods.
            
            The ruling is expected to have significant implications for ongoing investigations and may require law enforcement to modify current practices. 
            Privacy rights organizations hailed the decision as a crucial step forward for digital privacy protection.
            
            Legal experts note that this ruling builds on previous decisions that have gradually extended constitutional protections to digital information. 
            The Court's opinion makes clear that technological advances cannot be used to circumvent established constitutional protections.
            
            The decision affects not only federal agencies but also state and local law enforcement across the country, who will need to adapt their procedures to comply with the new warrant requirements.
            """,
            "author": "Legal Reporter",
            "publish_date": datetime(2025, 7, 21, 10, 0, 0)
        },
        {
            "url": "https://example.com/legal-news/corporate-law-update-2", 
            "source_name": "Legal News Sample",
            "title": "New Corporate Disclosure Requirements Take Effect for Public Companies",
            "content": """
            New federal regulations requiring enhanced corporate disclosure went into effect today, mandating that public companies provide more detailed information about their cybersecurity practices and climate-related risks.
            
            The Securities and Exchange Commission (SEC) finalized these rules after extensive public comment periods, citing increased investor demand for transparency in these critical areas.
            
            Under the new requirements, companies must disclose material cybersecurity incidents within four business days and provide annual reports on their cybersecurity governance and risk management practices.
            
            Additionally, companies must now report on climate-related risks that could materially impact their business, including both physical risks from climate change and transition risks from regulatory changes.
            
            Corporate law firms report that their clients have been preparing for these changes for months, updating their internal processes and disclosure procedures.
            
            The regulations apply to all public companies regardless of size, though smaller companies will have additional time to comply with certain provisions.
            
            Legal experts expect these changes will lead to increased litigation as shareholders gain access to more detailed information about potential corporate risks and governance practices.
            
            The business community has expressed mixed reactions, with some viewing the requirements as necessary for investor protection while others worry about increased compliance costs.
            """,
            "author": "Corporate Law Correspondent",
            "publish_date": datetime(2025, 7, 21, 14, 30, 0)
        },
        {
            "url": "https://example.com/legal-news/employment-law-decision-3",
            "source_name": "Legal News Sample", 
            "title": "Federal Appeals Court Clarifies Remote Work Rights Under ADA",
            "content": """
            The Third Circuit Court of Appeals issued an important decision today clarifying when employers must provide remote work as a reasonable accommodation under the Americans with Disabilities Act (ADA).
            
            The unanimous three-judge panel ruled that employers cannot automatically reject remote work requests from qualified employees with disabilities, even for positions traditionally performed on-site.
            
            The case involved an accountant with mobility limitations who requested to work from home two days per week. The employer had denied the request, claiming that physical presence was essential to the position.
            
            The court emphasized that employers must engage in an individualized assessment of each accommodation request and demonstrate that remote work would cause undue hardship or fundamentally alter the nature of the job.
            
            This decision comes as many workplaces continue to adapt to post-pandemic work arrangements and as employees with disabilities increasingly seek flexible work options.
            
            Employment lawyers note that this ruling provides important guidance for both employers and employees navigating accommodation requests in the modern workplace.
            
            The court's opinion makes clear that blanket policies against remote work accommodations are likely to violate the ADA's requirement for individualized assessment.
            
            The decision is expected to influence similar cases in other circuits and may prompt the Supreme Court to address the issue if circuit courts reach conflicting conclusions.
            """,
            "author": "Employment Law Reporter",
            "publish_date": datetime(2025, 7, 21, 12, 15, 0)
        }
    ]
    
    # Insert sample articles
    for article_data in sample_articles:
        content = article_data["content"].strip()
        scraped_data = ScrapedData(
            url=HttpUrl(article_data["url"]),
            source_name=article_data["source_name"],
            title=article_data["title"],
            content=content,
            excerpt=content[:200] + "...",
            author=article_data["author"],
            publish_date=article_data["publish_date"],
            status_code=200,
            headers={"content-type": "text/html"},
            user_agent="Legal Research System Sample Data",
            response_time=0.5,
            content_length=len(content)
        )
        
        await database_agent.store_article(scraped_data)
        print(f"✅ Inserted sample article: {article_data['title']}")
    
    print(f"\n🎉 Successfully created {len(sample_articles)} sample legal articles!")
    
    # Close database
    await database_agent.close()


if __name__ == "__main__":
    asyncio.run(create_sample_data())