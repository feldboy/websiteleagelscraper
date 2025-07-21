#!/usr/bin/env python3
"""
Demo script to send a legal blog post via Telegram.
This simulates the final step of the complete workflow.
"""

import asyncio
from datetime import datetime

from agents.telegram_bot import telegram_bot_agent
from agents.writer import GeneratedArticle
from agents.quality_assurance import ValidationResult


async def send_demo_article():
    """Send a demo legal article via Telegram."""
    
    # Create a sample generated article (constrained to 650 chars max)
    demo_article = GeneratedArticle(
        title="Legal AI Weekly: Key Privacy and Corporate Disclosure Updates",
        content="Legal Developments Weekly: Digital Privacy Breakthrough and Corporate Compliance Updates. The legal landscape experienced significant shifts this week with landmark decisions affecting both individual privacy rights and corporate governance obligations. Supreme Court Strengthens Digital Privacy Protections. The Supreme Court delivered a decisive ruling on digital surveillance powers, establishing new constitutional boundaries for law enforcement access to personal location data. In a 6-3 decision, the Court held that government agencies must obtain judicial warrants before accessing extended location tracking information from cellular service providers. Justice Roberts, writing for the majority, emphasized that Fourth Amendment protections must evolve alongside technological advancement. The decision specifically addresses how prolonged digital surveillance differs fundamentally from traditional law enforcement methods, requiring additional constitutional safeguards. The ruling emerged from a case involving suspected criminal activity where law enforcement had obtained months of location data without judicial oversight. Privacy advocacy organizations argued this practice violated constitutional protections against unreasonable searches. Legal experts predict this decision will force significant procedural changes across federal, state, and local law enforcement agencies. Departments nationwide must now modify their investigative protocols to comply with new warrant requirements for digital location tracking. The decision builds upon previous Supreme Court rulings that have gradually extended constitutional protections to digital information, recognizing that technological capabilities cannot circumvent established constitutional principles. New Corporate Disclosure Standards Take Effect. Simultaneously, comprehensive corporate disclosure regulations became operational, requiring public companies to provide enhanced transparency regarding cybersecurity practices and climate-related business risks. The Securities and Exchange Commission finalized these requirements following extensive public consultation, responding to increased investor demand for corporate transparency in critical operational areas. Under new regulations, companies must disclose material cybersecurity incidents within four business days of determination. Additionally, organizations must provide detailed annual reports describing their cybersecurity governance structures, risk management processes, and incident response capabilities. Climate-related disclosure requirements mandate companies report both physical risks from environmental changes and transition risks from evolving regulatory frameworks. This includes potential financial impacts from climate change on business operations and strategic planning. Corporate legal departments report months of preparation for these compliance changes, involving substantial updates to internal reporting processes, governance structures, and disclosure procedures. Employment Law Advances in Remote Work Accommodations. Federal appeals courts clarified Americans with Disabilities Act requirements for remote work accommodations. The Third Circuit unanimously ruled that employers cannot automatically reject telework requests from qualified employees with disabilities. The decision requires individualized assessment of accommodation requests rather than blanket policies against remote work arrangements. Employers must demonstrate that remote work would create undue hardship or fundamentally alter essential job functions. This ruling comes as workplace flexibility remains prominent following pandemic-related changes to traditional office environments. Legal professionals note the decision provides crucial guidance for employers navigating accommodation requests in evolving workplace configurations. Looking Forward. These developments reflect broader legal trends toward enhanced privacy protection, corporate transparency, and workplace accommodation rights. Legal practitioners should monitor how these decisions influence ongoing litigation strategies, compliance programs, and employment policies. The convergence of digital privacy strengthening, corporate transparency requirements, and workplace accommodation expansion indicates a legal environment increasingly focused on individual rights protection and organizational accountability. Organizations across all sectors should evaluate current practices against these new standards and consider proactive compliance measures to address evolving legal requirements in privacy, transparency, and workplace accommodation areas.",
        summary="Weekly legal update covering Supreme Court digital privacy ruling and new SEC corporate disclosure requirements.",
        tags=["privacy", "supreme court", "corporate law", "SEC", "cybersecurity", "legal news"],
        word_count=545,
        quality_score=0.95,
        originality_score=0.98,
        readability_score=0.92
    )
    
    # Create a sample validation result
    validation_result = ValidationResult(
        validation_id="val-demo-001", 
        article_id="demo-article-001",
        passed=True,
        overall_score=8.5,
        confidence=0.95,
        word_count_compliant=True,
        originality_score=0.98,
        legal_quality_score=8.5,
        readability_score=8.2,
        factual_accuracy_score=8.0,
        legal_terminology_adequate=True,
        structure_adequate=True,
        critical_issues=[],
        warnings=["Consider adding more specific citations"],
        recommendations=[
            "Include publication dates for referenced cases",
            "Add more direct quotes from court decisions"
        ]
    )
    
    print("🚀 Sending demo legal blog post via Telegram...")
    print(f"📰 Article: {demo_article.title}")
    print(f"📝 Word count: {demo_article.word_count}")
    print(f"✅ Quality score: {demo_article.quality_score}")
    
    try:
        # Send the article via Telegram
        distribution_result = await telegram_bot_agent.distribute_article(
            demo_article, validation_result
        )
        
        if distribution_result.success:
            print(f"🎉 Successfully sent to Telegram!")
            print(f"📱 Message ID: {distribution_result.telegram_message_id}")
            print(f"👥 Channel: {distribution_result.channel_info}")
            print(f"🔗 Message URL: {distribution_result.message_url}")
        else:
            print(f"❌ Failed to send: {distribution_result.error_message}")
            
    except Exception as e:
        print(f"❌ Error during Telegram distribution: {e}")


if __name__ == "__main__":
    asyncio.run(send_demo_article())