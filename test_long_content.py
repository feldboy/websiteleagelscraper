#!/usr/bin/env python3
"""
Test the Telegram truncation logic with very long content.
"""

import asyncio
from datetime import datetime

from agents.telegram_bot import telegram_bot_agent
from agents.writer import GeneratedArticle
from agents.quality_assurance import ValidationResult


async def test_long_content():
    """Test truncation with very long content."""
    
    # Create very long content (over 4096 characters)
    long_content = """**Supreme Court Delivers Landmark Digital Privacy Ruling**

The Supreme Court issued a decisive 6-3 ruling this week that fundamentally reshapes digital surveillance law across all fifty states and federal jurisdictions. The decision requires law enforcement agencies at federal, state, and local levels to obtain judicial warrants before accessing extended location data from cellular service providers—a requirement that will force significant procedural changes across all levels of law enforcement nationwide within the next fiscal year.

**How Fourth Amendment Protections Evolved for the Digital Age**

At the heart of the case was whether prolonged digital surveillance constitutes a fundamentally different intrusion than traditional law enforcement methods that have been used for decades by police departments across the country. Justice Roberts, writing for the majority opinion, emphasized that constitutional protections must evolve alongside technological advancement and the rapid digitization of personal information. The ruling emerged from a case involving suspected criminal activity where law enforcement had obtained months of detailed location data without judicial oversight, tracking a suspect's movements with unprecedented precision and scope across multiple states and jurisdictions.

Privacy advocacy organizations had argued this practice violated constitutional protections against unreasonable searches, while law enforcement contended that location data fell outside traditional Fourth Amendment frameworks that were established in the pre-digital era. The Court's majority opinion firmly rejected this narrow interpretation, establishing that digital surveillance requires the same constitutional protections as physical searches of homes and personal property.

**A Decision That Reshapes Investigative Procedures Nationwide**

The Court's opinion makes clear that technological capabilities cannot circumvent established constitutional principles that have protected American citizens for over two centuries. Legal experts across the country predict the ruling will require substantial updates to investigative protocols, with departments nationwide needing to modify their digital surveillance procedures to comply with new warrant requirements within the next fiscal year, potentially requiring significant budget allocations for legal compliance and training.

The decision builds upon previous Supreme Court rulings that have gradually extended constitutional protections to digital information, recognizing that the "digital revolution" demands adaptive legal frameworks rather than technological workarounds to constitutional rights. Lower federal courts had been split on this issue for several years, with some circuits allowing warrantless location tracking and others requiring judicial approval, creating a patchwork of legal standards across different jurisdictions.

**Corporate Disclosure Standards Take Simultaneous Effect**

In a parallel development that reflects broader regulatory trends, comprehensive corporate disclosure regulations became operational this week, requiring public companies to provide enhanced transparency regarding cybersecurity practices and climate-related business risks that could affect shareholder value and market stability. The Securities and Exchange Commission finalized these requirements following extensive public consultation periods, responding to increased investor demand for corporate transparency in critical operational areas that affect shareholder value and long-term business sustainability.

Under the new regulations, companies must disclose material cybersecurity incidents within four business days of determination and provide detailed annual reports describing their cybersecurity governance structures, risk management processes, and incident response capabilities that protect corporate assets and customer data. Climate-related disclosure requirements mandate companies report both physical risks from environmental changes and transition risks from evolving regulatory frameworks that could significantly impact business operations and profitability.

Corporate legal departments across industries report months of intensive preparation for these compliance changes, involving substantial updates to internal reporting processes, governance structures, and disclosure procedures that require coordination across multiple business units and external advisors. The requirements apply to all public companies regardless of size, though smaller companies receive additional implementation time for certain provisions to accommodate limited resources.

**Why These Changes Signal Broader Legal Trends**

The convergence of digital privacy strengthening and corporate transparency requirements indicates a legal environment increasingly focused on accountability and rights protection across both government and private sector activities. These developments reflect broader judicial and regulatory trends toward enhanced oversight of both government surveillance powers and corporate risk disclosure practices that affect millions of Americans.

Organizations across all sectors should evaluate current practices against these new standards and consider proactive compliance measures to address evolving legal requirements in both privacy protection and corporate governance areas. The timing of these simultaneous changes suggests coordinated efforts to strengthen oversight in an increasingly digital business environment."""

    demo_article = GeneratedArticle(
        title="Supreme Court Digital Privacy Ruling and Corporate Disclosure Standards",
        content=long_content,
        summary="Comprehensive analysis of Supreme Court digital privacy decision and new SEC corporate disclosure requirements.",
        tags=["digital privacy", "supreme court", "corporate governance", "SEC", "fourth amendment"],
        word_count=len(long_content.split()),
        quality_score=0.90,
        originality_score=0.95,
        readability_score=0.85,
        source_summaries=[
            "Supreme Court digital privacy case analysis from Law.com",
            "SEC corporate disclosure requirements from Legal Reader", 
            "Federal court surveillance ruling from Above the Law"
        ]
    )
    
    validation_result = ValidationResult(
        validation_id="val-long-001", 
        article_id="long-demo-001",
        passed=True,
        overall_score=8.5,
        confidence=0.90,
        word_count_compliant=True,
        originality_score=0.95,
        legal_quality_score=8.5,
        readability_score=8.0,
        factual_accuracy_score=8.2,
        legal_terminology_adequate=True,
        structure_adequate=True,
        critical_issues=[],
        warnings=[],
        recommendations=[]
    )
    
    print("🧪 Testing long content with source preservation...")
    print(f"📝 Content length: {len(long_content)} characters")
    print(f"📰 Word count: {demo_article.word_count}")
    
    try:
        result = await telegram_bot_agent.distribute_article(demo_article, validation_result)
        
        if result.success:
            print(f"✅ Successfully sent long content to Telegram!")
            print(f"📱 Message ID: {result.telegram_message_id}")
        else:
            print(f"❌ Failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_long_content())