#!/usr/bin/env python3
"""
Create a properly formatted demo legal article matching the meta.txst style.
"""

import asyncio
from datetime import datetime

from agents.telegram_bot import telegram_bot_agent
from agents.writer import GeneratedArticle
from agents.quality_assurance import ValidationResult


async def send_formatted_demo():
    """Send a formatted legal article via Telegram."""
    
    # Create an article in the same format as meta.txst example
    formatted_content = """**Supreme Court Delivers Landmark Digital Privacy Ruling**

The Supreme Court issued a decisive 6-3 ruling this week that fundamentally reshapes digital surveillance law. The decision requires law enforcement agencies to obtain judicial warrants before accessing extended location data from cellular service providers—a requirement that will force significant procedural changes across all levels of law enforcement nationwide.

**How Fourth Amendment Protections Evolved for the Digital Age**

At the heart of the case was whether prolonged digital surveillance constitutes a fundamentally different intrusion than traditional law enforcement methods. Justice Roberts, writing for the majority, emphasized that constitutional protections must evolve alongside technological advancement. The ruling emerged from a case involving suspected criminal activity where law enforcement had obtained months of location data without judicial oversight, tracking a suspect's movements with unprecedented precision and scope.

Privacy advocacy organizations had argued this practice violated constitutional protections against unreasonable searches, while law enforcement contended that location data fell outside traditional Fourth Amendment frameworks. The Court's majority opinion rejected this narrow interpretation, establishing that digital surveillance requires the same constitutional protections as physical searches.

**A Decision That Reshapes Investigative Procedures**

The Court's opinion makes clear that technological capabilities cannot circumvent established constitutional principles. Legal experts predict the ruling will require substantial updates to investigative protocols, with departments nationwide needing to modify their digital surveillance procedures to comply with new warrant requirements within the next fiscal year.

The decision builds upon previous Supreme Court rulings that have gradually extended constitutional protections to digital information, recognizing that the "digital revolution" demands adaptive legal frameworks rather than technological workarounds to constitutional rights. Lower federal courts had been split on this issue, with some circuits allowing warrantless location tracking and others requiring judicial approval.

**Corporate Disclosure Standards Take Simultaneous Effect**

In a parallel development, comprehensive corporate disclosure regulations became operational this week, requiring public companies to provide enhanced transparency regarding cybersecurity practices and climate-related business risks. The Securities and Exchange Commission finalized these requirements following extensive public consultation, responding to increased investor demand for corporate transparency in critical operational areas that affect shareholder value.

Under the new regulations, companies must disclose material cybersecurity incidents within four business days of determination and provide detailed annual reports describing their cybersecurity governance structures, risk management processes, and incident response capabilities. Climate-related disclosure requirements mandate companies report both physical risks from environmental changes and transition risks from evolving regulatory frameworks.

Corporate legal departments report months of preparation for these compliance changes, involving substantial updates to internal reporting processes, governance structures, and disclosure procedures. The requirements apply to all public companies regardless of size, though smaller companies receive additional implementation time for certain provisions.

**Why These Changes Signal Broader Legal Trends**

The convergence of digital privacy strengthening and corporate transparency requirements indicates a legal environment increasingly focused on accountability and rights protection. These developments reflect broader judicial and regulatory trends toward enhanced oversight of both government surveillance powers and corporate risk disclosure practices.

Organizations across all sectors should evaluate current practices against these new standards and consider proactive compliance measures to address evolving legal requirements in both privacy protection and corporate governance areas. The timing of these simultaneous changes suggests coordinated efforts to strengthen oversight in an increasingly digital business environment."""

    demo_article = GeneratedArticle(
        title="Supreme Court Strengthens Digital Privacy as Corporate Disclosure Standards Take Effect",
        content=formatted_content,
        summary="Supreme Court ruling on digital surveillance and new SEC corporate disclosure requirements reshape legal landscape for privacy and transparency.",
        tags=["digital privacy", "supreme court", "corporate governance", "SEC", "fourth amendment", "cybersecurity"],
        word_count=len(formatted_content.split()),
        quality_score=0.92,
        originality_score=0.96,
        readability_score=0.88,
        source_summaries=[
            "Supreme Court issues landmark digital privacy ruling from Law.com legal analysis",
            "SEC announces new corporate disclosure requirements covered by Legal Reader", 
            "Federal appeals court clarifies remote work rights from Above the Law reporting"
        ]
    )
    
    # Create validation result
    validation_result = ValidationResult(
        validation_id="val-formatted-001", 
        article_id="formatted-demo-001",
        passed=True,
        overall_score=8.8,
        confidence=0.92,
        word_count_compliant=True,
        originality_score=0.96,
        legal_quality_score=8.8,
        readability_score=8.5,
        factual_accuracy_score=8.2,
        legal_terminology_adequate=True,
        structure_adequate=True,
        critical_issues=[],
        warnings=[],
        recommendations=["Consider adding specific case citations for enhanced credibility"]
    )
    
    print("🚀 Sending formatted legal analysis via Telegram...")
    print(f"📰 Title: {demo_article.title}")
    print(f"📝 Word count: {demo_article.word_count}")
    print(f"✅ Quality score: {demo_article.quality_score}")
    
    try:
        # Send the formatted article via Telegram
        distribution_result = await telegram_bot_agent.distribute_article(
            demo_article, validation_result
        )
        
        if distribution_result.success:
            print(f"🎉 Successfully sent formatted article to Telegram!")
            print(f"📱 Message ID: {distribution_result.telegram_message_id}")
        else:
            print(f"❌ Failed to send: {distribution_result.error_message}")
            
    except Exception as e:
        print(f"❌ Error during Telegram distribution: {e}")


if __name__ == "__main__":
    asyncio.run(send_formatted_demo())