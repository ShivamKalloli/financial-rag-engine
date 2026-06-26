"""
Seed sample financial documents for development and testing.

Generates 3 realistic financial text documents in data/raw/ WITHOUT any
external API calls. All content is statically defined below.

Documents:
    1. apple_q4_2023_earnings.txt    (~600 words)
    2. microsoft_fy2023_annual.txt   (~500 words)
    3. tesla_q3_2023_earnings.txt    (~500 words)

Usage:
    python scripts/seed_sample_data.py
"""

import os
import sys

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join("data", "raw")

# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------

APPLE_Q4_2023 = """Apple Inc. — Q4 Fiscal Year 2023 Earnings Results
Quarterly Period Ended: September 30, 2023
Released: November 2, 2023

FINANCIAL HIGHLIGHTS

Apple Inc. reported revenue of $89.5 billion for the fourth fiscal quarter of 2023,
representing a 1% increase year-over-year compared to $90.1 billion in the same period
of fiscal year 2022. The modest year-over-year growth reflects resilience in the company's
diversified portfolio despite macroeconomic headwinds and foreign exchange pressures.

Product Revenue Breakdown:
- iPhone revenue: $43.8 billion, a slight decrease from $42.6 billion in Q4 2022
- Mac revenue: $7.6 billion, up from $7.4 billion in the prior year period
- iPad revenue: $6.4 billion, compared to $7.2 billion a year ago
- Wearables, Home and Accessories revenue: $9.3 billion
- Services revenue: $22.3 billion — an all-time quarterly record for the category

The Services segment, which includes the App Store, Apple Music, Apple TV+, iCloud,
Apple Pay, and AppleCare, achieved a record $22.3 billion in revenue, up 16% year-over-year.
This milestone underscores Apple's accelerating transition toward high-margin recurring
revenue and its growing ecosystem of over 1 billion paid subscriptions globally.

PROFITABILITY

Apple reported a gross margin of 45.2% for the quarter, up from 42.3% in Q4 2022.
The improvement in gross margin reflects the favorable revenue mix shift toward Services,
which carries significantly higher margins than hardware products.

Operating income for the quarter was $26.9 billion, representing an operating margin
of 30.1%. Net income was $22.96 billion, or $1.46 per diluted share, compared to
$20.72 billion, or $1.29 per diluted share, in Q4 fiscal 2022.

GEOGRAPHIC PERFORMANCE AND CHINA HEADWINDS

Americas revenue reached $40.0 billion, Europe contributed $22.5 billion, and
Greater China generated $15.1 billion — a decline of 2.5% year-over-year reflecting
intensifying competition from domestic Chinese smartphone manufacturers and ongoing
regulatory scrutiny. The China weakness was a primary concern cited by analysts.

Japan and Rest of Asia Pacific combined for $11.9 billion.

EXECUTIVE COMMENTARY

Chief Executive Officer Tim Cook commented on the quarter's results:

"We are thrilled to close out fiscal 2023 with an all-time revenue record for Services
and strong performance across our product lineup. Our teams are continuously pushing
the boundaries of what's possible. Innovation remains at the heart of everything we do,
and we are incredibly excited about the artificial intelligence capabilities we are
integrating across our entire ecosystem. AI will be transformative for Apple, and we
believe we are uniquely positioned to deliver AI in a way that is powerful, personal,
and private."

Cook emphasized Apple's ongoing investment in research and development to build
proprietary large language models and on-device machine learning capabilities.

CFO GUIDANCE FOR Q1 FISCAL 2024

Chief Financial Officer Luca Maestri provided the following guidance for the first
fiscal quarter of 2024 (October–December 2023):

- Total revenue expected to be similar to Q1 fiscal 2023 ($117.2 billion), with
  approximately 3–4 percentage point adverse impact from foreign exchange fluctuations
- Gross margin expected between 45% and 46%
- Operating expenses between $14.3 billion and $14.5 billion
- Tax rate approximately 16%

Maestri noted that the Services segment is expected to continue its strong double-digit
growth trajectory, while iPhone demand for the iPhone 15 lineup is tracking ahead of
the iPhone 14 family at the comparable period.

CAPITAL RETURN PROGRAM

Apple returned over $24 billion to shareholders in Q4, including $3.8 billion in
dividends and $20.5 billion in share repurchases. The company's board of directors
declared a cash dividend of $0.24 per share.

SENTIMENT ASSESSMENT

The quarter presents a mixed sentiment picture. On the positive side, the record
$22.3 billion Services revenue demonstrates the strength of Apple's ecosystem and
software monetization capabilities. Gross margin expansion to 45.2% reflects
operational excellence and favorable mix shift.

However, cautionary signals include the Greater China revenue decline of 2.5%,
competitive pressure from Huawei's Mate 60 Pro launch, and currency headwinds
expected to persist into Q1 FY2024. The flat iPhone revenue trajectory suggests
unit volume pressure in the premium smartphone market.
"""

MICROSOFT_FY2023 = """Microsoft Corporation — Fiscal Year 2023 Annual Results
Period Ended: June 30, 2023
Released: July 25, 2023

FISCAL YEAR 2023 FINANCIAL OVERVIEW

Microsoft Corporation reported total revenue of $211.9 billion for fiscal year 2023,
representing a 7% increase compared to $198.3 billion in fiscal year 2022. The
results demonstrate Microsoft's ability to sustain growth despite macroeconomic
headwinds impacting technology spending across enterprise and consumer markets.

Operating income increased 6% to $88.5 billion, with an operating margin of 41.8%.
Net income reached $72.4 billion, translating to earnings per diluted share of $9.72,
up from $9.65 in the prior fiscal year.

SEGMENT PERFORMANCE

Microsoft reports results across three business segments:

1. Productivity and Business Processes
Revenue: $69.3 billion (+9% YoY)
- Microsoft 365 Commercial revenue grew 13% year-over-year in constant currency
- Microsoft 365 Consumer revenue grew 3%, with 76.7 million subscribers
- LinkedIn revenue grew 10%, driven by Talent Solutions and Marketing Solutions
- Dynamics 365 revenue grew 26%, reflecting strong adoption of cloud ERP/CRM solutions

2. Intelligent Cloud
Revenue: $87.9 billion (+17% YoY)
This segment represents Microsoft's largest and fastest-growing revenue stream.
- Azure and other cloud services grew 29% year-over-year in constant currency
- Azure's growth reflects enterprise migration to cloud infrastructure and the
  strong adoption of AI-powered services including Azure OpenAI Service
- Server products and cloud services revenue grew 17% overall
- Enterprise Mobility + Security installed base exceeded 270 million seats

3. More Personal Computing
Revenue: $54.7 billion (-9% YoY)
- Windows OEM revenue declined 25% reflecting PC market normalization post-pandemic
- Xbox content and services revenue grew 3%
- Search and news advertising revenue (ex-traffic acquisition costs) grew 11%,
  benefiting from AI-enhanced Bing and Edge market share gains

ARTIFICIAL INTELLIGENCE INTEGRATION

Chief Executive Officer Satya Nadella highlighted Microsoft's AI strategy:

"We are rapidly infusing AI across every layer of our technology stack. The launch
of Microsoft 365 Copilot, GitHub Copilot, Azure AI, and the new AI-powered Bing
represents the most significant platform shift we have seen in decades. Our partnership
with OpenAI provides us a structural advantage, and we are moving with urgency to
embed AI capabilities into every product we ship. This is a generational opportunity
for Microsoft to help every person and organization on the planet achieve more."

Microsoft committed $10 billion to OpenAI and expects AI to materially accelerate
revenue across all three segments in fiscal year 2024.

CAPITAL ALLOCATION

Microsoft returned $9.7 billion to shareholders through dividends ($7.0 billion)
and share repurchases ($2.7 billion) in fiscal year 2023.

The company declared a quarterly dividend of $0.68 per share, a 10% increase from
the prior year, reflecting confidence in long-term cash generation.

OUTLOOK AND MACROECONOMIC CONTEXT

Despite positive momentum in cloud and AI, Microsoft noted macroeconomic headwinds
including cautious enterprise IT spending, foreign exchange headwinds of approximately
4 percentage points on revenue growth, and softness in the PC market impacting
Windows OEM and Surface device revenue.

For fiscal year 2024, management expects continued double-digit growth in Azure
and Microsoft 365 Commercial, acceleration in AI-driven workload revenues, and
gradual recovery in More Personal Computing as the PC market stabilizes.

SENTIMENT: BROADLY POSITIVE with acknowledgment of macro headwinds.
"""

TESLA_Q3_2023 = """Tesla, Inc. — Q3 2023 Earnings Results
Quarter Ended: September 30, 2023
Released: October 18, 2023

PRODUCTION AND DELIVERY HIGHLIGHTS

Tesla, Inc. delivered 435,059 vehicles in the third quarter of 2023, representing
a 27% increase year-over-year and a quarterly record for the company at that time.
Production totaled 430,488 vehicles, comprised of 416,800 Model 3 and Model Y
units and 13,688 units of other models (Model S, Model X, Cybertruck pre-production).

The strong delivery volume reflects Tesla's manufacturing scale-up at Gigafactory
Shanghai, Gigafactory Texas, and Gigafactory Berlin-Brandenburg.

FINANCIAL RESULTS

Total revenue for Q3 2023 was $23.4 billion, a 9% increase from $21.5 billion in
Q3 2022. However, automotive revenue growth was partially offset by ongoing price
reductions implemented across the vehicle lineup.

Revenue breakdown:
- Automotive revenue: $19.6 billion (including $554 million in regulatory credit sales)
- Energy generation and storage revenue: $1.56 billion (+40% YoY) — a record quarter
- Services and other revenue: $2.2 billion (+32% YoY)

MARGIN PRESSURE FROM PRICE REDUCTION STRATEGY

Automotive gross margin declined to 17.9% in Q3 2023, down from 25.1% in Q3 2022
and 18.1% in Q2 2023. The contraction reflects Tesla's deliberate strategy of
reducing vehicle prices to stimulate demand and protect volume growth.

CEO Elon Musk commented on the margin dynamics and long-term vision:

"The macroeconomic environment remains challenging and high interest rates are making
it difficult for many consumers to afford new vehicles. We've made the deliberate
choice to prioritize volume growth over near-term margin. I am focused on the long
game. Full Self-Driving capability will transform Tesla from a hardware company into
the most valuable company in the world. When robotaxis are deployed at scale, the
economics are extraordinary. One Tesla vehicle will generate more revenue than five
average new cars sold today. That is the future we are building."

Musk also addressed energy storage, noting that Megapack deployments are accelerating.

CYBERTRUCK UPDATE

Tesla provided an update on the highly anticipated Cybertruck. Production at
Gigafactory Texas has begun for the Cybertruck, with the first customer deliveries
expected in late Q4 2023 and volume ramp throughout 2024. The Cybertruck represents
Tesla's entry into the full-size pickup truck segment, with a starting price expected
above $60,000. Management cautioned that Cybertruck production ramp will take
approximately 12–18 months to reach meaningful volume.

ENERGY DIVISION RECORD

The Energy Generation and Storage segment delivered $1.56 billion in revenue, an
all-time record, driven by record Megapack deployments of 4.0 GWh in the quarter.
The Lathrop, California Megafactory is now operational and expected to significantly
increase Megapack production capacity in 2024.

COST REDUCTION INITIATIVES

Tesla reduced its cost of goods sold per vehicle to approximately $37,500, the lowest
in the company's history. The company targets further reductions through manufacturing
process improvements, battery cell cost declines, and next-generation platform design.

CAPITAL EXPENDITURES AND CASH

Capital expenditures were $2.46 billion for the quarter. Tesla ended Q3 with
$26.1 billion in cash, cash equivalents, and investments.

SENTIMENT SUMMARY

The quarter reflects mixed sentiment. Volume growth of 27% YoY and the energy storage
record are clearly positive indicators of Tesla's scale and diversification. However,
the automotive gross margin of 17.9% — the lowest in recent years — reflects the
painful trade-off between volume and profitability. Bears cite the margin compression
as structurally concerning, while bulls point to long-term FSD monetization potential.
Cybertruck production ramp adds near-term execution risk but also represents a
significant incremental revenue opportunity.
"""

DOCUMENTS = [
    ("apple_q4_2023_earnings.txt", APPLE_Q4_2023),
    ("microsoft_fy2023_annual.txt", MICROSOFT_FY2023),
    ("tesla_q3_2023_earnings.txt", TESLA_Q3_2023),
]


def main() -> None:
    """Generate sample financial documents in data/raw/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Seeding sample financial documents to {OUTPUT_DIR}/")
    print("-" * 60)

    for filename, content in DOCUMENTS:
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(content.strip())
        word_count = len(content.split())
        print(f"  ✓ {filename} ({word_count} words, {len(content)} chars)")

    print("-" * 60)
    print(f"✅ {len(DOCUMENTS)} documents seeded successfully.")
    print("   Next: python scripts/ingest_documents.py --path data/raw")


if __name__ == "__main__":
    main()
