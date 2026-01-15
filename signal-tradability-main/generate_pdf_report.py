"""
Generate PDF Report: Alpha Decay vs Tradability Research Findings

Creates a comprehensive PDF document with research findings and documentation.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os


def create_pdf_report(output_filename='Alpha_Decay_vs_Tradability_Research.pdf'):
    """
    Create comprehensive PDF report with research findings.
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        leading=12,
        fontName='Courier',
        backColor=colors.HexColor('#f5f5f5'),
        leftIndent=20,
        rightIndent=20
    )
    
    # Title page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Alpha Decay vs Tradability", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Measuring When Surviving Alpha Stops Paying", styles['Title']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Research Report", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Joseph Ahn", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Date: December 26, 2025", styles['Normal']))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading1_style))
    story.append(Paragraph(
        """
        This research addresses a fundamental question in quantitative finance: 
        <b>If a signal still exists statistically, when does it stop being economically tradable?</b>
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        Most quantitative research asks "Does this signal work?" This project asks a different 
        question: "Can the signal still be traded?" This distinction matters because many published 
        signals persist in backtests yet fail in live trading - not because markets are perfectly 
        efficient, but because implementation frictions overwhelm residual edge.
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>Key Finding:</b> Signals can have statistical edge (hit rate > 50%) without economic 
        edge (net Sharpe > 0). The failure mode is cost-driven, not signal falsification.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Core Research Question
    story.append(Paragraph("Core Research Question", heading1_style))
    story.append(Paragraph(
        """
        At what point do transaction costs, slippage, and capacity constraints eliminate the 
        remaining alpha?
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        This reframes the problem from "Is the signal real?" to "Can the signal still be traded?"
        """,
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Key Findings
    story.append(Paragraph("Key Findings", heading1_style))
    
    story.append(Paragraph("1. Statistical vs Economic Edge Distinction", heading2_style))
    story.append(Paragraph(
        """
        Signals can maintain statistical edge (hit rate > 50%, significant returns) while losing 
        economic edge (negative net Sharpe after costs). Mean reversion demonstrates this clearly:
        """,
        body_style
    ))
    
    findings_data = [
        ['Metric', 'Value'],
        ['Statistical Edge', 'Hit Rate: 50.8% (statistically significant)'],
        ['Economic Edge', 'Net Sharpe: -3.883 (untradable)'],
        ['Failure Mode', 'Costs overwhelm edge'],
        ['Annual Turnover', '135.3x'],
        ['Cost Drag', '97.64% annualized'],
        ['Break-even Cost', '0.079% per trade']
    ]
    
    findings_table = Table(findings_data, colWidths=[2*inch, 4*inch])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("2. Mechanism Chain", heading2_style))
    story.append(Paragraph(
        """
        The causal chain from signal characteristics to economic failure is explicitly quantified:
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>Signal Horizon → Turnover → Cost Drag → Sharpe Decay → Capacity Collapse</b>
        """,
        body_style
    ))
    
    mechanism_data = [
        ['Link', 'Mechanism', 'Example (Mean Reversion)'],
        ['1', 'Signal Horizon → Turnover', '3.7 period horizon → 135.3x annual turnover'],
        ['2', 'Turnover → Cost Drag', '135.3x turnover × 0.50% cost = 97.64% drag'],
        ['3', 'Cost Drag → Sharpe Decay', '97.64% drag → Sharpe: 0.542 → -3.883'],
        ['4', 'Capacity Scaling', 'Maximum viable capacity limited by participation rate']
    ]
    
    mechanism_table = Table(mechanism_data, colWidths=[0.5*inch, 2*inch, 3.5*inch])
    mechanism_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(mechanism_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("3. Generalization Claim", heading2_style))
    story.append(Paragraph(
        """
        <b>Primary Claim:</b> Signals with annual turnover above 135x require transaction costs 
        below 0.08% per trade to preserve economic viability. Above this turnover threshold, cost 
        drag overwhelms residual alpha even when statistical edge persists.
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>Formal Condition:</b> Turnover > 135.3 → BreakEvenCost < 0.08%
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>Scope:</b> Simple technical signals with fixed parameters, tested on liquid equity 
        indices (SPY) during 2000-2020.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("4. Turnover-Sharpe Relationship", heading2_style))
    story.append(Paragraph(
        """
        Quantified relationship: A 1x increase in annual turnover implies 0.0653 units of Sharpe 
        decay at 0.50% cost per trade (context-dependent).
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>Formula:</b> Sharpe Decay = -2.2501 + 0.0653 × Turnover
        """,
        code_style
    ))
    story.append(Paragraph(
        """
        <b>Important Caveat:</b> This coefficient is context-dependent and may vary with signal 
        type, market regime, and cost structure. It is a stylized fact from this study, not a 
        universal law.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Add the canonical chart
    story.append(Paragraph("Sharpe Ratio Sensitivity to Transaction Costs", heading2_style))
    chart_path = 'sharpe_sensitivity_chart.png'
    if os.path.exists(chart_path):
        img = Image(chart_path, width=6*inch, height=4.5*inch)
        story.append(Spacer(1, 0.1*inch))
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(
            """
            <i>Mean reversion signal demonstrates extreme cost sensitivity. Statistical edge 
            (Sharpe 0.542 at zero cost) disappears completely at break-even cost of approximately 
            0.08% per trade.</i>
            """,
            ParagraphStyle('Caption', parent=body_style, fontSize=10, textColor=colors.HexColor('#666666'), fontStyle='italic')
        ))
    story.append(PageBreak())
    
    # Methodology
    story.append(Paragraph("Methodology", heading1_style))
    
    story.append(Paragraph("Experimental Design", heading2_style))
    story.append(Paragraph(
        """
        All signal definitions are held constant. No parameter tuning, universe changes, 
        lookback optimization, or leverage adjustments. This ensures that changes in performance 
        are attributable solely to implementation effects.
        """,
        body_style
    ))
    
    story.append(Paragraph("Cost Modeling", heading2_style))
    story.append(Paragraph("Costs are introduced incrementally:", body_style))
    story.append(Paragraph("• Explicit costs: Fixed commissions (0.5% per trade), bid-ask spread (0.1% half-spread)", body_style))
    story.append(Paragraph("• Slippage: Volatility-scaled price impact, volume-scaled execution cost", body_style))
    story.append(Paragraph("• Simple linear models are used deliberately to avoid overfitting", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Capacity Analysis", heading2_style))
    story.append(Paragraph(
        """
        Capacity estimation uses explicit mathematical derivation:
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>Formula:</b> Capacity = Participation_Rate × Average_Daily_Dollar_Volume
        """,
        code_style
    ))
    story.append(Paragraph(
        """
        <b>Assumptions:</b>
        """,
        body_style
    ))
    story.append(Paragraph("• Participation rate: 1% of daily volume (conservative, based on Almgren & Chriss 2000)", body_style))
    story.append(Paragraph("• Linear impact model (simplification; actual is sublinear)", body_style))
    story.append(Paragraph("• Impact coefficient: 10 bps per 1% participation", body_style))
    story.append(Paragraph(
        """
        <b>Limitations:</b> Estimates are rough but defensible. They provide a framework for 
        thinking about scalability, even if precise numbers require more sophisticated modeling.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Results Summary
    story.append(Paragraph("Results Summary", heading1_style))
    
    story.append(Paragraph("Momentum Signal (12-1 Month)", heading2_style))
    momentum_data = [
        ['Metric', 'Gross (Before Costs)', 'Net (After Costs)'],
        ['Sharpe Ratio', '0.102', '-0.005'],
        ['Annualized Return', '2.00%', '-0.10%'],
        ['Hit Rate', '50.7%', '50.5%'],
        ['Max Drawdown', '-54.1%', '-60.2%'],
        ['Annual Turnover', '3.0x', '3.0x'],
        ['Cost Drag', 'N/A', '2.10%'],
        ['Break-even Cost', 'N/A', '0.672% per trade']
    ]
    
    momentum_table = Table(momentum_data, colWidths=[2*inch, 2*inch, 2*inch])
    momentum_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(momentum_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Mean Reversion Signal", heading2_style))
    story.append(Paragraph(
        """
        Mean reversion demonstrates the core phenomenon most clearly: statistical edge exists 
        (50.8% hit rate) but economic edge is eliminated (net Sharpe: -3.883) due to high 
        turnover (135.3x annual) and resulting cost drag (97.64%).
        """,
        body_style
    ))
    story.append(PageBreak())
    
    # Implications
    story.append(Paragraph("Implications", heading1_style))
    
    story.append(Paragraph("Who This Research Critiques", heading2_style))
    story.append(Paragraph(
        """
        <b>1. Retail Quants Who Believe Hit Rate Implies Profitability</b>
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        Our results show hit rates of 50.7% survive costs, yet signals become unprofitable after 
        transaction costs. Directional correctness ≠ profitability.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        """
        <b>2. Academic Papers That Ignore Implementability</b>
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        Signals can have statistical edge (Sharpe > 0, hit rate > 50%) but zero economic edge 
        when costs are included. Academic backtests without cost modeling are incomplete.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        """
        <b>3. Backtests That Implicitly Assume Zero Friction</b>
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        Cost drag can exceed gross returns (2.10% drag vs 2.00% gross return for momentum). 
        Realistic retail costs (0.5% commission + 0.1% spread) eliminate edge for many signals.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph(
        """
        <b>4. Systematic Traders Who Optimize for Gross Sharpe</b>
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        Gross-to-net Sharpe decay of -104.9% (momentum signal). Gross Sharpe: 0.102 → Net Sharpe: 
        -0.005. Gross Sharpe is not a reliable proxy for tradability.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # What This Research Does NOT Claim
    story.append(Paragraph("What This Research Does NOT Claim", heading1_style))
    story.append(Paragraph(
        """
        This project explicitly does not claim that any signal is profitable today, that alpha 
        persists forever, that backtested results imply tradability, or that optimization can 
        rescue decayed signals.
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        This research also does not claim that all signals are untradable, that costs always 
        eliminate alpha, that backtests are useless, or that markets are perfectly efficient.
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>What it DOES claim:</b> Signals can have statistical edge without economic edge. 
        Turnover is the primary mechanism for cost-driven failure. Hit rate persistence does not 
        imply profitability. Gross performance metrics are misleading without cost modeling.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Limitations
    story.append(Paragraph("Scope and Limitations", heading1_style))
    story.append(Paragraph(
        """
        <b>Important:</b> All empirical relationships, thresholds, and coefficients in this research 
        are context-dependent. They are valid within our study's scope but should not be generalized 
        beyond tested conditions without additional validation.
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>Scope:</b> Simple technical signals with fixed parameters (no optimization), tested on 
        liquid equity indices (SPY) during 2000-2020.
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        <b>Limitations:</b> Cost models are simplified. Capacity estimates are rough but defensible. 
        Turnover-Sharpe relationship coefficients are context-dependent, not universal. See 
        LIMITATIONS.md for detailed scope restrictions.
        """,
        body_style
    ))
    story.append(PageBreak())
    
    # Conclusion
    story.append(Paragraph("Conclusion", heading1_style))
    story.append(Paragraph(
        """
        This research demonstrates that alpha does not disappear at discovery - it disappears when 
        implementation costs overwhelm what remains.
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        Signals can maintain statistical edge (hit rate > 50%) while losing economic edge 
        (negative net Sharpe after costs). The failure mode is cost-driven, not signal 
        falsification.
        """,
        body_style
    ))
    story.append(Paragraph(
        """
        Understanding this gap between statistical and economic viability is the real bottleneck 
        in quantitative finance. This research provides a framework for measuring and explaining 
        this gap.
        """,
        body_style
    ))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("One-Line Summary", heading2_style))
    story.append(Paragraph(
        """
        <i>Alpha does not disappear at discovery. It disappears when implementation costs 
        overwhelm what remains. Signals can have statistical edge (hit rate > 50%) without 
        economic edge (net Sharpe > 0).</i>
        """,
        body_style
    ))
    story.append(Spacer(1, 0.5*inch))
    
    
    # Build PDF
    doc.build(story)
    print(f"PDF report generated: {output_filename}")
    return output_filename


if __name__ == "__main__":
    try:
        output_file = create_pdf_report()
        print(f"\nSuccessfully created PDF: {output_file}")
        print(f"  Location: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("\nNote: This script requires 'reportlab' library.")
        print("Install it with: pip install reportlab")
        import traceback
        traceback.print_exc()

