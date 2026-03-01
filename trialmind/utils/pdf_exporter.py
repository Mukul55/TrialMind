"""
PDF Export Utility for TrialMind Reports.

Converts Markdown analysis output to formatted PDF reports using ReportLab.
PDF reports include:
- TrialMind header with generation metadata
- Formatted analysis sections with headers
- Evidence table with NCT ID links
- Color-coded risk indicators
"""

import re
from datetime import datetime
from io import BytesIO
from loguru import logger

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed — PDF export unavailable")


class ReportExporter:
    """
    Exports TrialMind analysis reports to PDF format.
    """

    BRAND_COLOR = colors.HexColor('#1a365d')  # Dark blue
    ACCENT_COLOR = colors.HexColor('#3182ce')  # Medium blue
    WARNING_COLOR = colors.HexColor('#c05621')  # Orange
    DANGER_COLOR = colors.HexColor('#c53030')  # Red
    SUCCESS_COLOR = colors.HexColor('#276749')  # Green

    def export_to_pdf(self, analysis_result: dict, output_path: str = None) -> bytes:
        """
        Export a TrialMind analysis result to PDF.

        analysis_result: dict from ProtocolAnalyzer.analyze()
        output_path: Optional file path to write PDF to

        Returns: PDF as bytes
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF export. "
                "Install with: pip install reportlab"
            )

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )

        styles = self._build_styles()
        story = self._build_story(analysis_result, styles)

        doc.build(story)

        pdf_bytes = buffer.getvalue()

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)
            logger.info(f"PDF exported to: {output_path}")

        return pdf_bytes

    def _build_styles(self) -> dict:
        """Build custom paragraph styles."""
        base_styles = getSampleStyleSheet()

        styles = {
            'title': ParagraphStyle(
                'TrialMindTitle',
                parent=base_styles['Title'],
                fontSize=24,
                textColor=self.BRAND_COLOR,
                spaceAfter=12
            ),
            'h1': ParagraphStyle(
                'TrialMindH1',
                parent=base_styles['Heading1'],
                fontSize=16,
                textColor=self.BRAND_COLOR,
                spaceBefore=20,
                spaceAfter=8
            ),
            'h2': ParagraphStyle(
                'TrialMindH2',
                parent=base_styles['Heading2'],
                fontSize=13,
                textColor=self.ACCENT_COLOR,
                spaceBefore=14,
                spaceAfter=6
            ),
            'body': ParagraphStyle(
                'TrialMindBody',
                parent=base_styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                leading=15
            ),
            'meta': ParagraphStyle(
                'TrialMindMeta',
                parent=base_styles['Normal'],
                fontSize=9,
                textColor=colors.gray,
                spaceAfter=4
            ),
            'table_header': ParagraphStyle(
                'TableHeader',
                parent=base_styles['Normal'],
                fontSize=9,
                textColor=colors.white,
                fontName='Helvetica-Bold'
            ),
            'table_cell': ParagraphStyle(
                'TableCell',
                parent=base_styles['Normal'],
                fontSize=9
            ),
            'warning': ParagraphStyle(
                'Warning',
                parent=base_styles['Normal'],
                fontSize=10,
                textColor=self.WARNING_COLOR,
                spaceAfter=6
            ),
        }

        return styles

    def _build_story(self, analysis_result: dict, styles: dict) -> list:
        """Build the PDF story (list of flowables)."""
        story = []

        # Title
        story.append(Paragraph("TrialMind Protocol Optimization Report", styles['title']))
        story.append(HRFlowable(width="100%", thickness=2, color=self.BRAND_COLOR))
        story.append(Spacer(1, 12))

        # Metadata
        intent = analysis_result.get('intent', 'unknown').replace('_', ' ').title()
        trial_count = len(analysis_result.get('retrieved_trials', []))
        tokens = analysis_result.get('tokens_used', 0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        meta_lines = [
            f"<b>Generated:</b> {timestamp}",
            f"<b>Query Type:</b> {intent}",
            f"<b>Trials Analyzed:</b> {trial_count:,}",
            f"<b>Tokens Used:</b> {tokens:,}",
        ]
        for line in meta_lines:
            story.append(Paragraph(line, styles['meta']))

        story.append(Spacer(1, 20))

        # Main analysis content
        analysis_text = analysis_result.get('analysis', '')
        story.extend(self._parse_markdown_to_flowables(analysis_text, styles))

        # Evidence table
        candidates = analysis_result.get('retrieved_trials', [])
        if candidates:
            story.append(PageBreak())
            story.append(Paragraph("Supporting Evidence", styles['h1']))
            story.append(self._build_evidence_table(candidates, styles))

        # Footer
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Paragraph(
            "Generated by TrialMind — Evidence-Based Clinical Trial Protocol Optimization",
            styles['meta']
        ))

        return story

    def _parse_markdown_to_flowables(self, text: str, styles: dict) -> list:
        """Convert Markdown text to ReportLab flowables."""
        flowables = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                flowables.append(Spacer(1, 6))
                continue

            # Headers
            if line.startswith('## '):
                flowables.append(Paragraph(line[3:], styles['h2']))
            elif line.startswith('# '):
                flowables.append(Paragraph(line[2:], styles['h1']))
            # Bold lines
            elif line.startswith('**') and line.endswith('**'):
                text_content = line[2:-2]
                flowables.append(Paragraph(f"<b>{text_content}</b>", styles['body']))
            # Bullet points
            elif line.startswith('- ') or line.startswith('• '):
                content = line[2:]
                # Handle inline bold
                content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
                flowables.append(Paragraph(f"&bull; {content}", styles['body']))
            # Table rows (skip — handled separately)
            elif line.startswith('|'):
                pass
            # Regular paragraph
            else:
                # Handle inline bold/italic
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
                # Handle inline code
                line = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', line)
                try:
                    flowables.append(Paragraph(line, styles['body']))
                except Exception:
                    # Fallback for problematic text
                    flowables.append(Paragraph(re.sub(r'<[^>]+>', '', line), styles['body']))

        return flowables

    def _build_evidence_table(self, candidates: list, styles: dict) -> Table:
        """Build a formatted evidence table."""
        header = ['#', 'NCT ID', 'Type', 'Score', 'Indication', 'Phase']
        data = [header]

        for i, candidate in enumerate(candidates[:20], 1):
            meta = candidate.get('metadata', {})
            data.append([
                str(i),
                meta.get('nct_id', 'N/A'),
                meta.get('chunk_type', 'N/A'),
                f"{candidate.get('rerank_score', 0):.2f}",
                (meta.get('conditions_str', 'N/A') or 'N/A')[:40],
                meta.get('phase', 'N/A'),
            ])

        table = Table(data, colWidths=[0.3*inch, 1.2*inch, 1.0*inch, 0.7*inch, 2.5*inch, 1.0*inch])
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), self.BRAND_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            # Data rows
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))

        return table

    def export_to_bytes(self, analysis_result: dict) -> bytes:
        """Export to PDF and return as bytes (for API streaming)."""
        return self.export_to_pdf(analysis_result)
