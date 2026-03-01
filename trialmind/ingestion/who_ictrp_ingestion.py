"""
WHO International Clinical Trials Registry Platform (ICTRP) ingestion.

The WHO ICTRP aggregates trial registries from around the world including:
- ANZCTR (Australia/New Zealand)
- ChiCTR (China)
- CTRI (India)
- DRKS (Germany)
- EudraCT (European Union)
- IRCT (Iran)
- ISRCTN (International)
- JapicCTI (Japan)
- JMACCT (Japan)
- NTR (Netherlands)
- PACTR (Pan-African)
- ReBec (Brazil)
- RPCEC (Cuba)
- SLCTR (Sri Lanka)
- TCTR (Thailand)

This is valuable for global enrollment benchmarks, especially for
non-US centric trials and rare disease registries.

Search API: https://trialsearch.who.int/
"""

import aiohttp
import asyncio
import xml.etree.ElementTree as ET
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import re

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import WHO_RATE_LIMIT


class WHOICTRPIngestion:
    """
    Ingests trial data from the WHO ICTRP search portal.
    Focuses on trials NOT registered on ClinicalTrials.gov to supplement AACT.
    """

    SEARCH_BASE = "https://trialsearch.who.int/Trial2.aspx"
    API_BASE = "https://trialsearch.who.int"

    # Therapeutic areas with high non-US trial activity
    SEARCH_QUERIES = [
        "phase 3 cancer randomized",
        "phase 3 cardiovascular randomized",
        "phase 2 oncology endpoint",
        "phase 3 diabetes randomized controlled",
        "rare disease phase 2 phase 3",
        "phase 3 neurology randomized",
        "infectious disease phase 3 randomized",
    ]

    def __init__(self):
        self.rate_limit = WHO_RATE_LIMIT
        self.semaphore = asyncio.Semaphore(self.rate_limit)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30)
    )
    async def search_trials(
        self,
        session: aiohttp.ClientSession,
        query: str,
        page: int = 1
    ) -> list:
        """
        Search WHO ICTRP for trials matching a query.
        Returns list of trial records.
        """
        params = {
            "query": query,
            "page": page,
        }

        async with self.semaphore:
            try:
                async with session.get(
                    self.SEARCH_BASE,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"WHO ICTRP returned {resp.status}")
                        return []
                    content = await resp.text()
                    return self._parse_who_response(content, query)
            except Exception as e:
                logger.error(f"WHO ICTRP request error: {e}")
                return []

    def _parse_who_response(self, content: str, query: str) -> list:
        """
        Parse WHO ICTRP HTML/XML response.
        Extracts key trial fields for supplemental data.
        """
        records = []

        # Extract trial IDs and basic info using regex patterns
        # WHO ICTRP uses mixed formats across registries
        trial_id_pattern = r'(ACTRN\d+|ChiCTR-\w+-\d+|CTRI/\d+/\d+/\d+|DRKS\d+|EUCTR\d+-\d+-\w+|IRCT\d+N\d+|ISRCTN\d+|JapicCTI-\d+|NTR\d+|PACTR\d+|RBR-\w+|RPCEC\d+|SLCTR/\d+/\d+|TCTR\d+)'
        trial_ids = re.findall(trial_id_pattern, content)

        # Extract enrollment figures
        enrollment_pattern = r'(?:enrollment|sample size|target number)[:\s]+(\d+)'
        enrollments = re.findall(enrollment_pattern, content.lower())

        # Extract phases
        phase_pattern = r'phase\s+([1-4](?:/[1-4])?)'
        phases = re.findall(phase_pattern, content.lower())

        for i, trial_id in enumerate(trial_ids[:50]):  # Cap per query
            record = {
                "id": f"who_{trial_id}",
                "trial_id": trial_id,
                "source": "who_ictrp",
                "query_context": query,
                "estimated_enrollment": int(enrollments[i]) if i < len(enrollments) else None,
                "phase": f"Phase {phases[i]}" if i < len(phases) else None,
                "text": f"""WHO ICTRP TRIAL RECORD
Trial ID: {trial_id}
Registry: {self._identify_registry(trial_id)}
Search Context: {query}
Estimated Enrollment: {enrollments[i] if i < len(enrollments) else 'Not specified'}
Phase: {f"Phase {phases[i]}" if i < len(phases) else 'Not specified'}
Source: WHO International Clinical Trials Registry Platform""",
                "metadata": {
                    "trial_id": trial_id,
                    "registry": self._identify_registry(trial_id),
                    "source": "who_ictrp",
                    "query_context": query[:100]
                }
            }
            records.append(record)

        return records

    def _identify_registry(self, trial_id: str) -> str:
        """Identify which registry a trial ID belongs to."""
        prefixes = {
            'ACTRN': 'ANZCTR (Australia/New Zealand)',
            'ChiCTR': 'ChiCTR (China)',
            'CTRI': 'CTRI (India)',
            'DRKS': 'DRKS (Germany)',
            'EUCTR': 'EudraCT (European Union)',
            'IRCT': 'IRCT (Iran)',
            'ISRCTN': 'ISRCTN (International)',
            'Japic': 'JapicCTI (Japan)',
            'NTR': 'NTR (Netherlands)',
            'PACTR': 'PACTR (Pan-Africa)',
            'RBR': 'ReBec (Brazil)',
            'RPCEC': 'RPCEC (Cuba)',
            'SLCTR': 'SLCTR (Sri Lanka)',
            'TCTR': 'TCTR (Thailand)',
        }
        for prefix, name in prefixes.items():
            if trial_id.startswith(prefix):
                return name
        return 'Unknown registry'

    async def run_ingestion(self, max_per_query: int = 200) -> list:
        """
        Main WHO ICTRP ingestion loop.
        """
        all_records = {}

        async with aiohttp.ClientSession() as session:
            for query in self.SEARCH_QUERIES:
                logger.info(f"WHO ICTRP search: {query}")

                records = await self.search_trials(session, query)

                for rec in records:
                    trial_id = rec['trial_id']
                    if trial_id not in all_records:
                        all_records[trial_id] = rec

                await asyncio.sleep(2.0 / self.rate_limit)

        unique_records = list(all_records.values())
        logger.info(f"WHO ICTRP ingestion complete: {len(unique_records)} unique trials")
        return unique_records
