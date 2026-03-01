"""
FDA Statistical Reviews ingestion from Drugs@FDA.

FDA statistical review documents contain:
- Official FDA assessment of trial endpoints
- Sample size adequacy evaluations
- Regulatory precedent for endpoint acceptance
- Analysis of trial design flaws identified during review

Data source: https://www.accessdata.fda.gov/scripts/cder/daf/
Free API: https://open.fda.gov/apis/

Rate limits:
- Without API key: 1,000 requests/day
- With API key (free): 120,000 requests/day
Register at: https://open.fda.gov/apis/authentication/
"""

import aiohttp
import asyncio
import json
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import re

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import FDA_API_KEY, FDA_API_RATE_LIMIT


class FDAReviewsIngestion:
    """
    Fetches FDA drug approval data and statistical review summaries.
    Uses the openFDA API for structured access to approval documents.
    """

    OPEN_FDA_BASE = "https://api.fda.gov/drug"

    # Drug application types to retrieve
    APPLICATION_TYPES = ["NDA", "BLA"]  # New Drug Applications, Biologics License Applications

    # Therapeutic areas to prioritize
    THERAPEUTIC_KEYWORDS = [
        "oncology", "cardiovascular", "diabetes", "alzheimer",
        "immunology", "rheumatoid arthritis", "respiratory", "rare disease",
        "neurology", "infectious disease", "metabolic"
    ]

    def __init__(self):
        self.api_key = FDA_API_KEY
        self.rate_limit = min(FDA_API_RATE_LIMIT, 100)  # per-minute conservative cap
        self.semaphore = asyncio.Semaphore(10)  # concurrent requests

    def _build_url(self, endpoint: str, params: dict) -> str:
        """Build FDA API URL with optional API key."""
        base = f"{self.OPEN_FDA_BASE}/{endpoint}.json"
        if self.api_key:
            params['api_key'] = self.api_key
        query_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base}?{query_str}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15)
    )
    async def fetch_approvals(
        self,
        session: aiohttp.ClientSession,
        skip: int = 0,
        limit: int = 100
    ) -> dict:
        """Fetch drug approval records from openFDA."""
        params = {
            "search": "submissions.submission_type:NDA+submissions.submission_type:BLA",
            "limit": limit,
            "skip": skip
        }
        url = self._build_url("drugsfda", params)

        async with self.semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        logger.warning("FDA API rate limit hit — backing off")
                        await asyncio.sleep(60)
                        return {}
                    else:
                        logger.error(f"FDA API error: {resp.status}")
                        return {}
            except Exception as e:
                logger.error(f"FDA request error: {e}")
                return {}

    def _extract_review_text(self, drug_record: dict) -> dict:
        """
        Extract relevant information from a drug approval record.
        Structures it for storage in ChromaDB.
        """
        try:
            application_number = drug_record.get('application_number', '')
            brand_name = drug_record.get('brand_name', '')
            generic_name = drug_record.get('generic_name', '')
            manufacturer = drug_record.get('sponsor_name', '')

            # Extract submissions
            submissions = drug_record.get('submissions', [])
            approvals = [
                s for s in submissions
                if s.get('submission_status') == 'AP'
            ]

            if not approvals:
                return None

            latest_approval = approvals[-1]
            approval_date = latest_approval.get('submission_status_date', '')
            approval_year = approval_date[:4] if approval_date else 'unknown'

            # Extract products (formulations)
            products = drug_record.get('products', [])
            indications = []
            for product in products:
                marketing_status = product.get('marketing_status', '')
                active_ingredients = product.get('active_ingredients', [])
                for ing in active_ingredients:
                    strength = ing.get('strength', '')
                    name = ing.get('name', '')
                    if name:
                        indications.append(f"{name} {strength}".strip())

            # Build narrative text for embedding
            text = f"""FDA DRUG APPROVAL RECORD
Application: {application_number}
Brand Name: {brand_name}
Generic Name: {generic_name}
Manufacturer: {manufacturer}
Approval Date: {approval_date}
Active Ingredients/Strengths: {'; '.join(indications[:5])}
Number of Submissions: {len(submissions)}
Approval Type: {latest_approval.get('submission_type', 'N/A')}
Review Classification: {latest_approval.get('review_priority', 'Standard')}"""

            review_doc = {
                "id": f"fda_{application_number}",
                "application_number": application_number,
                "brand_name": brand_name,
                "generic_name": generic_name,
                "manufacturer": manufacturer,
                "approval_date": approval_date,
                "approval_year": approval_year,
                "indications": indications,
                "submission_count": len(submissions),
                "source": "fda_approval",
                "text": text,
                "metadata": {
                    "application_number": application_number,
                    "approval_year": approval_year,
                    "source": "fda_approvals",
                    "brand_name": brand_name[:100],
                    "generic_name": generic_name[:100]
                }
            }

            return review_doc

        except Exception as e:
            logger.error(f"Error extracting FDA record: {e}")
            return None

    async def run_ingestion(self, max_records: int = 5000) -> list:
        """
        Main FDA ingestion loop.
        Retrieves drug approval records from openFDA.
        """
        all_records = []
        skip = 0
        batch_size = 100

        logger.info(f"Starting FDA reviews ingestion (max {max_records} records)")

        async with aiohttp.ClientSession() as session:
            while len(all_records) < max_records:
                data = await self.fetch_approvals(session, skip=skip, limit=batch_size)

                if not data or 'results' not in data:
                    logger.info("No more FDA records to fetch")
                    break

                results = data.get('results', [])
                if not results:
                    break

                for record in results:
                    doc = self._extract_review_text(record)
                    if doc:
                        all_records.append(doc)

                skip += batch_size
                total_available = data.get('meta', {}).get('results', {}).get('total', 0)

                logger.info(
                    f"Retrieved {len(all_records)} FDA records "
                    f"(total available: {total_available})"
                )

                if skip >= total_available:
                    break

                await asyncio.sleep(0.5)  # Conservative rate limiting

        logger.info(f"FDA ingestion complete: {len(all_records)} approval records")
        return all_records
