"""
PubMed ingestion for published clinical trial results.

This supplements AACT structured data with narrative result information.
Published papers contain:
- Actual efficacy data (hazard ratios, response rates, p-values)
- Subgroup analyses
- Reasons for trial failure
- Lessons learned commentary from investigators

Free API — register at ncbi.nlm.nih.gov/account for higher rate limits.
"""

import re
import aiohttp
import asyncio
import xml.etree.ElementTree as ET
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import NCBI_API_KEY, PUBMED_RATE_LIMIT, MIN_ABSTRACT_LENGTH


class PubMedIngestion:

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    TRIAL_RESULT_QUERIES = [
        # Oncology
        "phase 3 clinical trial results oncology randomized",
        "phase 2 clinical trial results cancer randomized controlled",
        "clinical trial failure oncology why stopped early terminated",

        # Cardiovascular
        "phase 3 randomized controlled trial cardiovascular outcomes",
        "clinical trial heart failure results primary endpoint",

        # Metabolic / Endocrine
        "phase 3 diabetes clinical trial results glycemic control",
        "obesity clinical trial results phase 3 randomized",

        # Neurology / CNS
        "phase 3 neurology clinical trial results alzheimer parkinson",
        "CNS clinical trial failure terminated stopped early",

        # Immunology / Autoimmune
        "phase 3 rheumatoid arthritis clinical trial results",
        "autoimmune disease clinical trial results randomized",

        # Rare Disease
        "orphan drug clinical trial phase 2 phase 3 results",
        "rare disease trial enrollment challenges recruitment",

        # Respiratory
        "phase 3 asthma COPD clinical trial results randomized",

        # Infectious Disease
        "phase 3 infectious disease clinical trial results antibiotic antiviral",

        # Protocol design and amendments
        "clinical trial protocol amendment reasons analysis",
        "clinical trial early termination enrollment failure reasons",
        "trial design failure lessons learned protocol optimization"
    ]

    def __init__(self):
        self.api_key = NCBI_API_KEY
        self.rate_limit = PUBMED_RATE_LIMIT
        self.semaphore = asyncio.Semaphore(self.rate_limit)

    async def search_pmids(
        self, session: aiohttp.ClientSession, query: str,
        max_results: int = 500
    ) -> list:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "datetype": "pdat",
            "mindate": "2010",
            "maxdate": "2024"
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with self.semaphore:
            async with session.get(
                f"{self.BASE_URL}/esearch.fcgi", params=params
            ) as response:
                data = await response.json()
                return data.get("esearchresult", {}).get("idlist", [])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def fetch_abstracts(
        self, session: aiohttp.ClientSession, pmids: list
    ) -> list:
        """
        Fetch full abstract data for a batch of PMIDs.
        Parses XML to extract: title, abstract, authors, MeSH terms, publication type.
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml"
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with self.semaphore:
            async with session.get(
                f"{self.BASE_URL}/efetch.fcgi", params=params
            ) as response:
                xml_content = await response.text()

        return self._parse_pubmed_xml(xml_content)

    def _parse_pubmed_xml(self, xml_content: str) -> list:
        """Parse PubMed XML response into structured records."""
        records = []
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return []

        for article in root.findall('.//PubmedArticle'):
            try:
                pmid = article.findtext('.//PMID', default='')
                title = article.findtext('.//ArticleTitle', default='')

                abstract_texts = []
                abstract_element = article.find('.//Abstract')
                if abstract_element is not None:
                    for text_el in abstract_element.findall('AbstractText'):
                        label = text_el.get('Label', '')
                        text = text_el.text or ''
                        if label:
                            abstract_texts.append(f"{label}: {text}")
                        else:
                            abstract_texts.append(text)
                abstract = " ".join(abstract_texts)

                year = article.findtext('.//PubDate/Year', default='')
                if not year:
                    year = article.findtext('.//PubDate/MedlineDate', default='')[:4]

                journal = article.findtext('.//Journal/Title', default='')

                pub_types = [
                    pt.text for pt in article.findall('.//PublicationType')
                    if pt.text
                ]

                mesh_terms = [
                    mh.findtext('DescriptorName', default='')
                    for mh in article.findall('.//MeshHeading')
                ]

                nct_matches = re.findall(r'NCT\d{8}', abstract + title)

                is_trial_result = any(
                    t in pub_types for t in [
                        'Clinical Trial', 'Clinical Trial, Phase I',
                        'Clinical Trial, Phase II', 'Clinical Trial, Phase III',
                        'Clinical Trial, Phase IV', 'Randomized Controlled Trial',
                        'Controlled Clinical Trial'
                    ]
                ) or any(
                    kw in title.lower() for kw in [
                        'phase 1', 'phase 2', 'phase 3', 'randomized',
                        'randomised', 'clinical trial', 'results of'
                    ]
                )

                if not is_trial_result:
                    continue

                if len(abstract) < MIN_ABSTRACT_LENGTH:
                    continue

                record = {
                    "id": f"pubmed_{pmid}",
                    "pmid": pmid,
                    "source": "pubmed",
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "journal": journal,
                    "pub_types": pub_types,
                    "mesh_terms": mesh_terms,
                    "mesh_terms_str": "; ".join(mesh_terms),
                    "nct_references": nct_matches,
                    "text": f"Title: {title}\n\nAbstract: {abstract}"
                }
                records.append(record)

            except Exception as e:
                logger.error(f"Error parsing article {pmid}: {e}")
                continue

        return records

    async def run_ingestion(self, max_per_query: int = 500) -> list:
        """
        Main ingestion loop. Queries PubMed with all trial-related queries
        and returns deduplicated list of abstract records.
        """
        all_records = {}  # pmid -> record (for deduplication)

        async with aiohttp.ClientSession() as session:
            for query in self.TRIAL_RESULT_QUERIES:
                logger.info(f"Searching PubMed: {query[:60]}...")

                pmids = await self.search_pmids(session, query, max_per_query)
                logger.info(f"Found {len(pmids)} PMIDs for query")

                for i in range(0, len(pmids), 100):
                    batch = pmids[i:i+100]
                    records = await self.fetch_abstracts(session, batch)

                    for rec in records:
                        if rec['pmid'] not in all_records:
                            all_records[rec['pmid']] = rec

                    await asyncio.sleep(1.0 / self.rate_limit)

        unique_records = list(all_records.values())
        logger.info(f"PubMed ingestion complete: {len(unique_records)} unique trial result abstracts")
        return unique_records
