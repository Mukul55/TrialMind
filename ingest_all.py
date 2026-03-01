"""
Full ingestion pipeline orchestrator.
Run this once to populate the TrialMind vector store.

This will take several hours depending on internet speed and hardware.
The pipeline is resumable — if interrupted, re-run and it will skip
already-processed records.

Estimated data sizes after ingestion:
- AACT trials: ~80,000-100,000 trial profiles (filtered from 450k+)
- Chunked documents: ~500,000 total chunks across 5 collections
- PubMed abstracts: ~50,000-100,000 relevant trial result abstracts
- ChromaDB size: approximately 2-4GB on disk

Prerequisites:
1. pip install -r requirements.txt
2. python -m spacy download en_core_sci_lg
3. cp .env.example .env && edit .env with your credentials
4. Register free AACT account: https://aact.ctti-clinicaltrials.org/users/sign_up
"""

import sys
import os
import asyncio

# Add trialmind to path
sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger
from tqdm import tqdm


def run_pipeline(
    start_year: int = 2010,
    batch_size: int = 1000,
    skip_aact: bool = False,
    skip_pubmed: bool = False,
    skip_fda: bool = False,
    max_pubmed_per_query: int = 500
):
    """
    Run the complete TrialMind ingestion pipeline.

    Args:
        start_year: Only include trials starting from this year
        batch_size: AACT processing batch size
        skip_aact: Skip AACT ingestion (use if already indexed)
        skip_pubmed: Skip PubMed ingestion
        skip_fda: Skip FDA reviews ingestion
        max_pubmed_per_query: Max PubMed results per search query
    """
    logger.info("=" * 70)
    logger.info("TRIALMIND INGESTION PIPELINE STARTING")
    logger.info("=" * 70)
    logger.info(f"Configuration: start_year={start_year}, batch_size={batch_size}")

    # Initialize vector store (creates ChromaDB collections)
    from trialmind.processing.embedder import TrialMindVectorStore
    from trialmind.processing.chunker import TrialChunker

    vector_store = TrialMindVectorStore()
    chunker = TrialChunker()

    # ── STAGE 1: AACT Database Ingestion ──────────────────────────────────────
    profiles = []

    if not skip_aact:
        logger.info("\n[STAGE 1] AACT Database Ingestion")
        logger.info("Connecting to aact-db.ctti-clinicaltrials.org...")
        logger.info("Register free at: https://aact.ctti-clinicaltrials.org/users/sign_up")

        from trialmind.ingestion.aact_ingestion import AACTIngestion
        aact = AACTIngestion()

        try:
            profiles = aact.run_full_ingestion(
                batch_size=batch_size,
                start_year=start_year
            )
            logger.info(f"AACT ingestion complete: {len(profiles)} profiles")
        except Exception as e:
            logger.error(f"AACT ingestion failed: {e}")
            logger.warning(
                "Continuing without AACT data.\n"
                "To fix: Register at https://aact.ctti-clinicaltrials.org/users/sign_up\n"
                "Then add credentials to your .env file:\n"
                "  AACT_DB_USER=your_username\n"
                "  AACT_DB_PASS=your_password"
            )
    else:
        logger.info("[STAGE 1] Skipping AACT ingestion (--skip-aact flag)")

    # ── STAGE 2: Chunking ──────────────────────────────────────────────────────
    if profiles:
        logger.info(f"\n[STAGE 2] Creating specialized chunks for {len(profiles)} trial profiles")

        all_chunks = []
        for profile in tqdm(profiles, desc="Chunking profiles"):
            chunks = chunker.create_all_chunks(profile)
            if chunks:
                all_chunks.append(chunks)

        logger.info(f"Created chunk sets for {len(all_chunks)} trials")

        # ── STAGE 3: Embedding + ChromaDB ─────────────────────────────────────
        logger.info(f"\n[STAGE 3] Embedding and storing {len(all_chunks)} trial chunk sets in ChromaDB")
        vector_store.add_trial_chunks(all_chunks)
        logger.info("Trial profile chunks stored in ChromaDB")
    else:
        logger.info("\n[STAGE 2-3] No AACT profiles to chunk/embed — skipping")

    # ── STAGE 4: PubMed Ingestion ──────────────────────────────────────────────
    if not skip_pubmed:
        logger.info("\n[STAGE 4] PubMed Trial Results Ingestion")
        logger.info("Querying PubMed for clinical trial result abstracts...")
        logger.info(
            "Note: NCBI_API_KEY in .env increases rate limit from 3 to 10 req/s\n"
            "Free registration: https://ncbi.nlm.nih.gov/account"
        )

        from trialmind.ingestion.pubmed_ingestion import PubMedIngestion
        pubmed = PubMedIngestion()

        try:
            pubmed_records = asyncio.run(
                pubmed.run_ingestion(max_per_query=max_pubmed_per_query)
            )
            vector_store.add_pubmed_records(pubmed_records)
            logger.info(f"PubMed ingestion complete: {len(pubmed_records)} abstracts indexed")
        except Exception as e:
            logger.error(f"PubMed ingestion error: {e}")
    else:
        logger.info("[STAGE 4] Skipping PubMed ingestion (--skip-pubmed flag)")

    # ── STAGE 5: FDA Reviews Ingestion ────────────────────────────────────────
    if not skip_fda:
        logger.info("\n[STAGE 5] FDA Drug Approval Records Ingestion")
        logger.info(
            "Note: FDA_API_KEY in .env increases rate limit to 120,000/day\n"
            "Free registration: https://open.fda.gov/apis/authentication/"
        )

        from trialmind.ingestion.fda_reviews_ingestion import FDAReviewsIngestion
        fda = FDAReviewsIngestion()

        try:
            fda_records = asyncio.run(fda.run_ingestion(max_records=5000))
            vector_store.add_fda_records(fda_records)
            logger.info(f"FDA ingestion complete: {len(fda_records)} approval records indexed")
        except Exception as e:
            logger.error(f"FDA ingestion error: {e}")
    else:
        logger.info("[STAGE 5] Skipping FDA ingestion (--skip-fda flag)")

    # ── Final Status ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("INGESTION PIPELINE COMPLETE")
    logger.info("\nCollection sizes:")
    total_docs = 0
    for name, coll in vector_store.collections.items():
        count = coll.count()
        total_docs += count
        logger.info(f"  {name:<30} {count:>8,} documents")

    logger.info(f"\n  {'TOTAL':<30} {total_docs:>8,} documents")
    logger.info("=" * 70)

    if total_docs == 0:
        logger.warning(
            "\n⚠️  No documents indexed. Check:\n"
            "  1. AACT credentials in .env (AACT_DB_USER, AACT_DB_PASS)\n"
            "  2. Network connectivity to aact-db.ctti-clinicaltrials.org\n"
            "  3. PubMed/FDA connectivity\n"
            "\nOnce resolved, re-run: python ingest_all.py"
        )
    else:
        logger.info("\n✅ TrialMind is ready! Next steps:")
        logger.info("   Start API:  uvicorn trialmind.api.main:app --reload --port 8000")
        logger.info("   Start UI:   streamlit run trialmind/ui/app.py")
        logger.info("   Run eval:   python trialmind/evaluation/evaluator.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TrialMind data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_all.py                          # Full ingestion
  python ingest_all.py --start-year 2015        # Only trials from 2015+
  python ingest_all.py --skip-aact              # Skip AACT, only PubMed+FDA
  python ingest_all.py --skip-pubmed --skip-fda # Only AACT
        """
    )

    parser.add_argument(
        "--start-year", type=int, default=2010,
        help="Only include trials starting from this year (default: 2010)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="AACT processing batch size (default: 1000)"
    )
    parser.add_argument(
        "--skip-aact", action="store_true",
        help="Skip AACT database ingestion"
    )
    parser.add_argument(
        "--skip-pubmed", action="store_true",
        help="Skip PubMed ingestion"
    )
    parser.add_argument(
        "--skip-fda", action="store_true",
        help="Skip FDA reviews ingestion"
    )
    parser.add_argument(
        "--max-pubmed", type=int, default=500,
        help="Max PubMed results per search query (default: 500)"
    )

    args = parser.parse_args()

    run_pipeline(
        start_year=args.start_year,
        batch_size=args.batch_size,
        skip_aact=args.skip_aact,
        skip_pubmed=args.skip_pubmed,
        skip_fda=args.skip_fda,
        max_pubmed_per_query=args.max_pubmed
    )
