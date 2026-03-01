"""
AACT (Aggregate Analysis of ClinicalTrials.gov) Ingestion Pipeline

Registration: https://aact.ctti-clinicaltrials.org/users/sign_up (free)
Documentation: https://aact.ctti-clinicaltrials.org/schema

The AACT database has 67 tables. We use the following critical tables:
- studies: core trial metadata
- eligibilities: inclusion/exclusion criteria
- interventions: drug/treatment information
- outcomes: primary and secondary endpoints
- result_groups: enrollment by arm
- calculated_values: derived metrics including enrollment
- countries: trial site countries
- facilities: individual trial sites
- drop_withdrawals: dropout data
- reported_events: adverse events from trials
- design_outcomes: outcome measure details
- brief_summaries: trial summaries
"""

import re
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from tqdm import tqdm
import json
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    AACT_DB_HOST, AACT_DB_PORT, AACT_DB_NAME, AACT_DB_USER, AACT_DB_PASS
)


class AACTIngestion:
    """
    Connects to the AACT PostgreSQL database and extracts structured
    trial profile data optimized for RAG retrieval.
    """

    def __init__(self):
        self.engine = None
        self.conn = None

    def connect(self):
        """
        Establish connection to AACT. Handles connection errors gracefully
        with informative messages about free account registration.
        """
        connection_string = (
            f"postgresql://{AACT_DB_USER}:{AACT_DB_PASS}"
            f"@{AACT_DB_HOST}:{AACT_DB_PORT}/{AACT_DB_NAME}"
        )
        try:
            self.engine = create_engine(connection_string)
            self.conn = self.engine.connect()
            logger.info("Connected to AACT database successfully")
        except Exception as e:
            logger.error(
                f"AACT connection failed: {e}\n"
                "Register for a free account at: "
                "https://aact.ctti-clinicaltrials.org/users/sign_up"
            )
            raise

    def extract_core_trials(
        self,
        therapeutic_areas: list = None,
        phases: list = None,
        start_year: int = 2010,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Extract core trial data from the studies table with intelligent filtering.

        Returns a DataFrame of trial records filtered by:
        - Phase (1, 2, 3, 4)
        - Study type (INTERVENTIONAL only — we don't want observational studies)
        - Completion status (completed or terminated — both provide learning signal)
        - Date range (default: 2010 to present)
        - Therapeutic area (optional MeSH condition filter)
        """
        base_query = """
        SELECT
            s.nct_id,
            s.brief_title,
            s.official_title,
            s.overall_status,
            s.phase,
            s.study_type,
            s.start_date,
            s.completion_date,
            s.primary_completion_date,
            s.enrollment,
            s.enrollment_type,
            s.why_stopped,
            s.has_expanded_access,
            s.is_fda_regulated_drug,
            s.is_fda_regulated_device,
            s.number_of_arms,
            s.number_of_groups,
            s.allocation,
            s.intervention_model,
            s.masking,
            s.primary_purpose,
            s.source,
            bs.description as brief_summary,
            ds.description as detailed_description
        FROM studies s
        LEFT JOIN brief_summaries bs ON s.nct_id = bs.nct_id
        LEFT JOIN detailed_descriptions ds ON s.nct_id = ds.nct_id
        WHERE
            s.study_type = 'INTERVENTIONAL'
            AND s.overall_status IN ('Completed', 'Terminated', 'Suspended',
                                     'Active, not recruiting')
            AND s.phase IN ('Phase 1', 'Phase 2', 'Phase 3', 'Phase 4',
                           'Phase 1/Phase 2', 'Phase 2/Phase 3')
            AND s.is_fda_regulated_drug = TRUE
            AND EXTRACT(YEAR FROM s.start_date) >= :start_year
        """

        params = {"start_year": start_year}

        if limit:
            base_query += f" LIMIT {limit}"

        logger.info(f"Extracting core trial data from AACT (start_year={start_year})...")
        df = pd.read_sql(text(base_query), self.conn, params=params)
        logger.info(f"Extracted {len(df)} trials from AACT studies table")
        return df

    def extract_eligibility_criteria(self, nct_ids: list) -> pd.DataFrame:
        """
        Extract inclusion/exclusion criteria for a list of trials.

        The eligibilities table stores criteria as a single text block.
        We parse this into individual criteria items.
        """
        query = """
        SELECT
            nct_id,
            gender,
            minimum_age,
            maximum_age,
            healthy_volunteers,
            criteria
        FROM eligibilities
        WHERE nct_id = ANY(:nct_ids)
        """
        df = pd.read_sql(
            text(query), self.conn,
            params={"nct_ids": nct_ids}
        )

        # Parse criteria text into inclusion/exclusion lists
        df['inclusion_criteria'] = df['criteria'].apply(
            self._parse_inclusion_criteria
        )
        df['exclusion_criteria'] = df['criteria'].apply(
            self._parse_exclusion_criteria
        )
        return df

    def _parse_inclusion_criteria(self, criteria_text: str) -> list:
        """
        Parse the criteria text block to extract inclusion criteria.
        AACT stores criteria as unstructured text with headers like
        'Inclusion Criteria:' and 'Exclusion Criteria:'.
        """
        if not criteria_text:
            return []

        incl_pattern = r'(?i)inclusion criteria[:\s]*(.*?)(?=exclusion criteria|$)'
        match = re.search(incl_pattern, criteria_text, re.DOTALL)
        if not match:
            return []

        section = match.group(1)
        items = re.split(r'\n\s*[-•*\d+\.]\s+', section)
        items = [i.strip() for i in items if len(i.strip()) > 10]
        return items[:20]  # Cap at 20 criteria per trial

    def _parse_exclusion_criteria(self, criteria_text: str) -> list:
        """Mirror of _parse_inclusion_criteria for exclusion section."""
        if not criteria_text:
            return []
        excl_pattern = r'(?i)exclusion criteria[:\s]*(.*?)$'
        match = re.search(excl_pattern, criteria_text, re.DOTALL)
        if not match:
            return []

        section = match.group(1)
        items = re.split(r'\n\s*[-•*\d+\.]\s+', section)
        items = [i.strip() for i in items if len(i.strip()) > 10]
        return items[:25]

    def extract_interventions(self, nct_ids: list) -> pd.DataFrame:
        """
        Extract drug/intervention information.
        Filter to DRUG and BIOLOGICAL interventions only.
        """
        query = """
        SELECT
            nct_id,
            intervention_type,
            name as intervention_name,
            description as intervention_description
        FROM interventions
        WHERE nct_id = ANY(:nct_ids)
            AND intervention_type IN ('Drug', 'Biological', 'Combination Product')
        """
        return pd.read_sql(text(query), self.conn, params={"nct_ids": nct_ids})

    def extract_outcomes(self, nct_ids: list) -> pd.DataFrame:
        """
        Extract primary and secondary outcomes.
        This is the source of endpoint data — critical for endpoint benchmarking.
        """
        query = """
        SELECT
            nct_id,
            outcome_type,
            title as outcome_title,
            description as outcome_description,
            time_frame,
            population,
            units,
            units_analyzed
        FROM design_outcomes
        WHERE nct_id = ANY(:nct_ids)
        ORDER BY nct_id,
                 CASE outcome_type
                    WHEN 'Primary' THEN 1
                    WHEN 'Secondary' THEN 2
                    ELSE 3
                 END
        """
        return pd.read_sql(text(query), self.conn, params={"nct_ids": nct_ids})

    def extract_enrollment_actuals(self, nct_ids: list) -> pd.DataFrame:
        """
        Extract actual vs. planned enrollment.
        The calculated_values table has derived metrics including enrollment duration.

        Critical metric: actual_enrollment / planned_enrollment ratio.
        Ratio < 0.8 = recruitment challenge. Ratio > 1.1 = over-enrollment.
        """
        query = """
        SELECT
            cv.nct_id,
            cv.actual_duration,
            cv.months_to_report_results,
            cv.number_of_facilities,
            s.enrollment as planned_enrollment,
            rg.count as actual_participants
        FROM calculated_values cv
        JOIN studies s ON cv.nct_id = s.nct_id
        LEFT JOIN (
            SELECT nct_id, SUM(count) as count
            FROM result_groups
            WHERE result_type = 'Participant'
            GROUP BY nct_id
        ) rg ON cv.nct_id = rg.nct_id
        WHERE cv.nct_id = ANY(:nct_ids)
        """
        df = pd.read_sql(text(query), self.conn, params={"nct_ids": nct_ids})

        # Calculate enrollment ratio
        df['enrollment_ratio'] = (
            df['actual_participants'] / df['planned_enrollment'].replace(0, pd.NA)
        )
        df['recruitment_challenge_flag'] = df['enrollment_ratio'] < 0.80
        df['enrollment_shortfall'] = (
            df['planned_enrollment'] - df['actual_participants']
        ).clip(lower=0)
        return df

    def extract_dropout_data(self, nct_ids: list) -> pd.DataFrame:
        """
        Extract dropout and withdrawal data by period.
        Computes overall dropout rate per trial.
        """
        query = """
        SELECT
            nct_id,
            period,
            reason,
            count
        FROM drop_withdrawals
        WHERE nct_id = ANY(:nct_ids)
        """
        df = pd.read_sql(text(query), self.conn, params={"nct_ids": nct_ids})

        if len(df) > 0:
            total_dropouts = df.groupby('nct_id')['count'].sum().reset_index()
            total_dropouts.columns = ['nct_id', 'total_dropouts']
            return total_dropouts
        return pd.DataFrame(columns=['nct_id', 'total_dropouts'])

    def extract_site_countries(self, nct_ids: list) -> pd.DataFrame:
        """
        Extract countries where trials were conducted.
        Used for site selection benchmarking.
        """
        query = """
        SELECT
            nct_id,
            name as country,
            removed
        FROM countries
        WHERE nct_id = ANY(:nct_ids)
            AND removed = FALSE
        """
        return pd.read_sql(text(query), self.conn, params={"nct_ids": nct_ids})

    def extract_conditions(self, nct_ids: list) -> pd.DataFrame:
        """Extract disease/condition data with downcase normalization."""
        query = """
        SELECT nct_id, downcase_name as condition
        FROM conditions
        WHERE nct_id = ANY(:nct_ids)
        """
        return pd.read_sql(text(query), self.conn, params={"nct_ids": nct_ids})

    def extract_amendments(self, nct_ids: list) -> pd.DataFrame:
        """
        Estimate amendment count from study updates.
        AACT doesn't directly store amendment count, but we can proxy it
        from the number of significant protocol updates.
        """
        query = """
        SELECT
            nct_id,
            COUNT(*) as update_count
        FROM study_records
        WHERE nct_id = ANY(:nct_ids)
        GROUP BY nct_id
        """
        return pd.read_sql(text(query), self.conn, params={"nct_ids": nct_ids})

    def run_full_ingestion(
        self,
        batch_size: int = 1000,
        start_year: int = 2010,
        therapeutic_areas: list = None
    ) -> list:
        """
        Orchestrate the full ingestion pipeline.

        Process trials in batches to manage memory.
        Returns a list of TrialProfile dicts ready for ChromaDB ingestion.
        """
        self.connect()

        # Step 1: Get all qualifying trial IDs
        core_df = self.extract_core_trials(start_year=start_year)
        all_nct_ids = core_df['nct_id'].tolist()
        logger.info(f"Processing {len(all_nct_ids)} trials in batches of {batch_size}")

        all_profiles = []

        # Step 2: Process in batches
        for i in tqdm(range(0, len(all_nct_ids), batch_size), desc="Ingesting AACT batches"):
            batch_ids = all_nct_ids[i:i+batch_size]

            try:
                eligibility_df = self.extract_eligibility_criteria(batch_ids)
                interventions_df = self.extract_interventions(batch_ids)
                outcomes_df = self.extract_outcomes(batch_ids)
                enrollment_df = self.extract_enrollment_actuals(batch_ids)
                dropout_df = self.extract_dropout_data(batch_ids)
                countries_df = self.extract_site_countries(batch_ids)
                conditions_df = self.extract_conditions(batch_ids)
                amendments_df = self.extract_amendments(batch_ids)

                batch_core = core_df[core_df['nct_id'].isin(batch_ids)]

                for _, trial in batch_core.iterrows():
                    profile = self._build_trial_profile(
                        trial=trial,
                        eligibility=eligibility_df[
                            eligibility_df['nct_id'] == trial.nct_id
                        ],
                        interventions=interventions_df[
                            interventions_df['nct_id'] == trial.nct_id
                        ],
                        outcomes=outcomes_df[
                            outcomes_df['nct_id'] == trial.nct_id
                        ],
                        enrollment=enrollment_df[
                            enrollment_df['nct_id'] == trial.nct_id
                        ],
                        dropouts=dropout_df[
                            dropout_df['nct_id'] == trial.nct_id
                        ],
                        countries=countries_df[
                            countries_df['nct_id'] == trial.nct_id
                        ],
                        conditions=conditions_df[
                            conditions_df['nct_id'] == trial.nct_id
                        ],
                        amendments=amendments_df[
                            amendments_df['nct_id'] == trial.nct_id
                        ]
                    )
                    if profile:
                        all_profiles.append(profile)

            except Exception as e:
                logger.error(f"Error processing batch starting at {i}: {e}")
                continue

        logger.info(f"Successfully built {len(all_profiles)} trial profiles")
        return all_profiles

    def _build_trial_profile(
        self, trial, eligibility, interventions, outcomes,
        enrollment, dropouts, countries, conditions, amendments
    ) -> dict:
        """
        Consolidate all extracted data into a unified TrialProfile dict.
        This is the core data unit that gets stored in ChromaDB.
        """
        try:
            drug_names = interventions['intervention_name'].tolist() if len(interventions) else []

            primary_outcomes = outcomes[outcomes['outcome_type'] == 'Primary']
            primary_endpoint = primary_outcomes['outcome_title'].iloc[0] if len(primary_outcomes) else None
            primary_endpoint_timeframe = primary_outcomes['time_frame'].iloc[0] if len(primary_outcomes) else None

            secondary_outcomes = outcomes[outcomes['outcome_type'] == 'Secondary']
            secondary_endpoints = secondary_outcomes['outcome_title'].tolist()[:5]

            planned_enrollment = int(trial.enrollment) if pd.notna(trial.enrollment) else None

            enroll_row = enrollment.iloc[0] if len(enrollment) else None
            actual_enrollment = int(enroll_row['actual_participants']) if (
                enroll_row is not None and pd.notna(enroll_row.get('actual_participants'))
            ) else None

            enrollment_ratio = (
                actual_enrollment / planned_enrollment
                if (actual_enrollment and planned_enrollment and planned_enrollment > 0)
                else None
            )

            recruitment_challenge = (
                enrollment_ratio is not None and enrollment_ratio < 0.80
            )

            dropout_row = dropouts.iloc[0] if len(dropouts) else None
            total_dropouts = int(dropout_row['total_dropouts']) if dropout_row is not None else 0
            dropout_rate = (
                total_dropouts / actual_enrollment
                if (actual_enrollment and actual_enrollment > 0)
                else None
            )

            country_list = countries['country'].tolist()
            condition_list = conditions['condition'].tolist()[:5]

            elig_row = eligibility.iloc[0] if len(eligibility) else None
            inclusion_criteria = elig_row['inclusion_criteria'].tolist() if elig_row is not None else []
            exclusion_criteria = elig_row['exclusion_criteria'].tolist() if elig_row is not None else []
            min_age = elig_row['minimum_age'] if elig_row is not None else None
            max_age = elig_row['maximum_age'] if elig_row is not None else None
            gender = elig_row['gender'] if elig_row is not None else None

            duration_months = None
            if pd.notna(trial.start_date) and pd.notna(trial.completion_date):
                try:
                    start = pd.to_datetime(trial.start_date)
                    end = pd.to_datetime(trial.completion_date)
                    duration_months = max(0, int((end - start).days / 30))
                except Exception:
                    pass

            amendment_row = amendments.iloc[0] if len(amendments) else None
            amendment_count = int(amendment_row['update_count']) if amendment_row is not None else 0

            endpoint_type = self._classify_endpoint(primary_endpoint)

            profile = {
                "nct_id": trial.nct_id,
                "source": "AACT",
                "title": trial.brief_title or trial.official_title or "",
                "phase": str(trial.phase) if pd.notna(trial.phase) else "",
                "study_type": str(trial.study_type) if pd.notna(trial.study_type) else "",
                "status": str(trial.overall_status) if pd.notna(trial.overall_status) else "",
                "why_stopped": str(trial.why_stopped) if pd.notna(trial.why_stopped) else "",
                "allocation": str(trial.allocation) if pd.notna(trial.allocation) else "",
                "intervention_model": str(trial.intervention_model) if pd.notna(trial.intervention_model) else "",
                "masking": str(trial.masking) if pd.notna(trial.masking) else "",
                "primary_purpose": str(trial.primary_purpose) if pd.notna(trial.primary_purpose) else "",
                "number_of_arms": int(trial.number_of_arms) if pd.notna(trial.number_of_arms) else None,
                "is_fda_regulated": bool(trial.is_fda_regulated_drug),
                "drug_names": drug_names,
                "drug_names_str": ", ".join(drug_names),
                "conditions": condition_list,
                "conditions_str": ", ".join(condition_list),
                "primary_endpoint": primary_endpoint or "",
                "primary_endpoint_timeframe": primary_endpoint_timeframe or "",
                "endpoint_type": endpoint_type,
                "secondary_endpoints": secondary_endpoints,
                "secondary_endpoints_str": "; ".join(secondary_endpoints),
                "planned_enrollment": planned_enrollment,
                "actual_enrollment": actual_enrollment,
                "enrollment_ratio": enrollment_ratio,
                "recruitment_challenge_flag": recruitment_challenge,
                "enrollment_shortfall": (
                    planned_enrollment - actual_enrollment
                    if (planned_enrollment and actual_enrollment)
                    else None
                ),
                "number_of_sites": int(enroll_row['number_of_facilities']) if enroll_row is not None else None,
                "total_dropouts": total_dropouts,
                "dropout_rate": dropout_rate,
                "duration_months": duration_months,
                "start_year": int(pd.to_datetime(trial.start_date).year) if pd.notna(trial.start_date) else None,
                "countries": country_list,
                "countries_str": ", ".join(country_list),
                "num_countries": len(country_list),
                "min_age": min_age,
                "max_age": max_age,
                "gender": gender,
                "inclusion_criteria": inclusion_criteria,
                "exclusion_criteria": exclusion_criteria,
                "inclusion_count": len(inclusion_criteria),
                "exclusion_count": len(exclusion_criteria),
                "brief_summary": str(trial.brief_summary)[:2000] if pd.notna(trial.brief_summary) else "",
                "amendment_count": amendment_count,
                "high_amendment_flag": amendment_count > 3,
            }

            return profile

        except Exception as e:
            logger.error(f"Error building profile for {trial.nct_id}: {e}")
            return None

    def _classify_endpoint(self, endpoint_text: str) -> str:
        """
        Classify primary endpoint into standard categories.
        These categories are used for metadata filtering in retrieval.
        """
        if not endpoint_text:
            return "unknown"

        endpoint_lower = endpoint_text.lower()

        if any(t in endpoint_lower for t in ['overall survival', 'os ', 'mortality', 'death']):
            return "overall_survival"
        if any(t in endpoint_lower for t in ['progression-free', 'pfs', 'progression free']):
            return "progression_free_survival"
        if any(t in endpoint_lower for t in ['response rate', 'orr', 'objective response', 'tumor response']):
            return "response_rate"
        if any(t in endpoint_lower for t in ['disease-free', 'dfs', 'relapse-free']):
            return "disease_free_survival"
        if any(t in endpoint_lower for t in ['hba1c', 'blood glucose', 'glycemic']):
            return "glycemic_control"
        if any(t in endpoint_lower for t in ['pain', 'symptom', 'quality of life', 'qol', 'proms']):
            return "patient_reported_outcome"
        if any(t in endpoint_lower for t in ['safety', 'tolerability', 'adverse', 'mtt', 'mtd']):
            return "safety_tolerability"
        if any(t in endpoint_lower for t in ['pharmacokinetic', 'pk', 'auc', 'cmax', 'bioavailability']):
            return "pharmacokinetics"
        if any(t in endpoint_lower for t in ['biomarker', 'expression', 'mutation', 'ctdna']):
            return "biomarker"
        if any(t in endpoint_lower for t in ['hospitalization', 'readmission', 'hospital']):
            return "healthcare_utilization"

        return "composite_other"
