"""
Benchmark Builder: Constructs comparison tables from retrieved trial data.

Transforms raw retrieved trial chunks into structured benchmark tables
that can be rendered in reports or passed to the LLM as context.

Key benchmark tables:
1. Sample Size Distribution — percentile breakdown by phase/indication
2. Enrollment Rate Table — patients/month by country
3. Endpoint Frequency Table — which endpoints dominate in an indication
4. Dropout Rate Table — by phase and therapeutic area
5. Site Count vs. Enrollment Rate — efficiency analysis
"""

import statistics
from collections import defaultdict
from loguru import logger


class BenchmarkBuilder:
    """
    Builds structured benchmark tables from retrieved trial data.
    """

    def build_sample_size_benchmark(self, candidates: list) -> dict:
        """
        Build a sample size distribution table from retrieved trials.

        Returns:
        {
            'summary': dict with median, mean, p25, p75, min, max
            'by_status': breakdown of sizes for completed vs. terminated
            'distribution': list of (nct_id, N, status) tuples
            'table_text': formatted ASCII table for LLM context
        }
        """
        sizes = []
        completed_sizes = []
        terminated_sizes = []
        distribution = []

        for candidate in candidates:
            meta = candidate.get('metadata', {})
            text = candidate.get('text', '')

            # Try metadata first, then parse from text
            actual_n = meta.get('actual_enrollment', 0)
            planned_n = meta.get('planned_enrollment', 0)
            status = meta.get('status', 'unknown')
            nct_id = meta.get('nct_id', 'unknown')

            n = actual_n or planned_n
            if n and n > 0:
                sizes.append(n)
                distribution.append({
                    'nct_id': nct_id,
                    'n': n,
                    'status': status,
                    'planned': planned_n,
                    'actual': actual_n
                })
                if 'completed' in status.lower():
                    completed_sizes.append(n)
                elif 'terminated' in status.lower():
                    terminated_sizes.append(n)

        if not sizes:
            return {'error': 'No enrollment data in retrieved trials'}

        sizes.sort()

        def safe_stats(vals):
            if not vals:
                return {}
            return {
                'median': statistics.median(vals),
                'mean': round(statistics.mean(vals)),
                'min': min(vals),
                'max': max(vals),
                'p25': vals[len(vals) // 4] if len(vals) >= 4 else vals[0],
                'p75': vals[3 * len(vals) // 4] if len(vals) >= 4 else vals[-1],
                'n_trials': len(vals)
            }

        summary = safe_stats(sizes)
        completed_stats = safe_stats(sorted(completed_sizes))
        terminated_stats = safe_stats(sorted(terminated_sizes))

        # Build ASCII table
        table_lines = [
            "SAMPLE SIZE BENCHMARK",
            f"{'Metric':<30} {'All Trials':>15} {'Completed':>12} {'Terminated':>12}",
            "-" * 72,
            f"{'N (trials)':<30} {summary.get('n_trials', 0):>15} {completed_stats.get('n_trials', 0):>12} {terminated_stats.get('n_trials', 0):>12}",
            f"{'Median N':<30} {summary.get('median', 'N/A'):>15} {completed_stats.get('median', 'N/A'):>12} {terminated_stats.get('median', 'N/A'):>12}",
            f"{'Mean N':<30} {summary.get('mean', 'N/A'):>15} {completed_stats.get('mean', 'N/A'):>12} {terminated_stats.get('mean', 'N/A'):>12}",
            f"{'25th Percentile':<30} {summary.get('p25', 'N/A'):>15} {completed_stats.get('p25', 'N/A'):>12} {terminated_stats.get('p25', 'N/A'):>12}",
            f"{'75th Percentile':<30} {summary.get('p75', 'N/A'):>15} {completed_stats.get('p75', 'N/A'):>12} {terminated_stats.get('p75', 'N/A'):>12}",
            f"{'Min N':<30} {summary.get('min', 'N/A'):>15} {completed_stats.get('min', 'N/A'):>12} {terminated_stats.get('min', 'N/A'):>12}",
            f"{'Max N':<30} {summary.get('max', 'N/A'):>15} {completed_stats.get('max', 'N/A'):>12} {terminated_stats.get('max', 'N/A'):>12}",
        ]

        return {
            'summary': summary,
            'by_status': {
                'completed': completed_stats,
                'terminated': terminated_stats
            },
            'distribution': distribution[:20],  # Top 20
            'table_text': "\n".join(table_lines)
        }

    def build_enrollment_rate_benchmark(self, candidates: list) -> dict:
        """
        Build enrollment rate benchmarks (patients/month) from retrieved trials.
        """
        rates = []
        by_country = defaultdict(list)

        for candidate in candidates:
            meta = candidate.get('metadata', {})

            actual_n = meta.get('actual_enrollment', 0) or meta.get('enrollment_ratio', 0)
            duration = meta.get('duration_months', 0)
            countries = meta.get('countries_str', '')

            if actual_n and duration and duration > 0:
                rate = actual_n / duration
                rates.append(rate)

                if countries:
                    for country in countries.split(','):
                        country = country.strip()
                        if country:
                            by_country[country].append(rate)

        if not rates:
            return {'error': 'No enrollment rate data available'}

        rates.sort()
        median_rate = statistics.median(rates)

        # Country rankings
        country_rankings = []
        for country, country_rates in by_country.items():
            if len(country_rates) >= 3:  # Need at least 3 data points
                country_rankings.append({
                    'country': country,
                    'median_rate': round(statistics.median(country_rates), 1),
                    'n_trials': len(country_rates)
                })

        country_rankings.sort(key=lambda x: x['median_rate'], reverse=True)

        return {
            'overall_median_rate': round(median_rate, 1),
            'overall_mean_rate': round(statistics.mean(rates), 1),
            'n_trials': len(rates),
            'country_rankings': country_rankings[:10],
        }

    def build_endpoint_frequency_table(self, candidates: list) -> dict:
        """
        Build endpoint frequency table showing which endpoints dominate.
        """
        endpoint_counts = defaultdict(int)
        endpoint_success = defaultdict(lambda: {'completed': 0, 'terminated': 0})
        total = 0

        for candidate in candidates:
            meta = candidate.get('metadata', {})
            endpoint_type = meta.get('endpoint_type', 'unknown')
            status = meta.get('status', '')

            if endpoint_type and endpoint_type != 'unknown':
                endpoint_counts[endpoint_type] += 1
                total += 1
                if 'completed' in status.lower():
                    endpoint_success[endpoint_type]['completed'] += 1
                elif 'terminated' in status.lower():
                    endpoint_success[endpoint_type]['terminated'] += 1

        if not endpoint_counts:
            return {'error': 'No endpoint data in retrieved trials'}

        # Build frequency table
        table_rows = []
        for endpoint, count in sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True):
            pct = round(100 * count / total, 1) if total > 0 else 0
            success_data = endpoint_success[endpoint]
            n_completed = success_data['completed']
            n_terminated = success_data['terminated']
            success_rate = (
                round(100 * n_completed / (n_completed + n_terminated), 0)
                if (n_completed + n_terminated) > 0 else 'N/A'
            )

            table_rows.append({
                'endpoint_type': endpoint.replace('_', ' ').title(),
                'count': count,
                'percentage': pct,
                'completed': n_completed,
                'terminated': n_terminated,
                'success_rate': success_rate
            })

        # ASCII table
        table_lines = [
            "ENDPOINT FREQUENCY TABLE",
            f"{'Endpoint Type':<30} {'Count':>8} {'%':>6} {'Completed':>10} {'Terminated':>12} {'Success%':>10}",
            "-" * 80,
        ]
        for row in table_rows:
            table_lines.append(
                f"{row['endpoint_type']:<30} {row['count']:>8} {row['percentage']:>6.1f} "
                f"{row['completed']:>10} {row['terminated']:>12} {str(row['success_rate']):>9}%"
            )

        return {
            'endpoint_frequency': table_rows,
            'total_trials': total,
            'table_text': "\n".join(table_lines)
        }

    def build_dropout_benchmark(self, candidates: list) -> dict:
        """
        Build dropout rate benchmark from retrieved trial data.
        """
        dropout_rates = []
        by_phase = defaultdict(list)

        for candidate in candidates:
            meta = candidate.get('metadata', {})
            dropout_rate = meta.get('dropout_rate', None)
            phase = meta.get('phase', 'unknown')

            if dropout_rate and 0 <= dropout_rate <= 1:
                dropout_rates.append(dropout_rate * 100)  # Convert to percentage
                by_phase[phase].append(dropout_rate * 100)

        if not dropout_rates:
            return {'error': 'No dropout rate data in retrieved trials'}

        dropout_rates.sort()
        median_dropout = statistics.median(dropout_rates)

        # Phase breakdown
        phase_summary = {}
        for phase, rates in by_phase.items():
            if rates:
                phase_summary[phase] = {
                    'median_pct': round(statistics.median(rates), 1),
                    'mean_pct': round(statistics.mean(rates), 1),
                    'n_trials': len(rates)
                }

        return {
            'overall_median_dropout_pct': round(median_dropout, 1),
            'overall_mean_dropout_pct': round(statistics.mean(dropout_rates), 1),
            'p25_dropout_pct': round(dropout_rates[len(dropout_rates) // 4], 1) if len(dropout_rates) >= 4 else None,
            'p75_dropout_pct': round(dropout_rates[3 * len(dropout_rates) // 4], 1) if len(dropout_rates) >= 4 else None,
            'n_trials': len(dropout_rates),
            'by_phase': phase_summary
        }

    def build_comprehensive_scorecard(
        self,
        candidates: list,
        protocol_context: dict = None
    ) -> dict:
        """
        Build a comprehensive scorecard comparing a protocol to benchmarks.
        """
        sample_size_bench = self.build_sample_size_benchmark(candidates)
        dropout_bench = self.build_dropout_benchmark(candidates)
        endpoint_bench = self.build_endpoint_frequency_table(candidates)

        scorecard = {
            'sample_size': sample_size_bench,
            'dropout': dropout_bench,
            'endpoints': endpoint_bench,
        }

        if protocol_context:
            # Add protocol-specific risk flags
            flags = []

            planned_n = protocol_context.get('planned_enrollment', 0)
            if planned_n and 'summary' in sample_size_bench:
                median_n = sample_size_bench['summary'].get('median', 0)
                if median_n and planned_n < sample_size_bench['summary'].get('p25', 0):
                    flags.append({
                        'dimension': 'Sample Size',
                        'risk': 'HIGH',
                        'note': f"Planned N={planned_n} is below 25th percentile (N={sample_size_bench['summary']['p25']}) of comparable trials"
                    })

            dropout_assumption = protocol_context.get('dropout_assumption', 0)
            if dropout_assumption and 'overall_median_dropout_pct' in dropout_bench:
                historical_median = dropout_bench['overall_median_dropout_pct']
                assumed_pct = dropout_assumption * 100
                if assumed_pct < historical_median - 5:
                    flags.append({
                        'dimension': 'Dropout Assumption',
                        'risk': 'MEDIUM',
                        'note': f"Assumed dropout {assumed_pct:.0f}% is optimistic vs historical median {historical_median:.0f}%"
                    })

            scorecard['risk_flags'] = flags

        return scorecard
